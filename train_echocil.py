"""Train and evaluate ECHO-CIL on the processed CWRU shards.

Run from this directory:
    python train_echocil.py --data echo_cil_processed --epochs 5 --batch-size 64
"""
from __future__ import annotations

import argparse, json, random
from collections import Counter
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from echo_cil.data import DictDataset, EchoCILDataset, collate_numpy
from echo_cil.guard import SubspaceGuard
from echo_cil.memory import ReplayBuffer
from echo_cil.model import ECHOModel


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    """Resolve auto/cpu/cuda requests and fail clearly for invalid CUDA use."""
    requested = requested.lower()
    if requested == 'auto':
        if torch.cuda.is_available():
            requested = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            requested = 'mps'
        else:
            requested = 'cpu'
    if requested.startswith('cuda'):
        if not torch.cuda.is_available():
            raise RuntimeError(
                'CUDA was requested but this PyTorch environment cannot use it. '
                'Install a CUDA-enabled PyTorch build and a compatible NVIDIA '
                'driver, or run with --device cpu/auto.')
        device = torch.device(requested)
        index = device.index if device.index is not None else 0
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f'CUDA device {index} does not exist; available device count is '
                f'{torch.cuda.device_count()}.')
    else:
        device = torch.device(requested)
    return device


def amp_context(device: torch.device, enabled: bool):
    if enabled:
        return torch.autocast(device_type='cuda', dtype=torch.float16)
    return nullcontext()


def loader_options(device: torch.device, num_workers: int):
    options = {'num_workers': num_workers,
               'pin_memory': device.type == 'cuda'}
    if num_workers > 0:
        options['persistent_workers'] = True
    return options


def to_device(batch, device, non_blocking=False):
    tensors = {}
    for key, value in batch.items():
        if not isinstance(value, np.ndarray):
            continue
        tensor = torch.from_numpy(value)
        # collate_numpy intentionally returns NumPy arrays.  Pin the completed
        # batch here so CUDA copies can overlap with compute.
        if non_blocking and not tensor.is_pinned():
            tensor = tensor.pin_memory()
        tensors[key] = tensor.to(device, non_blocking=non_blocking)
    return tensors


def balanced_sampler(dataset, old_classes, new_classes):
    """Balance classes and, during increments, old/new sample mass."""
    labels = [int(dataset[i]['y']) for i in range(len(dataset))]
    counts = Counter(labels)
    old_set, new_set = set(old_classes), set(new_classes)
    weights = []
    for y in labels:
        if old_set:
            group_classes = old_set if y in old_set else new_set
            weight = 0.5 / max(1, len(group_classes)) / counts[y]
        else:
            weight = 1.0 / max(1, len(new_set)) / counts[y]
        weights.append(weight)
    return WeightedRandomSampler(torch.as_tensor(weights, dtype=torch.double),
                                 num_samples=len(weights), replacement=True)


@torch.no_grad()
def restore_old_rows(model, old_model):
    """Keep old classifier and router rows exactly immutable."""
    if old_model is None:
        return
    n_classes = old_model.num_classes
    n_experts = len(old_model.experts)
    model.classifier.weight[:n_classes].copy_(old_model.classifier.weight)
    model.classifier.bias[:n_classes].copy_(old_model.classifier.bias)
    model.expert_gate.weight[:n_experts].copy_(old_model.expert_gate.weight)
    model.expert_gate.bias[:n_experts].copy_(old_model.expert_gate.bias)


@torch.no_grad()
def extract_features(model, dataset, device, batch_size=256, num_workers=0):
    """Extract frozen features in FP32 for reproducible head selection."""
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_numpy,
                        **loader_options(device, num_workers))
    model.eval(); features, labels = [], []
    for raw in loader:
        b = to_device(raw, device, non_blocking=device.type == 'cuda')
        # Deliberately do not use autocast: the final errors are boundary cases
        # where FP16 rounding can change the predicted class.
        _, z = model(b['x'], b['order'], b['quality'])
        features.append(z.detach().float().cpu())
        labels.append(b['y'].detach().cpu())
    return torch.cat(features).to(torch.float64), torch.cat(labels)


def fit_logistic_head(z, y, num_classes, weight_decay, class_balanced,
                      max_iter=200):
    """Fit one deterministic convex candidate and return raw-space weights."""
    mean = z.mean(dim=0)
    std = z.std(dim=0).clamp_min(1e-5)
    standardized = (z - mean) / std
    head = nn.Linear(z.shape[1], num_classes, dtype=torch.float64)
    nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
    optimizer = torch.optim.LBFGS(
        head.parameters(), lr=1.0, max_iter=max_iter,
        tolerance_grad=1e-10, tolerance_change=1e-12,
        line_search_fn='strong_wolfe')

    sample_weights = None
    if class_balanced:
        counts = torch.bincount(y, minlength=num_classes).to(torch.float64)
        class_weights = counts.sum() / counts.clamp_min(1.0)
        class_weights = class_weights / class_weights.mean()
        sample_weights = class_weights[y]

    def closure():
        optimizer.zero_grad()
        losses = nn.functional.cross_entropy(
            head(standardized), y, reduction='none')
        loss = ((losses * sample_weights).mean()
                if sample_weights is not None else losses.mean())
        loss = loss + 0.5 * float(weight_decay) * head.weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        raw_weight = head.weight / std.unsqueeze(0)
        raw_bias = head.bias - (head.weight * mean.unsqueeze(0)
                                / std.unsqueeze(0)).sum(dim=1)
    return raw_weight, raw_bias


@torch.no_grad()
def score_head(weight, bias, z, y, classes, old_classes):
    pred = (z @ weight.t() + bias).argmax(dim=1)
    per_class = {}
    for c in classes:
        mask = y == c
        per_class[c] = float((pred[mask] == y[mask]).double().mean())
    new_classes = [c for c in classes if c not in old_classes]
    group_mean = lambda cs: float(np.mean([per_class[c] for c in cs])) if cs else 0.0
    old_acc = group_mean(old_classes)
    new_acc = group_mean(new_classes)
    macro = group_mean(classes)
    overall = float((pred == y).double().mean())
    metrics = {'overall': overall, 'macro': macro, 'old': old_acc,
               'new': new_acc, 'worst_class': min(per_class.values())}
    # Protect both sides of class-incremental learning first, then macro and
    # worst-class performance. Test labels are never consulted here.
    score = (min(old_acc, new_acc), macro, metrics['worst_class'], overall)
    return score, metrics


def calibrate_classifier(model, train_dataset, validation_dataset, classes,
                         old_classes, device, batch_size=256,
                         weight_decays=(1e-4,), max_iter=200, num_workers=0):
    """Select the final linear head using validation data only.

    Candidates include the SGD head plus balanced and ordinary logistic heads
    across several regularization strengths. Only the selected linear matrix
    is retained, so inference cost remains unchanged.
    """
    if not weight_decays or len(train_dataset) == 0:
        return {'method': 'disabled'}
    train_z, train_y = extract_features(
        model, train_dataset, device, batch_size, num_workers)
    val_z, val_y = extract_features(
        model, validation_dataset, device, batch_size, num_workers)

    original_weight = model.classifier.weight.detach().cpu().to(torch.float64)
    original_bias = model.classifier.bias.detach().cpu().to(torch.float64)
    best_score, best_metrics = score_head(
        original_weight, original_bias, val_z, val_y, classes, old_classes)
    best = {'method': 'sgd', 'weight_decay': None,
            'class_balanced': False, **best_metrics}
    best_weight, best_bias = original_weight, original_bias

    for class_balanced in (True, False):
        for weight_decay in weight_decays:
            weight, bias = fit_logistic_head(
                train_z, train_y, model.num_classes, weight_decay,
                class_balanced, max_iter=max_iter)
            candidate_score, metrics = score_head(
                weight, bias, val_z, val_y, classes, old_classes)
            if candidate_score > best_score:
                best_score = candidate_score
                best_weight, best_bias = weight, bias
                best = {'method': 'logistic',
                        'weight_decay': float(weight_decay),
                        'class_balanced': class_balanced, **metrics}

    with torch.no_grad():
        model.classifier.weight.copy_(best_weight.to(
            device=device, dtype=model.classifier.weight.dtype))
        model.classifier.bias.copy_(best_bias.to(
            device=device, dtype=model.classifier.bias.dtype))
    return best


@torch.no_grad()
def evaluate(model, dataset, classes, device, old_classes=None, batch_size=128,
             use_amp=False, num_workers=0):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_numpy,
                        **loader_options(device, num_workers))
    model.eval(); hit = total = 0; per = {c: [0, 0] for c in classes}
    class_index = {c: i for i, c in enumerate(classes)}
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for raw in loader:
        b = to_device(raw, device, non_blocking=device.type == 'cuda')
        with amp_context(device, use_amp):
            logits, _ = model(b['x'], b['order'], b['quality'])
        pred = logits.argmax(1).cpu().numpy(); y = b['y'].cpu().numpy()
        hit += int((pred == y).sum()); total += len(y)
        for p, t in zip(pred, y):
            per[int(t)][0] += int(p == t); per[int(t)][1] += 1
            if int(p) in class_index:
                confusion[class_index[int(t)], class_index[int(p)]] += 1
    old_classes = old_classes or []
    new_classes = [c for c in classes if c not in old_classes]
    mean = lambda cs: float(np.mean([per[c][0] / max(1, per[c][1]) for c in cs])) if cs else 0.0
    return {'overall': hit / max(1, total), 'macro': mean(classes),
            'old': mean(old_classes), 'new': mean(new_classes),
            'per_class': {str(c): per[c][0] / max(1, per[c][1]) for c in classes},
            'confusion': confusion.tolist()}


def main():
    ap = argparse.ArgumentParser()
    # PyCharm 右键运行时通常不会自动填写命令行参数，因此默认读取
    # 当前工作目录下的 echo_cil_processed；命令行仍可通过 --data 覆盖。
    ap.add_argument('--data', type=Path, default=Path('echo_cil_processed'))
    ap.add_argument('--output', type=Path, default=Path('runs/echo_cil'))
    ap.add_argument('--epochs', type=int, default=10); ap.add_argument('--batch-size', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-3); ap.add_argument('--seed', type=int, default=2027)
    ap.add_argument('--weight-decay', type=float, default=1e-4)
    ap.add_argument('--temperature', type=float, default=2.0)
    ap.add_argument('--kd-weight', type=float, default=0.5)
    ap.add_argument('--guard-strength', type=float, default=1.0)
    ap.add_argument('--calibration-weight-decay', type=float, default=1e-4,
                    help='preferred final calibration strength; <0 disables it')
    ap.add_argument('--calibration-grid', type=float, nargs='+',
                    default=[1e-5, 1e-4, 1e-3],
                    help='regularization candidates selected on validation')
    ap.add_argument('--calibration-max-iter', type=int, default=200)
    ap.add_argument('--memory-mb', type=int, default=32)
    ap.add_argument('--device', default='auto',
                    help='auto, cpu, cuda, cuda:0, ... (default: auto)')
    ap.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True,
                    help='use CUDA mixed precision (default: enabled on CUDA)')
    ap.add_argument('--num-workers', type=int, default=0,
                    help='DataLoader worker processes; 0 is safest on Windows')
    ap.add_argument('--deterministic', action='store_true',
                    help='prefer deterministic CUDA algorithms over speed')
    args = ap.parse_args()
    if not args.data.exists():
        # Support both layouts: scripts/data at repository root, or inside
        # the CWRU轴承数据 subdirectory opened by PyCharm.
        candidates = [Path(__file__).resolve().parent / args.data,
                      Path(__file__).resolve().parent / 'CWRU轴承数据' / args.data]
        for candidate in candidates:
            if candidate.exists():
                args.data = candidate
                break
    if not args.data.exists():
        raise FileNotFoundError(f'Processed data directory not found: {args.data}')
    if args.num_workers < 0:
        raise ValueError('--num-workers must be >= 0')
    device = resolve_device(args.device)
    use_amp = bool(args.amp and device.type == 'cuda')
    seed_everything(args.seed)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = not args.deterministic
        torch.backends.cudnn.deterministic = args.deterministic
        if args.deterministic:
            torch.use_deterministic_algorithms(True, warn_only=True)
        torch.set_float32_matmul_precision('high')
        print(f'Using CUDA device: {torch.cuda.get_device_name(device)}; '
              f'AMP={use_amp}; workers={args.num_workers}')
    else:
        print(f'Using device: {device}; AMP=False; workers={args.num_workers}')
    args.output.mkdir(parents=True, exist_ok=True)
    order = [[0,1,2,3], [4,5,6], [7,8,9]]
    model = ECHOModel(num_classes=4).to(device); memory = ReplayBuffer(args.memory_mb * 1024 * 1024, args.seed)
    stage_results = []
    for stage, new_classes in enumerate(order):
        # Snapshot the actual previous-stage model before changing its expert
        # count or classifier size.  The former implementation copied after
        # expansion, so old_model.num_classes included the new random rows and
        # those rows were reset after every epoch.
        old_model = model.frozen_copy() if stage > 0 else None
        old_num_classes = old_model.num_classes if old_model is not None else 0
        if stage > 0:
            model.add_expert()
            model.expand_classes(max(new_classes) + 1)
        active = [c for cs in order[:stage + 1] for c in cs]
        old_classes = [c for cs in order[:stage] for c in cs]
        new_ds = EchoCILDataset(args.data, 'train', new_classes)
        val_ds = EchoCILDataset(args.data, 'val', active)
        test_ds = EchoCILDataset(args.data, 'test', active)
        # At incremental stages, only new-class data plus the fixed-budget
        # memory are visible.  This is the actual class-incremental protocol.
        old_items = memory.all()
        new_items = [new_ds[i] for i in range(len(new_ds))]
        train_ds = DictDataset(old_items + new_items)
        guard = SubspaceGuard(rank=16, strength=args.guard_strength)
        if old_model is not None and old_items:
            old_loader = DataLoader(
                DictDataset(old_items), batch_size=args.batch_size,
                shuffle=False, collate_fn=collate_numpy,
                **loader_options(device, args.num_workers))
            with torch.no_grad():
                features = []
                for raw in old_loader:
                    b = to_device(raw, device,
                                  non_blocking=device.type == 'cuda')
                    features.append(old_model(b['x'], b['order'], b['quality'])[1])
                old_features = torch.cat(features)
            guard.fit(old_features)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=args.lr, weight_decay=args.weight_decay)
        sampler = balanced_sampler(train_ds, old_classes, new_classes)
        loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                            collate_fn=collate_numpy,
                            **loader_options(device, args.num_workers))
        scaler = torch.amp.GradScaler('cuda') if use_amp else None
        for epoch in range(args.epochs):
            model.train()
            for raw in loader:
                b = to_device(raw, device,
                              non_blocking=device.type == 'cuda')
                with amp_context(device, use_amp):
                    logits, z = model(b['x'], b['order'], b['quality'])
                    loss = nn.functional.cross_entropy(logits, b['y'])
                    if old_model is not None:
                        # Replay examples are the reliable anchor for old-class
                        # distillation and subspace protection.  Applying these
                        # losses to unseen new classes unnecessarily blocks
                        # plasticity.
                        old_mask = b['y'] < old_num_classes
                        if old_mask.any():
                            with torch.no_grad():
                                old_logits, old_z = old_model(
                                    b['x'][old_mask], b['order'][old_mask],
                                    b['quality'][old_mask])
                            temperature = args.temperature
                            kd = nn.functional.kl_div(
                                nn.functional.log_softmax(
                                    logits[old_mask, :old_num_classes]
                                    / temperature, 1),
                                nn.functional.softmax(old_logits / temperature, 1),
                                reduction='batchmean') * temperature ** 2
                            loss = loss + args.kd_weight * kd
                            loss = loss + guard.penalty(old_z, z[old_mask])
                optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                    optimizer.step()
                # Restore after every optimizer step.  Only old rows are
                # immutable; newly appended classifier/router rows keep their
                # learned updates.
                restore_old_rows(model, old_model)
        # Consolidate only after the final increment.  Feeding an intermediate
        # calibrated head back into the next teacher changes the representation
        # learning trajectory and can reduce final new-class accuracy.  Final-
        # only calibration preserves the proven incremental trajectory while
        # removing its last classifier bias at deployment time.
        applied_calibration = (args.calibration_weight_decay
                               if stage == len(order) - 1 else -1.0)
        calibration = {'method': 'not_final_stage'}
        if applied_calibration >= 0:
            weight_decays = list(dict.fromkeys(
                [applied_calibration] + args.calibration_grid))
            calibration = calibrate_classifier(
                model, train_ds, val_ds, active, old_classes, device,
                batch_size=max(args.batch_size, 256),
                weight_decays=weight_decays,
                max_iter=args.calibration_max_iter,
                num_workers=args.num_workers)
            print(f'Final head selected on validation: {calibration}')
        result = evaluate(model, val_ds, active, device, old_classes,
                          use_amp=False, num_workers=args.num_workers)
        test_result = evaluate(model, test_ds, active, device, old_classes,
                               use_amp=False, num_workers=args.num_workers)
        memory_before = memory.bytes_used
        # ReplayBuffer.update already merges existing exemplars.  Passing only
        # this stage's new data prevents the old memory from being duplicated.
        memory.update(new_items, active)
        stage_results.append({'stage': stage, 'classes': active, 'val': result,
                              'test': test_result,
                              'device': str(device), 'amp': use_amp,
                              'calibration_weight_decay': applied_calibration,
                              'calibration': calibration,
                              'memory_bytes_before': memory_before,
                              'memory_bytes': memory.bytes_used})
        torch.save({'stage': stage, 'model': model.state_dict(), 'classes': active}, args.output / f'stage_{stage}.pt')
    (args.output / 'results.json').write_text(json.dumps(stage_results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(stage_results, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
