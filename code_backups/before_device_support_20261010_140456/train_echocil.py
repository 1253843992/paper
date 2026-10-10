"""Train and evaluate ECHO-CIL on the processed CWRU shards.

Run from this directory:
    python train_echocil.py --data echo_cil_processed --epochs 5 --batch-size 64
"""
from __future__ import annotations

import argparse, json, random
from collections import Counter
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


def to_device(batch, device):
    return {k: torch.from_numpy(v).to(device) for k, v in batch.items() if isinstance(v, np.ndarray)}


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


def calibrate_classifier(model, dataset, device, batch_size=256,
                         weight_decay=1e-4, max_iter=200):
    """Fit a balanced logistic head on standardized frozen features.

    Incremental SGD learns a useful representation, but its classifier is
    biased by the changing old/new optimization objective.  The replay set is
    class balanced, so a regularized logistic consolidation at deployment gives
    every active class an equal opportunity without using validation or test
    labels.  The fitted matrix is converted back to raw-feature coordinates and
    copied into ``nn.Linear``, adding no inference-time component or overhead.
    """
    if weight_decay < 0 or len(dataset) == 0:
        return
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_numpy)
    model.eval(); features, labels = [], []
    with torch.no_grad():
        for raw in loader:
            b = to_device(raw, device)
            _, z = model(b['x'], b['order'], b['quality'])
            features.append(z.detach().cpu())
            labels.append(b['y'].detach().cpu())

    # Standardization makes the regularization independent of individual
    # feature scales.  Float64 and deterministic zero initialization make the
    # small convex optimization stable across CPU/GPU training runs.
    z = torch.cat(features).to(torch.float64)
    y = torch.cat(labels)
    mean = z.mean(dim=0)
    std = z.std(dim=0).clamp_min(1e-5)
    standardized = (z - mean) / std
    head = nn.Linear(z.shape[1], model.num_classes, dtype=torch.float64)
    nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
    optimizer = torch.optim.LBFGS(
        head.parameters(), lr=1.0, max_iter=max_iter,
        tolerance_grad=1e-10, tolerance_change=1e-12,
        line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(head(standardized), y)
        loss = loss + 0.5 * float(weight_decay) * head.weight.pow(2).sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    with torch.no_grad():
        raw_weight = head.weight / std.unsqueeze(0)
        raw_bias = head.bias - (head.weight * mean.unsqueeze(0)
                                / std.unsqueeze(0)).sum(dim=1)
        model.classifier.weight.copy_(raw_weight.to(
            device=device, dtype=model.classifier.weight.dtype))
        model.classifier.bias.copy_(raw_bias.to(
            device=device, dtype=model.classifier.bias.dtype))


@torch.no_grad()
def evaluate(model, dataset, classes, device, old_classes=None, batch_size=128):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_numpy)
    model.eval(); hit = total = 0; per = {c: [0, 0] for c in classes}
    class_index = {c: i for i, c in enumerate(classes)}
    confusion = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for raw in loader:
        b = to_device(raw, device); logits, _ = model(b['x'], b['order'], b['quality'])
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
                    help='final balanced logistic calibration; <0 disables it')
    ap.add_argument('--memory-mb', type=int, default=32); ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
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
    seed_everything(args.seed); device = torch.device(args.device)
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
            old_loader = DataLoader(DictDataset(old_items), batch_size=args.batch_size, shuffle=False, collate_fn=collate_numpy)
            with torch.no_grad():
                features = []
                for raw in old_loader:
                    b = to_device(raw, device)
                    features.append(old_model(b['x'], b['order'], b['quality'])[1])
                old_features = torch.cat(features)
            guard.fit(old_features)
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                      lr=args.lr, weight_decay=args.weight_decay)
        sampler = balanced_sampler(train_ds, old_classes, new_classes)
        loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler,
                            collate_fn=collate_numpy)
        for epoch in range(args.epochs):
            model.train()
            for raw in loader:
                b = to_device(raw, device); logits, z = model(b['x'], b['order'], b['quality'])
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
                                logits[old_mask, :old_num_classes] / temperature, 1),
                            nn.functional.softmax(old_logits / temperature, 1),
                            reduction='batchmean') * temperature ** 2
                        loss = loss + args.kd_weight * kd
                        loss = loss + guard.penalty(old_z, z[old_mask])
                optimizer.zero_grad(); loss.backward()
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
        if applied_calibration >= 0:
            calibrate_classifier(model, train_ds, device,
                                 batch_size=max(args.batch_size, 256),
                                 weight_decay=applied_calibration)
        result = evaluate(model, val_ds, active, device, old_classes)
        test_result = evaluate(model, test_ds, active, device, old_classes)
        memory_before = memory.bytes_used
        # ReplayBuffer.update already merges existing exemplars.  Passing only
        # this stage's new data prevents the old memory from being duplicated.
        memory.update(new_items, active)
        stage_results.append({'stage': stage, 'classes': active, 'val': result,
                              'test': test_result,
                              'calibration_weight_decay': applied_calibration,
                              'memory_bytes_before': memory_before,
                              'memory_bytes': memory.bytes_used})
        torch.save({'stage': stage, 'model': model.state_dict(), 'classes': active}, args.output / f'stage_{stage}.pt')
    (args.output / 'results.json').write_text(json.dumps(stage_results, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(stage_results, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
