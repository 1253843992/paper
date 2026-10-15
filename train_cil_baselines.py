"""Run a single 1-D CIL comparison. No ECHO-CIL trainer changes are required."""
from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import io
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader

from echo_cil.baselines import (METHODS, CILModel, CosineHead, objective,
                                effective_weights, kd_loss)
from echo_cil.data import EchoCILDataset, DictDataset, collate_numpy
from echo_cil.memory import ReplayBuffer
from echo_cil.physics import load_physics_stats
from train_echocil import (ORDERS, resolve_device, seed_everything, amp_context,
                          to_device, experiment_summary)


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def initialize_device(requested):
    device = resolve_device(requested)
    if device.type == 'cuda':
        index = device.index if device.index is not None else torch.cuda.current_device()
        count = torch.cuda.device_count()
        if not 0 <= index < count:
            raise ValueError(f'CUDA device index {index} is out of range; {count} visible device(s)')
        device = torch.device('cuda', index)
        torch.cuda.set_device(index)
        # Allocator statistics may reject a valid index before lazy CUDA init.
        # Force a real allocation before the first reset_peak_memory_stats call.
        probe = torch.empty(1, device=device)
        del probe
    return device


def loader(items, args, shuffle=False):
    # Evaluation is single-process. Training workers are nonpersistent and fully
    # exhausted each epoch: no accumulating persistent worker/pipe pools.
    return DataLoader(DictDataset(items), batch_size=args.batch_size,
                      shuffle=shuffle, num_workers=args.num_workers if shuffle else 0,
                      persistent_workers=False, pin_memory=args.device.type == 'cuda',
                      collate_fn=collate_numpy,
                      generator=torch.Generator().manual_seed(args.seed))


def materialize(items, stats):
    dataset = DictDataset(items, stats)
    return [dataset[index] for index in range(len(dataset))]


def targets_for(labels, classes):
    mapping = torch.full((max(classes)+1,), -1, device=labels.device, dtype=torch.long)
    mapping[torch.as_tensor(classes, device=labels.device)] = torch.arange(len(classes), device=labels.device)
    result = mapping[labels]
    if (result < 0).any():
        raise ValueError('Label outside active class set')
    return result


@torch.no_grad()
def features_of(model, items, args):
    model.eval()
    return torch.cat([F.normalize(model(to_device(raw, args.device))['features'].float(), dim=1).cpu()
                      for raw in loader(items, args)], dim=0)


def select_herding(features, quota):
    mean = features.mean(0)
    chosen, running = [], torch.zeros_like(mean)
    available = torch.ones(len(features), dtype=torch.bool)
    for k in range(min(quota, len(features))):
        distance = ((running + features) / (k + 1) - mean).square().sum(1)
        distance[~available] = torch.inf
        index = int(distance.argmin())
        chosen.append(index)
        available[index] = False
        running += features[index]
    return chosen


@torch.no_grad()
def initialize_cosine_rows(model, new_items, old_count, args):
    if not isinstance(model.head, CosineHead):
        return
    for position in range(old_count, len(model.classes)):
        selected = [item for item in new_items if item['y'] == model.classes[position]]
        features = features_of(model, selected, args)
        if model.head.proxies == 1:
            centers = F.normalize(features.mean(0, keepdim=True), dim=1)
        else:
            # Deterministic farthest-point initialization followed by spherical k-means.
            indices = [0]
            for _ in range(1, model.head.proxies):
                distance = 1 - (features @ features[indices].T).max(1).values
                indices.append(int(distance.argmax()))
            centers = features[indices].clone()
            for _ in range(10):
                assignments = (features @ centers.T).argmax(1)
                for k in range(len(centers)):
                    if (assignments == k).any():
                        centers[k] = F.normalize(features[assignments == k].mean(0), dim=0)
        norm = model.head.weight[:old_count].norm(dim=2).mean() if old_count else .2
        model.head.weight[position].copy_(centers.to(args.device) * norm)


def quotas_for(candidates, classes, limit):
    quotas = {c: 0 for c in classes}
    while sum(quotas.values()) < limit:
        changed = False
        for c in classes:
            if quotas[c] < len(candidates[c]):
                quotas[c] += 1
                changed = True
                if sum(quotas.values()) == limit:
                    break
        if not changed:
            break
    return quotas


def replay_payload(items):
    """Portable numeric arrays; excludes recomputable physics features."""
    payload = {'samples': ReplayBuffer._array(items)}
    if items and 'dark_logits' in items[0]:
        payload['dark_logits'] = np.stack([x['dark_logits'] for x in items])
        payload['dark_mask'] = np.stack([x['dark_mask'] for x in items])
    return payload


def replay_size(items):
    stream = io.BytesIO()
    np.savez(stream, **replay_payload(items))
    return stream.tell()


def update_memory(model, old, new, seen_count, trajectory, args, stats):
    if args.method == 'finetune':
        return []
    limit = math.floor(seen_count * args.replay_fraction)
    candidates = {c: [x for x in old + new if x['y'] == c] for c in model.classes}
    if limit < len(model.classes):
        raise ValueError('Replay budget cannot retain one example/class; increase --replay-fraction')
    per_item = ReplayBuffer._array([new[0]]).nbytes + (50 if args.method == 'derpp' else 0)
    limit = min(limit, max(0, (args.memory_bytes - 2048) // per_item))
    if limit < len(model.classes):
        raise ValueError('Byte budget too small for one example/class')
    quotas = quotas_for(candidates, model.classes, limit)
    selected = []
    old_classes = {x['y'] for x in old}
    for c in model.classes:
        pool = candidates[c]
        if c in old_classes:
            # Ordered herding prefixes survive shrinking; never reopen old train shards.
            chosen = pool[:quotas[c]]
        elif args.method not in ('er', 'derpp'):
            indices = select_herding(features_of(model, materialize(pool, stats), args), quotas[c])
            chosen = [pool[i] for i in indices]
        else:
            chosen = sorted(pool, key=lambda x: hashlib.sha256(
                f'{args.seed}:{x["sample_id"]}'.encode()).digest())[:quotas[c]]
        for item in chosen:
            record = ReplayBuffer._persistent(item)
            if args.method == 'derpp':
                if c in old_classes:
                    record['dark_logits'] = item['dark_logits'].copy()
                    record['dark_mask'] = item['dark_mask'].copy()
                else:
                    # Captured on first presentation BEFORE its optimizer update.
                    values = trajectory[int(item['sample_id'])]
                    record['dark_logits'] = np.zeros(10, dtype=np.float32)
                    record['dark_mask'] = np.zeros(10, dtype=np.bool_)
                    record['dark_logits'][:len(values)] = values
                    record['dark_mask'][:len(values)] = True
            selected.append(record)
    if replay_size(selected) > args.memory_bytes:
        raise ValueError('Serialized replay exceeds byte budget')
    assert len(selected) <= math.floor(seen_count * args.replay_fraction)
    return selected


def split_bic(items, classes, fraction, seed):
    grouped = {c: [x for x in items if x['y'] == c] for c in classes}
    quota = min(max(1, int(len(group) * fraction)) for group in grouped.values())
    if min(map(len, grouped.values())) <= quota:
        raise ValueError('BiC needs at least two retained examples/class for disjoint calibration')
    train, held = [], []
    rng = random.Random(seed)
    for group in grouped.values():
        rng.shuffle(group)
        held.extend(group[:quota])
        train.extend(group[quota:])
    assert not ({x['sample_id'] for x in train} & {x['sample_id'] for x in held})
    return train, held


@torch.no_grad()
def evaluate(model, items, groups, args, means=None):
    model.eval()
    predictions, labels = [], []
    for raw in loader(items, args):
        batch = to_device(raw, args.device)
        out = model(batch)
        scores = out['logits']
        if means is not None:
            scores = F.normalize(out['features'].float(), dim=1) @ means.T
        predicted = np.asarray(model.classes)[scores.argmax(1).cpu().numpy()]
        predictions.extend(predicted.tolist())
        labels.extend(raw['y'].tolist())
    labels, predictions = np.asarray(labels), np.asarray(predictions)
    per_class = {str(c): float((predictions[labels == c] == c).mean()) for c in model.classes}
    task = [float(np.mean([per_class[str(c)] for c in group])) for group in groups]
    old = [c for group in groups[:-1] for c in group]
    confusion = [[int(((labels == c) & (predictions == p)).sum()) for p in model.classes] for c in model.classes]
    return {'overall': float((predictions == labels).mean()),
            'macro': float(np.mean(list(per_class.values()))),
            'old': float(np.mean([per_class[str(c)] for c in old])) if old else None,
            'new': task[-1], 'per_class': per_class, 'task_macro': task,
            'count': len(labels), 'class_order': model.classes.copy(),
            'confusion_matrix': confusion}, predictions


def train_phase(model, teacher, items, old_memory, val, groups, old_count, epochs,
                args, trajectory, phase='train', compression_weights=None):
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=args.lr if phase == 'train' else args.lr * .1,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, max(epochs, 1))
    enabled = args.amp and args.device.type == 'cuda'
    scaler = (torch.amp.GradScaler('cuda', enabled=enabled) if hasattr(torch.amp, 'GradScaler')
              else torch.cuda.amp.GradScaler(enabled=enabled))
    train_loader = loader(items, args, shuffle=True)
    rng = random.Random(args.seed + len(groups) * 1009)
    class_counts = [sum(x['y'] == c for x in items) for c in model.classes]
    boost_weights = effective_weights(class_counts, args.foster_beta1, args.device)
    best, best_score, history, steps = None, -math.inf, [], 0
    # LUCIR's old cosine prototypes stay fixed, including AdamW weight decay.
    frozen_rows = model.head.weight[:old_count].detach().clone() if args.method == 'lucir' and old_count else None
    for epoch in range(epochs):
        model.train()
        total, batches = 0., 0
        for raw in train_loader:
            batch = to_device(raw, args.device)
            targets = targets_for(batch['y'], model.classes)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(args.device, args.amp):
                output = model(batch)
                previous = None
                if teacher is not None:
                    with torch.no_grad():
                        previous = teacher(batch)
                if args.method == 'lucir':
                    output['cosine'] = output['logits'].float() / model.head.log_scale.exp().clamp(max=100)
                if phase == 'compression':
                    probabilities = (previous['logits'].float() / args.temperature).softmax(1)
                    probabilities *= compression_weights
                    probabilities /= probabilities.sum(1, keepdim=True)
                    loss = -(probabilities * F.log_softmax(output['logits'].float()/args.temperature, 1)).sum(1).mean()
                else:
                    loss = objective(args.method, output, targets, old_count, previous,
                                     temperature=args.temperature, kd_weight=args.kd_weight,
                                     feature_weight=args.lucir_feature_weight,
                                     ranking_margin=args.ranking_margin, ranking_k=args.ranking_k,
                                     pod_spatial=args.pod_spatial, pod_flat=args.pod_flat,
                                     class_weights=boost_weights)
                if args.method == 'derpp' and phase == 'train':
                    if epoch == 0:
                        for sid, values in zip(raw['sample_id'], output['logits'].detach().float().cpu().numpy()):
                            trajectory.setdefault(int(sid), values.copy())
                    if old_memory:
                        # Independent draws for logit matching and supervised replay.
                        dark = rng.choices(old_memory, k=min(args.batch_size, len(old_memory)))
                        labelled = rng.choices(old_memory, k=min(args.batch_size, len(old_memory)))
                        d_out = model(to_device(collate_numpy(dark), args.device))['logits'].float()
                        stored = torch.tensor(np.stack([x['dark_logits'] for x in dark]), device=args.device)[:, :len(model.classes)]
                        mask = torch.tensor(np.stack([x['dark_mask'] for x in dark]), device=args.device)[:, :len(model.classes)]
                        mse = ((d_out - stored).square() * mask).sum() / mask.sum().clamp_min(1)
                        replay = to_device(collate_numpy(labelled), args.device)
                        replay_ce = F.cross_entropy(model(replay)['logits'].float(), targets_for(replay['y'], model.classes))
                        loss += args.der_alpha * mse + args.der_beta * replay_ce
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{args.method}/{phase}: nonfinite loss')
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
            scaler.step(optimizer)
            scaler.update()
            if frozen_rows is not None:
                with torch.no_grad():
                    model.head.weight[:old_count].copy_(frozen_rows)
            steps += 1
            total += float(loss.detach())
            batches += 1
        scheduler.step()
        entry = {'phase': phase, 'epoch': epoch+1, 'loss': total/max(batches, 1)}
        if args.selection == 'val_macro':
            metric, _ = evaluate(model, val, groups, args)
            entry['val_macro_head'] = metric['macro']
            if metric['macro'] > best_score:
                best_score = metric['macro']
                best = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        history.append(entry)
        print(json.dumps({'method': args.method, 'stage': len(groups)-1, **entry}), flush=True)
    if best is not None:
        model.load_state_dict(best)
    return history, steps


def calibrate_bic(model, items, old_count, args):
    model.eval()
    scores, targets = [], []
    with torch.no_grad():
        for raw in loader(items, args):
            batch = to_device(raw, args.device)
            scores.append(model(batch)['logits'].float())
            targets.append(targets_for(batch['y'], model.classes))
    scores, targets = torch.cat(scores).detach(), torch.cat(targets)
    scale = nn.Parameter(torch.ones((), device=args.device))
    shift = nn.Parameter(torch.zeros((), device=args.device))
    optimizer = torch.optim.Adam([scale, shift], lr=.01)
    for _ in range(args.calibration_steps):
        corrected = torch.cat((scores[:, :old_count], scores[:, old_count:] * scale + shift), 1)
        loss = F.cross_entropy(corrected, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        model.bias_scale[old_count:] = scale
        model.bias_shift[old_count:] = shift
    return {'scale': float(scale.detach()), 'shift': float(shift.detach()), 'steps': args.calibration_steps}


def class_means(model, memory, args):
    features = features_of(model, memory, args)
    labels = torch.tensor([x['y'] for x in memory])
    return F.normalize(torch.stack([features[labels == c].mean(0) for c in model.classes]), dim=1).to(args.device)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method', choices=METHODS, required=True)
    parser.add_argument('--data', type=Path, default=Path('echo_cil_processed'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--order', choices=ORDERS, default='O0')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--base-epochs', type=int)
    parser.add_argument('--finetune-epochs', type=int, default=20)
    parser.add_argument('--compression-epochs', type=int, default=40)
    parser.add_argument('--calibration-steps', type=int, default=200)
    parser.add_argument('--calibration-fraction', type=float, default=.1)
    parser.add_argument('--backbone', choices=('echo', 'cnn1d'), default='echo')
    parser.add_argument('--selection', choices=('val_macro', 'fixed'), default='val_macro')
    parser.add_argument('--replay-fraction', type=float, default=.1)
    parser.add_argument('--memory-bytes', type=int, default=32*1024*1024)
    parser.add_argument('--lr', type=float, default=.0005)
    parser.add_argument('--weight-decay', type=float, default=.0001)
    parser.add_argument('--temperature', type=float, default=2.)
    parser.add_argument('--kd-weight', type=float, default=1.)
    parser.add_argument('--lucir-feature-weight', type=float, default=5.)
    parser.add_argument('--ranking-margin', type=float, default=.5)
    parser.add_argument('--ranking-k', type=int, default=2)
    parser.add_argument('--pod-spatial', type=float, default=5.)
    parser.add_argument('--pod-flat', type=float, default=1.)
    parser.add_argument('--proxies', type=int, default=10)
    parser.add_argument('--der-alpha', type=float, default=.5)
    parser.add_argument('--der-beta', type=float, default=.5)
    parser.add_argument('--foster-beta1', type=float, default=.96)
    parser.add_argument('--foster-beta2', type=float, default=.97)
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()
    for name in ('batch_size', 'epochs', 'finetune_epochs', 'compression_epochs',
                 'calibration_steps', 'cpu_threads', 'memory_bytes', 'proxies', 'ranking_k'):
        if getattr(args, name) <= 0:
            parser.error(f'{name} must be positive')
    if args.base_epochs is not None and args.base_epochs <= 0:
        parser.error('--base-epochs must be positive')
    if not 0 < args.replay_fraction <= .1:
        parser.error('--replay-fraction must be in (0, .1]')
    if not 0 < args.calibration_fraction < 1 or args.num_workers < 0:
        parser.error('Invalid calibration fraction / worker count')
    for name in ('lr', 'temperature'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) <= 0:
            parser.error(f'{name} must be finite and positive')
    for name in ('weight_decay', 'kd_weight', 'lucir_feature_weight', 'ranking_margin',
                 'pod_spatial', 'pod_flat', 'der_alpha', 'der_beta'):
        if not math.isfinite(getattr(args, name)) or getattr(args, name) < 0:
            parser.error(f'{name} must be finite and nonnegative')
    if not 0 < args.foster_beta1 < 1 or not 0 < args.foster_beta2 < 1:
        parser.error('FOSTER beta values must be in (0,1)')
    return args


def main():
    args = parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f'Output is not empty: {args.output}; use a new directory')
    stats = load_physics_stats(args.data)
    groups = ORDERS[args.order]
    if list(stats['source_classes']) != groups[0]:
        raise ValueError('Normalization classes do not match order; preprocess this order separately')
    args.device = initialize_device(args.device)
    args.amp = args.amp and args.device.type == 'cuda'
    print(f'Using {args.device}; AMP={args.amp}', flush=True)
    torch.set_num_threads(args.cpu_threads)
    seed_everything(args.seed)
    if args.deterministic:
        torch.use_deterministic_algorithms(True)
    configuration = {k: str(v) if isinstance(v, (Path, torch.device)) else v for k, v in vars(args).items()}
    configuration.update(implementation='1d_adaptation',
                         validation_protocol='all_seen_val_selection' if args.selection == 'val_macro' else 'no_validation_selection',
                         memory_protocol='growing_fraction_of_seen_train',
                         test_used_for_training=False, torch_version=torch.__version__)
    configuration['source_sha256'] = {str(path.relative_to(Path(__file__).parent)): hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in [Path(__file__), Path(__file__).parent/'echo_cil'/'baselines.py']}
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output/'configuration.json', configuration)
    results = {'configuration': configuration, 'stages': [], 'status': 'running'}
    model, memory, seen_count = None, [], 0
    for stage, new_classes in enumerate(groups):
        started = time.perf_counter()
        if args.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(args.device)
        old_count = sum(map(len, groups[:stage]))
        teacher = model.frozen_copy() if model is not None and args.method not in ('er', 'finetune', 'derpp') else None
        if model is None:
            model = CILModel(args.method, new_classes, args.backbone, args.proxies).to(args.device)
        else:
            model.expand(new_classes)
        data = EchoCILDataset(args.data, 'train', new_classes)
        new = [data[i] for i in range(len(data))]
        del data
        seen_count += len(new)
        old = materialize(memory, stats)
        train = new if args.method in ('finetune', 'derpp') else old + new
        calibration = []
        if args.method == 'bic' and stage:
            train, calibration = split_bic(train, model.classes, args.calibration_fraction, args.seed+stage)
        val_data = EchoCILDataset(args.data, 'val', model.classes)
        val = [val_data[i] for i in range(len(val_data))] if args.selection == 'val_macro' else []
        del val_data
        trajectory = {}
        if stage and args.method in ('lucir', 'podnet'):
            initialize_cosine_rows(model, new, old_count, args)
        history, steps = train_phase(model, teacher, train, old, val, groups[:stage+1], old_count,
                                     (args.base_epochs or args.epochs) if stage == 0 else args.epochs,
                                     args, trajectory)
        corrections = {}
        if args.method == 'bic' and stage:
            corrections = calibrate_bic(model, calibration, old_count, args)
            steps += args.calibration_steps
        if args.method in ('wa', 'der_expand', 'foster') and stage:
            corrections['weight_alignment'] = model.align(old_count)
        peak_parameters = sum(p.numel() for p in model.parameters())
        if args.method == 'foster' and stage:
            boosted = model.frozen_copy()
            student = CILModel('foster', model.classes, args.backbone, args.proxies).to(args.device)
            student.encoders[0].load_state_dict(teacher.encoders[0].state_dict())
            with torch.no_grad():
                student.head.weight[:old_count].copy_(teacher.head.weight)
                student.head.bias[:old_count].copy_(teacher.head.bias)
            counts = [sum(x['y'] == c for x in train) for c in model.classes]
            weights = effective_weights(counts, args.foster_beta2, args.device)
            more, extra = train_phase(student, boosted, train, [], val, groups[:stage+1], old_count,
                                      args.compression_epochs, args, {}, 'compression', weights)
            history += more
            steps += extra
            model = student
            del boosted, student
        memory = update_memory(model, memory, new, seen_count, trajectory, args, stats)
        if args.method in ('lucir', 'podnet') and stage:
            balanced = materialize(memory, stats)
            for encoder in model.encoders:
                encoder.requires_grad_(False)
            more, extra = train_phase(model, teacher, balanced, [], val, groups[:stage+1], old_count,
                                      args.finetune_epochs, args, {}, 'balanced_head')
            history += more
            steps += extra
            for encoder in model.encoders:
                encoder.requires_grad_(True)
            del balanced
        means = class_means(model, materialize(memory, stats), args) if args.method == 'icarl' else None
        test_data = EchoCILDataset(args.data, 'test', model.classes)
        test = [test_data[i] for i in range(len(test_data))]
        del test_data
        metrics, predicted = evaluate(model, test, groups[:stage+1], args, means)
        stage_result = {'stage': stage, 'test': metrics, 'memory_count': len(memory),
                        'seen_training_count': seen_count, 'new_training_count': len(new),
                        'old_replay_used': len(old), 'training_pool_count': len(train),
                        'calibration_count': len(calibration),
                        'replay_sample_limit': math.floor(seen_count*args.replay_fraction),
                        'replay_fraction_actual': len(memory)/seen_count,
                        'memory_bytes': replay_size(memory) if memory else 0,
                        'prototype_tensor_bytes': means.numel()*means.element_size() if means is not None else 0,
                        'model_parameters': sum(p.numel() for p in model.parameters()),
                        'peak_expanded_model_parameters': peak_parameters,
                        'model_tensor_bytes': sum(t.numel()*t.element_size() for t in model.state_dict().values()),
                        'optimizer_steps': steps, 'training_history': history, 'corrections': corrections,
                        'elapsed_seconds': time.perf_counter()-started,
                        'cuda_peak_allocated_bytes': torch.cuda.max_memory_allocated(args.device) if args.device.type == 'cuda' else None}
        if args.method in ('icarl', 'podnet'):
            head_metrics, _ = evaluate(model, test, groups[:stage+1], args)
            stage_result['test_head'] = head_metrics
            if args.method == 'podnet':
                nme = class_means(model, materialize(memory, stats), args)
                stage_result['test_nme'], _ = evaluate(model, test, groups[:stage+1], args, nme)
        with (args.output/f'predictions_stage_{stage}.csv').open('w', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(['sample_id', 'record_id', 'true_class', 'predicted_class'])
            writer.writerows((x['sample_id'], x['record_id'], x['y'], int(p)) for x, p in zip(test, predicted))
        np.savez(args.output/f'memory_stage_{stage}.npz', **replay_payload(memory))
        torch.save({'model': model.state_dict(), 'method': args.method, 'backbone': args.backbone,
                    'classes': model.classes, 'encoder_count': len(model.encoders), 'proxies': args.proxies,
                    'aux_outputs': model.aux.out_features if model.aux is not None else None,
                    'class_means': means.cpu() if means is not None else None,
                    'stage': stage, 'configuration': configuration}, args.output/f'stage_{stage}.pt')
        results['stages'].append(stage_result)
        results['summary'] = experiment_summary(results['stages'])
        write_json(args.output/'results.json', results)
        print(json.dumps({'method': args.method, 'stage': stage, 'test_macro': metrics['macro'],
                          'old': metrics['old'], 'new': metrics['new'], 'memory': len(memory)}), flush=True)
        del teacher, old, new, train, calibration, val, test, trajectory
    results['status'] = 'complete'
    write_json(args.output/'results.json', results)


if __name__ == '__main__':
    main()
