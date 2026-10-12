"""Train the accuracy-oriented ECHO-CIL implementation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time
from contextlib import nullcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from echo_cil.data import (
    DictDataset, EchoCILDataset, GroupBalancedBatchSampler, collate_numpy,
)
from echo_cil.guard import build_protection, projection_error
from echo_cil.memory import ReplayBuffer
from echo_cil.model import ECHOModel
from echo_cil.physics import load_physics_stats, physics_features


ORDERS = {
    'O0': [[0, 1, 2, 3], [4, 5, 6], [7, 8, 9]],
    'O1': [[0, 7, 8, 9], [4, 5, 6], [1, 2, 3]],
    'O2': [[0, 4, 9, 2], [6, 1, 8], [3, 7, 5]],
    'O3': [[0, 8, 3, 4], [2, 7, 6], [9, 5, 1]],
    'O4': [[0, 6, 7, 5], [1, 9, 2], [8, 4, 3]],
}


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    requested = requested.lower()
    if requested == 'auto':
        requested = 'cuda' if torch.cuda.is_available() else 'cpu'
    if requested.startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError(
            f'CUDA device {requested!r} requested, but CUDA is unavailable. '
            'Install a CUDA-enabled PyTorch build and verify the NVIDIA driver.')
    device = torch.device(requested)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('strict ECHO-CIL currently supports cpu and cuda')
    return device


def amp_context(device: torch.device, enabled: bool):
    return (torch.autocast('cuda', dtype=torch.float16)
            if enabled and device.type == 'cuda' else nullcontext())


def loader_options(device: torch.device, workers: int) -> dict:
    options = {'num_workers': workers, 'pin_memory': device.type == 'cuda'}
    if workers:
        options['persistent_workers'] = True
    return options


def to_device(raw: Dict[str, np.ndarray], device: torch.device) -> Dict[str, Tensor]:
    result = {}
    for key in ('x', 'order', 'quality', 'y', 'rpm'):
        tensor = torch.from_numpy(raw[key])
        result[key] = tensor.to(device, non_blocking=device.type == 'cuda')
    return result


def make_loader(dataset: DictDataset, old_classes: Sequence[int],
                new_classes: Sequence[int], batch_size: int, seed: int,
                device: torch.device, workers: int) -> tuple[DataLoader, GroupBalancedBatchSampler]:
    sampler = GroupBalancedBatchSampler(
        dataset, old_classes, new_classes, batch_size, seed)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_numpy,
                        **loader_options(device, workers))
    return loader, sampler


def supervised_contrastive_loss(features: Tensor, labels: Tensor,
                                temperature: float) -> Tensor:
    """Single-view supervised contrastive loss for fine-grained severities."""
    if features.shape[0] < 2:
        return features.new_zeros(())
    normalized = nn.functional.normalize(features.float(), dim=1)
    similarities = normalized @ normalized.transpose(0, 1) / temperature
    eye = torch.eye(features.shape[0], dtype=torch.bool, device=features.device)
    positives = labels[:, None].eq(labels[None, :]) & ~eye
    valid = positives.any(dim=1)
    if not valid.any():
        return features.new_zeros(())
    similarities = similarities - similarities.max(dim=1, keepdim=True).values.detach()
    log_denominator = torch.logsumexp(
        similarities.masked_fill(eye, float('-inf')), dim=1)
    log_probability = similarities - log_denominator[:, None]
    per_sample = -(log_probability.masked_fill(~positives, 0.0).sum(dim=1)
                   / positives.sum(dim=1).clamp_min(1))
    return per_sample[valid].mean()


def loss_terms(model: ECHOModel, logits: Tensor, batch: Dict[str, Tensor],
               features: Tensor, diagnostics: dict, old_classes: Sequence[int],
               new_classes: Sequence[int], competition_margin: float,
               lambda_competition: float, lambda_amplitude: float,
               teacher_logits: Optional[Tensor] = None,
               lambda_kd: float = 0.0, kd_temperature: float = 2.0,
               lambda_supcon: float = 0.0, supcon_temperature: float = 0.07,
               base_stage: bool = False):
    targets = model.classifier.target_indices(batch['y'])
    losses = nn.functional.cross_entropy(logits, targets, reduction='none')
    if base_stage:
        sample_weights = torch.ones_like(losses)
    else:
        total_classes = len(old_classes) + len(new_classes)
        old_weight = 2.0 * len(old_classes) / total_classes
        new_weight = 2.0 * len(new_classes) / total_classes
        old_mask = torch.zeros_like(batch['y'], dtype=torch.bool)
        for label in old_classes:
            old_mask |= batch['y'] == label
        sample_weights = torch.where(
            old_mask, losses.new_tensor(old_weight), losses.new_tensor(new_weight))
    ce = (losses * sample_weights).mean()
    supcon = supervised_contrastive_loss(
        features, batch['y'], supcon_temperature)
    if base_stage:
        total = ce + lambda_supcon * supcon
        return total, {
            'ce': float(ce.detach()), 'competition': 0.0, 'amplitude': 0.0,
            'kd': 0.0, 'supcon': float(supcon.detach()),
        }

    class_positions = {label: index for index, label in enumerate(model.class_ids)}
    competition = []
    for row, label, weight in zip(logits, batch['y'].tolist(), sample_weights):
        opposite = new_classes if int(label) in old_classes else old_classes
        opposite_indices = torch.as_tensor(
            [class_positions[value] for value in opposite],
            device=logits.device, dtype=torch.long)
        true_index = class_positions[int(label)]
        competition.append(weight * torch.relu(
            logits.new_tensor(competition_margin)
            + row[opposite_indices].max() - row[true_index]))
    competition_loss = torch.stack(competition).mean()
    amplitude = diagnostics['current_delta'].pow(2).sum(dim=1).mean()
    kd = logits.new_zeros(())
    if teacher_logits is not None and old_classes:
        old_mask = torch.zeros_like(batch['y'], dtype=torch.bool)
        for label in old_classes:
            old_mask |= batch['y'] == label
        if old_mask.any():
            positions = {label: index for index, label in enumerate(model.class_ids)}
            old_positions = torch.as_tensor(
                [positions[label] for label in old_classes],
                device=logits.device, dtype=torch.long)
            student_old = logits[old_mask].index_select(1, old_positions)
            teacher_old = teacher_logits[old_mask]
            kd = nn.functional.kl_div(
                nn.functional.log_softmax(student_old / kd_temperature, dim=1),
                nn.functional.softmax(teacher_old / kd_temperature, dim=1),
                reduction='batchmean') * (kd_temperature ** 2)
    total = (ce + lambda_competition * competition_loss
             + lambda_amplitude * amplitude + lambda_kd * kd
             + lambda_supcon * supcon)
    return total, {
        'ce': float(ce.detach()),
        'competition': float(competition_loss.detach()),
        'amplitude': float(amplitude.detach()),
        'kd': float(kd.detach()),
        'supcon': float(supcon.detach()),
    }


def train_epochs(model: ECHOModel, loader: DataLoader,
                 sampler: GroupBalancedBatchSampler, parameters: Iterable[nn.Parameter],
                 epochs: int, device: torch.device, use_amp: bool, lr: float,
                 weight_decay: float, old_classes: Sequence[int],
                 new_classes: Sequence[int], competition_margin: float,
                 lambda_competition: float, lambda_amplitude: float,
                 enabled_current: Optional[Iterable[int]] = None,
                 base_stage: bool = False, teacher: Optional[ECHOModel] = None,
                 lambda_kd: float = 0.0, kd_temperature: float = 2.0,
                 lambda_supcon: float = 0.0, supcon_temperature: float = 0.07,
                 validation: Optional[tuple] = None,
                 early_stopping_patience: int = 0) -> tuple[int, List[dict]]:
    parameter_spec = list(parameters)
    if parameter_spec and isinstance(parameter_spec[0], dict):
        optimizer_groups, clip_parameters = [], []
        for raw_group in parameter_spec:
            group = dict(raw_group)
            group_parameters = [parameter for parameter in group.pop('params')
                                if parameter.requires_grad]
            if group_parameters:
                optimizer_groups.append({'params': group_parameters, **group})
                clip_parameters.extend(group_parameters)
        parameters = clip_parameters
        optimizer_input = optimizer_groups
    else:
        parameters = [parameter for parameter in parameter_spec
                      if parameter.requires_grad]
        optimizer_input = parameters
    if epochs <= 0 or not parameters:
        return 0, []
    optimizer = torch.optim.AdamW(
        optimizer_input, lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, epochs), eta_min=lr * 0.01)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    history, steps = [], 0
    best_state, best_score, stale_epochs = None, -float('inf'), 0
    for epoch in range(epochs):
        sampler.set_epoch(epoch)
        model.train()
        running = {'loss': 0.0, 'ce': 0.0, 'competition': 0.0,
                   'amplitude': 0.0, 'kd': 0.0, 'supcon': 0.0}
        batches = 0
        for raw in loader:
            batch = to_device(raw, device)
            optimizer.zero_grad(set_to_none=True)
            with amp_context(device, use_amp):
                teacher_logits = None
                if teacher is not None:
                    with torch.no_grad():
                        teacher_logits, _, _ = teacher(
                            batch['x'], batch['order'], batch['quality'], batch['rpm'])
                logits, features, diagnostics = model(
                    batch['x'], batch['order'], batch['quality'], batch['rpm'],
                    enabled_current=enabled_current)
                loss, parts = loss_terms(
                    model, logits, batch, features, diagnostics,
                    old_classes, new_classes,
                    competition_margin, lambda_competition, lambda_amplitude,
                    teacher_logits=teacher_logits, lambda_kd=lambda_kd,
                    kd_temperature=kd_temperature, lambda_supcon=lambda_supcon,
                    supcon_temperature=supcon_temperature,
                    base_stage=base_stage)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(parameters, 5.0)
                optimizer.step()
            running['loss'] += float(loss.detach())
            for key, value in parts.items():
                running[key] += value
            batches += 1; steps += 1
        scheduler.step()
        record = {
            'epoch': epoch + 1,
            **{key: value / max(1, batches) for key, value in running.items()},
            'lr': optimizer.param_groups[0]['lr'],
            'group_lrs': [group['lr'] for group in optimizer.param_groups],
        }
        if validation is not None:
            val_dataset, active_classes, task_groups, val_batch, val_workers = validation
            metrics = evaluate(model, val_dataset, active_classes, task_groups,
                               device, val_batch, val_workers)
            score = 0.5 * (metrics['macro'] + metrics['old_new_harmonic'])
            record.update({'validation_macro': metrics['macro'],
                           'validation_harmonic': metrics['old_new_harmonic'],
                           'validation_score': score})
            if score > best_score + 1e-8:
                best_score, stale_epochs = score, 0
                best_state = {key: value.detach().cpu().clone()
                              for key, value in model.state_dict().items()}
                record['best'] = True
            else:
                stale_epochs += 1
                record['best'] = False
        history.append(record)
        if (validation is not None and early_stopping_patience > 0
                and stale_epochs >= early_stopping_patience):
            record['stopped_early'] = True
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return steps, history


@torch.no_grad()
def extract_representations(model: ECHOModel, dataset, device: torch.device,
                            batch_size: int, workers: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_numpy,
                        **loader_options(device, workers))
    model.eval(); features, labels = [], []
    for raw in loader:
        batch = to_device(raw, device)
        z, _ = model.representation(
            batch['x'], batch['order'], batch['quality'], batch['rpm'])
        features.append(z.float().cpu()); labels.append(batch['y'].cpu())
    return torch.cat(features), torch.cat(labels)


@torch.no_grad()
def initialize_new_head(model: ECHOModel, teacher: ECHOModel, head: nn.Linear,
                        dataset, new_classes: Sequence[int], device: torch.device,
                        batch_size: int, workers: int) -> None:
    features, labels = extract_representations(
        teacher, dataset, device, batch_size, workers)
    old_weight, _ = teacher.classifier.weight_bias()
    target_norm = old_weight.detach().float().norm(dim=1).median().cpu()
    rows = []
    for label in new_classes:
        mean = features[labels == label].mean(dim=0)
        rows.append(target_norm * mean / mean.norm().clamp_min(1e-12))
    head.weight.copy_(torch.stack(rows).to(head.weight.device))
    head.bias.zero_()


def _perturbed_item(item: dict, physics_stats: dict, variant: int) -> dict:
    x = np.asarray(item['x'], dtype=np.float32).copy()
    rpm = float(item['rpm'])
    if variant == 1:
        rng = np.random.default_rng(int(item['sample_id']) & 0xFFFFFFFF)
        power = float(np.mean(x.astype(np.float64) ** 2))
        noise = rng.normal(0.0, math.sqrt(power / 1000.0), size=x.shape)
        x = (x + noise).astype(np.float32)
    elif variant == 2:
        rpm *= 0.99
    elif variant == 3:
        rpm *= 1.01
    order, quality = physics_features(x, rpm, physics_stats)
    enriched = dict(item)
    enriched.update({'x': x, 'rpm': rpm, 'order': order, 'quality': quality})
    return enriched


@torch.no_grad()
def teacher_replay_views(teacher: ECHOModel, old_items: Sequence[dict],
                         physics_stats: dict, device: torch.device,
                         batch_size: int, workers: int):
    all_logits = []
    labels = torch.as_tensor([int(item['y']) for item in old_items])
    for variant in range(4):
        items = [_perturbed_item(item, physics_stats, variant)
                 for item in old_items]
        loader = DataLoader(DictDataset(items, physics_stats),
                            batch_size=batch_size, shuffle=False,
                            collate_fn=collate_numpy,
                            **loader_options(device, workers))
        logits = []
        teacher.eval()
        for raw in loader:
            batch = to_device(raw, device)
            output, _, _ = teacher(
                batch['x'], batch['order'], batch['quality'], batch['rpm'])
            logits.append(output.float().cpu())
        all_logits.append(torch.cat(logits))
    return torch.stack(all_logits), labels


@torch.no_grad()
def predict_global(model: ECHOModel, dataset, device: torch.device,
                   batch_size: int, workers: int):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_numpy,
                        **loader_options(device, workers))
    model.eval(); predictions, labels = [], []
    class_ids = torch.as_tensor(model.class_ids)
    for raw in loader:
        batch = to_device(raw, device)
        logits, _, _ = model(
            batch['x'], batch['order'], batch['quality'], batch['rpm'])
        predictions.append(class_ids[logits.argmax(dim=1).cpu()])
        labels.append(batch['y'].cpu())
    return torch.cat(predictions), torch.cat(labels)


def conflict_groups(model: ECHOModel, dataset, old_classes: Sequence[int],
                    new_classes: Sequence[int], device: torch.device,
                    batch_size: int, workers: int, maximum: int = 2):
    predictions, labels = predict_global(
        model, dataset, device, batch_size, workers)
    scores = {}
    for old in old_classes:
        for new in new_classes:
            old_mask, new_mask = labels == old, labels == new
            old_to_new = float((predictions[old_mask] == new).float().mean())
            new_to_old = float((predictions[new_mask] == old).float().mean())
            scores[(old, new)] = 0.5 * (old_to_new + new_to_old)
    ranked_new = sorted(
        new_classes,
        key=lambda new: (-max(scores[(old, new)] for old in old_classes), new))
    groups = []
    for new in ranked_new[:maximum]:
        old_ranked = sorted(old_classes,
                            key=lambda old: (-scores[(old, new)], old))
        conflicts = [old for old in old_ranked if scores[(old, new)] > 0][:2]
        group = [new] + conflicts
        if group not in groups:
            groups.append(group)
    return groups or [[new] for new in new_classes[:maximum]], scores


@torch.no_grad()
def evaluate(model: ECHOModel, dataset, active_classes: Sequence[int],
             task_groups: Sequence[Sequence[int]], device: torch.device,
             batch_size: int, workers: int) -> dict:
    predictions, labels = predict_global(
        model, dataset, device, batch_size, workers)
    per_class = {}
    confusion = np.zeros((len(active_classes), len(active_classes)), dtype=np.int64)
    position = {label: index for index, label in enumerate(active_classes)}
    for label in active_classes:
        mask = labels == label
        per_class[str(label)] = float((predictions[mask] == labels[mask]).float().mean())
    for predicted, target in zip(predictions.tolist(), labels.tolist()):
        confusion[position[target], position[predicted]] += 1
    macro = lambda values: float(np.mean([per_class[str(v)] for v in values])) if values else 0.0
    old_classes = [value for group in task_groups[:-1] for value in group]
    new_classes = list(task_groups[-1])
    old_acc, new_acc = macro(old_classes), macro(new_classes)
    harmonic = (2.0 * old_acc * new_acc / max(old_acc + new_acc, 1e-12)
                if old_classes else new_acc)
    return {
        'overall': float((predictions == labels).float().mean()),
        'macro': macro(active_classes), 'old': old_acc, 'new': new_acc,
        'old_new_harmonic': harmonic, 'per_class': per_class,
        'task_macro': [macro(group) for group in task_groups],
        'confusion': confusion.tolist(),
    }


@torch.no_grad()
def mechanism_metrics(model: ECHOModel, dataset, device: torch.device,
                      batch_size: int, workers: int) -> dict:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=collate_numpy,
                        **loader_options(device, workers))
    model.eval(); gammas, scales = [], [[], [], []]
    gates: List[List[Tensor]] = [[] for _ in model.experts]
    residual_norms = []
    for raw in loader:
        batch = to_device(raw, device)
        _, _, diagnostics = model(
            batch['x'], batch['order'], batch['quality'], batch['rpm'])
        gammas.append(diagnostics['gamma'].float().cpu())
        for index, alpha in enumerate(diagnostics['scale_weights']):
            scales[index].append(alpha.float().cpu())
        for index, gate in enumerate(diagnostics['expert_gates']):
            gates[index].append(gate.float().cpu())
        residual_norms.append(
            diagnostics['current_delta'].float().norm(dim=1).cpu())
    gamma = torch.cat(gammas)
    scale_metrics = []
    for values in scales:
        alpha = torch.cat(values)
        entropy = -(alpha * torch.log(alpha.clamp_min(1e-12))).sum(dim=1)
        scale_metrics.append({
            'mean': alpha.mean(dim=0).tolist(),
            'mean_entropy': float(entropy.mean()),
            'sample_variance': alpha.var(dim=0, unbiased=False).tolist(),
        })
    return {
        'gamma_mean': float(gamma.mean()),
        'gamma_std': float(gamma.std(unbiased=False)),
        'scale_blocks': scale_metrics,
        'expert_gate_mean': [float(torch.cat(values).mean()) if values else 0.0
                             for values in gates],
        'current_residual_norm_mean': float(torch.cat(residual_norms).mean()),
    }


@torch.no_grad()
def capture_rate(previous: ECHOModel, current: ECHOModel, dataset,
                 new_classes: Sequence[int], device: torch.device,
                 batch_size: int, workers: int) -> float:
    previous_prediction, labels = predict_global(
        previous, dataset, device, batch_size, workers)
    current_prediction, _ = predict_global(
        current, dataset, device, batch_size, workers)
    denominator = previous_prediction == labels
    if not denominator.any():
        return 0.0
    new_mask = torch.zeros_like(current_prediction, dtype=torch.bool)
    for label in new_classes:
        new_mask |= current_prediction == label
    return float((denominator & new_mask).sum() / denominator.sum())


def experiment_summary(stages: Sequence[dict]) -> dict:
    macros = [stage['test']['macro'] for stage in stages]
    final = stages[-1]['test']['task_macro']
    forgetting, backward = [], []
    for task in range(len(stages) - 1):
        history = [stages[stage]['test']['task_macro'][task]
                   for stage in range(task, len(stages) - 1)]
        forgetting.append(max(history) - final[task])
        backward.append(final[task] - stages[task]['test']['task_macro'][task])
    return {
        'average_incremental_accuracy': float(np.mean(macros)),
        'final_macro_accuracy': macros[-1],
        'forgetting': float(np.mean(forgetting)) if forgetting else None,
        'backward_transfer': float(np.mean(backward)) if backward else None,
    }


def save_checkpoint(path: Path, model: ECHOModel, stage: int,
                    configuration: dict) -> None:
    torch.save({
        'stage': stage, 'model': model.state_dict(),
        'architecture': model.architecture_metadata(),
        'configuration': configuration,
        'torch_rng_state': torch.get_rng_state(),
        'numpy_rng_state': np.random.get_state(),
        'python_rng_state': random.getstate(),
    }, path)


def main() -> None:
    default_config = Path(__file__).resolve().parent / 'echo_cil_config.json'
    bootstrap = argparse.ArgumentParser(add_help=False)
    bootstrap.add_argument('--config', type=Path, default=default_config)
    bootstrap_args, _ = bootstrap.parse_known_args()
    config_values = {}
    if bootstrap_args.config.exists():
        config_values = json.loads(
            bootstrap_args.config.read_text(encoding='utf-8'))

    parser = argparse.ArgumentParser(parents=[bootstrap])
    parser.add_argument('--data', type=Path, default=Path('echo_cil_processed'))
    parser.add_argument('--output', type=Path, default=Path('runs/echo_cil'))
    parser.add_argument('--order', choices=sorted(ORDERS),
                        default=config_values.get('order', 'O0'))
    parser.add_argument('--seed', type=int, default=config_values.get('seed', 0))
    parser.add_argument('--batch-size', type=int,
                        default=config_values.get('batch_size', 64))
    parser.add_argument('--base-epochs', type=int,
                        default=config_values.get('base_epochs', 60))
    parser.add_argument('--head-warmup-epochs', type=int,
                        default=config_values.get('head_warmup_epochs', 5))
    parser.add_argument('--expert-warmup-epochs', type=int,
                        default=config_values.get('expert_group_warmup_epochs', 3))
    parser.add_argument('--incremental-epochs', type=int,
                        default=config_values.get('incremental_epochs', 60))
    parser.add_argument('--lr', type=float, default=config_values.get('lr', 5e-4))
    parser.add_argument('--backbone-lr', type=float,
                        default=config_values.get('backbone_lr', 1e-5))
    parser.add_argument('--weight-decay', type=float,
                        default=config_values.get('weight_decay', 1e-4))
    parser.add_argument('--competition-margin', type=float,
                        default=config_values.get('competition_margin', 0.5))
    parser.add_argument('--lambda-competition', type=float,
                        default=config_values.get('lambda_competition', 0.1))
    parser.add_argument('--lambda-amplitude', type=float,
                        default=config_values.get('lambda_amplitude', 1e-4))
    parser.add_argument('--lambda-kd', type=float,
                        default=config_values.get('lambda_kd', 1.0))
    parser.add_argument('--kd-temperature', type=float,
                        default=config_values.get('kd_temperature', 2.0))
    parser.add_argument('--lambda-supcon', type=float,
                        default=config_values.get('lambda_supcon', 0.1))
    parser.add_argument('--supcon-temperature', type=float,
                        default=config_values.get('supcon_temperature', 0.07))
    parser.add_argument('--early-stopping-patience', type=int,
                        default=config_values.get('early_stopping_patience', 10))
    parser.add_argument('--memory-bytes', type=int,
                        default=config_values.get('memory_budget_bytes', 32 * 1024 * 1024))
    parser.add_argument('--experts-per-stage', type=int,
                        default=config_values.get('experts_per_stage_max', 3))
    parser.add_argument('--expert-limit', type=int,
                        default=config_values.get('experts_total_max', 6))
    parser.add_argument('--device', default='auto')
    parser.add_argument('--amp', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
    args = parser.parse_args()

    if not args.data.exists():
        candidate = Path(__file__).resolve().parent / args.data
        if candidate.exists():
            args.data = candidate
    if not args.data.exists():
        raise FileNotFoundError(f'processed data not found: {args.data}')
    physics_stats = load_physics_stats(args.data)
    order = ORDERS[args.order]
    if list(physics_stats['source_classes']) != order[0]:
        raise ValueError(
            f'normalization source classes {physics_stats["source_classes"]} '
            f'do not match {args.order} base classes {order[0]}; regenerate data '
            f'with --base-classes {" ".join(map(str, order[0]))}')
    if args.batch_size % 2:
        raise ValueError('--batch-size must be even')
    if args.num_workers < 0:
        raise ValueError('--num-workers must be non-negative')
    if args.backbone_lr <= 0 or args.lr <= 0:
        raise ValueError('learning rates must be positive')
    if args.kd_temperature <= 0 or args.supcon_temperature <= 0:
        raise ValueError('loss temperatures must be positive')

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
        print(f'Using CUDA: {torch.cuda.get_device_name(device)}; AMP={use_amp}')
    else:
        print('Using CPU; AMP=False')
    args.output.mkdir(parents=True, exist_ok=True)
    configuration = vars(args).copy()
    configuration['data'] = str(args.data)
    configuration['output'] = str(args.output)
    configuration['task_order'] = order
    if args.config.exists():
        configuration['config_sha256'] = hashlib.sha256(
            args.config.read_bytes()).hexdigest()
    (args.output / 'configuration.json').write_text(
        json.dumps(configuration, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8')

    model = ECHOModel(
        order[0], expert_rank=int(config_values.get('expert_rank', 8)),
        expert_limit=args.expert_limit).to(device)
    memory = ReplayBuffer(args.memory_bytes, args.seed)
    stage_results = []

    for stage, new_classes in enumerate(order):
        stage_started = time.perf_counter()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        active_classes = [label for group in order[:stage + 1] for label in group]
        old_classes = [label for group in order[:stage] for label in group]
        new_dataset = EchoCILDataset(args.data, 'train', new_classes)
        new_items = [new_dataset[index] for index in range(len(new_dataset))]
        old_items = memory.all()
        train_dataset = DictDataset(old_items + new_items, physics_stats)
        val_dataset = EchoCILDataset(args.data, 'val', active_classes)
        test_dataset = EchoCILDataset(args.data, 'test', active_classes)
        train_loader, train_sampler = make_loader(
            train_dataset, old_classes, new_classes, args.batch_size,
            args.seed + stage * 1000, device, args.num_workers)
        stage_log = {'head_warmup': [], 'expert_warmup': [], 'joint': []}
        protection_json = {'rank': 0, 'selected_edges': [], 'candidate_edges': []}
        conflicts_json, expert_groups, projection_max_error = {}, [], 0.0
        previous_model = None

        if stage == 0:
            steps, history = train_epochs(
                model, train_loader, train_sampler, model.parameters(),
                args.base_epochs, device, use_amp, args.lr, args.weight_decay,
                [], new_classes, args.competition_margin,
                args.lambda_competition, args.lambda_amplitude,
                base_stage=True, lambda_supcon=args.lambda_supcon,
                supcon_temperature=args.supcon_temperature,
                validation=(val_dataset, active_classes, order[:stage + 1],
                            args.batch_size, args.num_workers),
                early_stopping_patience=args.early_stopping_patience)
            stage_log['joint'] = history
            total_steps = steps
        else:
            previous_model = model.frozen_copy()
            new_head = model.begin_increment(new_classes, stage)
            initialize_new_head(
                model, previous_model, new_head, new_dataset, new_classes,
                device, args.batch_size, args.num_workers)

            head_steps, history = train_epochs(
                model, train_loader, train_sampler, new_head.parameters(),
                args.head_warmup_epochs, device, use_amp, args.lr,
                args.weight_decay, old_classes, new_classes,
                args.competition_margin, args.lambda_competition,
                args.lambda_amplitude, teacher=previous_model,
                lambda_kd=args.lambda_kd, kd_temperature=args.kd_temperature,
                lambda_supcon=args.lambda_supcon,
                supcon_temperature=args.supcon_temperature)
            stage_log['head_warmup'] = history

            replay_logits, replay_labels = teacher_replay_views(
                previous_model, old_items, physics_stats, device,
                args.batch_size, args.num_workers)
            old_weight, _ = previous_model.classifier.weight_bias()
            protection = build_protection(
                replay_logits, replay_labels, old_classes, old_weight,
                min_samples=5, lower_quantile=0.10)
            protection_json = protection.to_json()

            groups, conflict_scores = conflict_groups(
                model, train_dataset, old_classes, new_classes, device,
                args.batch_size, args.num_workers,
                maximum=args.experts_per_stage)
            remaining = args.expert_limit - len(model.experts)
            groups = groups[:max(0, remaining)]
            if not groups and remaining > 0:
                groups = [list(new_classes)]
            conflicts_json = {
                f'{old}-{new}': value
                for (old, new), value in conflict_scores.items()
            }
            expert_indices = [model.add_expert(protection.basis, group)
                              for group in groups]
            expert_groups = groups

            new_head.requires_grad_(False)
            warmup_steps = 0
            for expert_index, group in zip(expert_indices, groups):
                group_old = [label for label in group if label in old_classes]
                group_new = [label for label in group if label in new_classes]
                if not group_new:
                    group_new = list(new_classes)
                group_items = [item for item in old_items + new_items
                               if int(item['y']) in group_old + group_new]
                group_dataset = DictDataset(group_items, physics_stats)
                group_loader, group_sampler = make_loader(
                    group_dataset, group_old, group_new, args.batch_size,
                    args.seed + stage * 1000 + expert_index + 101,
                    device, args.num_workers)
                steps, history = train_epochs(
                    model, group_loader, group_sampler,
                    model.experts[expert_index].parameters(),
                    args.expert_warmup_epochs, device, use_amp, args.lr,
                    args.weight_decay, old_classes, new_classes,
                    args.competition_margin, args.lambda_competition,
                    args.lambda_amplitude, enabled_current=[expert_index],
                    teacher=previous_model, lambda_kd=args.lambda_kd,
                    kd_temperature=args.kd_temperature,
                    lambda_supcon=args.lambda_supcon,
                    supcon_temperature=args.supcon_temperature)
                warmup_steps += steps
                stage_log['expert_warmup'].append({
                    'expert_index': expert_index, 'group': group,
                    'epochs': history,
                })

            new_head.requires_grad_(True)
            backbone_parameters = model.enable_incremental_finetuning()
            adaptive_parameters = list(new_head.parameters())
            adaptive_parameters.append(model.classifier.trainable_scale())
            for expert_index in expert_indices:
                model.experts[expert_index].requires_grad_(True)
                adaptive_parameters.extend(model.experts[expert_index].parameters())
            joint_parameters = [
                {'params': adaptive_parameters, 'lr': args.lr},
                {'params': backbone_parameters, 'lr': args.backbone_lr},
            ]
            joint_steps, history = train_epochs(
                model, train_loader, train_sampler, joint_parameters,
                args.incremental_epochs, device, use_amp, args.lr,
                args.weight_decay, old_classes, new_classes,
                args.competition_margin, args.lambda_competition,
                args.lambda_amplitude, teacher=previous_model,
                lambda_kd=args.lambda_kd, kd_temperature=args.kd_temperature,
                lambda_supcon=args.lambda_supcon,
                supcon_temperature=args.supcon_temperature,
                validation=(val_dataset, active_classes, order[:stage + 1],
                            args.batch_size, args.num_workers),
                early_stopping_patience=args.early_stopping_patience)
            stage_log['joint'] = history
            total_steps = head_steps + warmup_steps + joint_steps

            if expert_indices and old_items:
                check_dataset = DictDataset(old_items[:min(32, len(old_items))],
                                            physics_stats)
                # Check the exact algebra directly on deterministic residuals.
                check_raw = collate_numpy([check_dataset[i]
                                           for i in range(len(check_dataset))])
                check_batch = to_device(check_raw, device)
                model.eval()
                with torch.no_grad():
                    _, _, check_diag = model(
                        check_batch['x'], check_batch['order'],
                        check_batch['quality'], check_batch['rpm'])
                    raw_residuals = [raw for raw, expert in zip(
                        check_diag['raw_residuals'], model.experts)
                        if expert.stage == stage]
                    raw = (torch.stack(raw_residuals).sum(dim=0)
                           if raw_residuals else check_batch['x'].new_zeros(
                               (len(check_dataset), model.feature_dim)))
                    projection_max_error = projection_error(
                        protection.basis, old_weight, old_classes,
                        protection.selected, raw)

        val_result = evaluate(
            model, val_dataset, active_classes, order[:stage + 1], device,
            args.batch_size, args.num_workers)
        test_result = evaluate(
            model, test_dataset, active_classes, order[:stage + 1], device,
            args.batch_size, args.num_workers)
        mechanisms = mechanism_metrics(
            model, test_dataset, device, args.batch_size, args.num_workers)
        capture = 0.0
        if previous_model is not None:
            old_test = EchoCILDataset(args.data, 'test', old_classes)
            capture = capture_rate(
                previous_model, model, old_test, new_classes, device,
                args.batch_size, args.num_workers)

        memory.update(new_items, active_classes)
        memory_size = memory.save(args.output / f'memory_stage_{stage}.npy')
        checkpoint_path = args.output / f'stage_{stage}.pt'
        save_checkpoint(checkpoint_path, model, stage, configuration)
        model_bytes = checkpoint_path.stat().st_size
        result = {
            'stage': stage, 'classes': active_classes,
            'new_classes': list(new_classes), 'val': val_result,
            'test': test_result, 'capture_rate': capture,
            'mechanisms': mechanisms,
            'protection': {**protection_json,
                           'projection_max_error': projection_max_error},
            'conflict_scores': conflicts_json,
            'expert_groups': expert_groups,
            'experts_total': len(model.experts),
            'memory_count': len(memory.all()),
            'memory_counts': memory.counts(),
            'memory_bytes': memory_size,
            'model_bytes': model_bytes,
            'optimization_steps': total_steps,
            'training_seconds': time.perf_counter() - stage_started,
            'peak_cuda_bytes': (torch.cuda.max_memory_allocated(device)
                                if device.type == 'cuda' else 0),
            'parameters_total': sum(parameter.numel()
                                    for parameter in model.parameters()),
            'training_log': stage_log,
            'device': str(device), 'amp': use_amp,
        }
        stage_results.append(result)
        print(json.dumps({
            'stage': stage, 'test_macro': test_result['macro'],
            'old': test_result['old'], 'new': test_result['new'],
            'memory_bytes': memory_size,
            'protection_rank': protection_json.get('rank', 0),
        }, ensure_ascii=False))

    output = {
        'protocol': 'ECHO-CIL-optimized', 'order_name': args.order,
        'order': order, 'seed': args.seed, 'configuration': configuration,
        'stages': stage_results, 'summary': experiment_summary(stage_results),
    }
    (args.output / 'results.json').write_text(
        json.dumps(output, ensure_ascii=False, indent=2, default=str),
        encoding='utf-8')
    print(json.dumps(output['summary'], ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
