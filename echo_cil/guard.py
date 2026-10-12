"""Reliable old-relation selection and exact classifier-difference basis."""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor


@dataclass
class ProtectionResult:
    basis: Tensor
    candidates: List[dict]
    selected: List[dict]
    singular_values: List[float]
    rank: int
    relative_residual: float
    valid_per_class: Dict[str, int]

    def to_json(self) -> dict:
        return {
            'candidate_edges': self.candidates,
            'selected_edges': self.selected,
            'singular_values': self.singular_values,
            'rank': self.rank,
            'relative_residual': self.relative_residual,
            'valid_per_class': self.valid_per_class,
        }


def reliable_edges(logits: Tensor, labels: Tensor, class_ids: Sequence[int],
                   classifier_weight: Tensor, min_samples: int = 5,
                   lower_quantile: float = 0.10) -> Tuple[List[dict], Dict[str, int]]:
    """Select positive low-quantile margins from teacher-correct replay.

    ``logits`` has shape [augmentations, samples, old_classes].  Correctness is
    decided on the unperturbed first view; reliability uses every fixed view.
    """
    logits = logits.detach().cpu().double()
    labels = labels.detach().cpu().long()
    class_ids = list(map(int, class_ids))
    index = {label: position for position, label in enumerate(class_ids)}
    targets = torch.as_tensor([index[int(value)] for value in labels.tolist()])
    correct = logits[0].argmax(dim=1) == targets
    valid = {label: torch.where(correct & (labels == label))[0]
             for label in class_ids}
    counts = {str(label): int(valid[label].numel()) for label in class_ids}
    candidates = []
    weight = classifier_weight.detach().cpu().double()
    for left_pos, left in enumerate(class_ids):
        for right in class_ids[left_pos + 1:]:
            if valid[left].numel() < min_samples or valid[right].numel() < min_samples:
                continue
            i, j = index[left], index[right]
            norm = float(torch.linalg.vector_norm(weight[i] - weight[j]))
            if norm <= 1e-12:
                continue
            left_margin = ((logits[:, valid[left], i]
                            - logits[:, valid[left], j]) / norm).reshape(-1)
            right_margin = ((logits[:, valid[right], j]
                             - logits[:, valid[right], i]) / norm).reshape(-1)
            reliability = min(
                float(torch.quantile(left_margin, lower_quantile)),
                float(torch.quantile(right_margin, lower_quantile)),
            )
            if reliability > 0:
                candidates.append({'i': left, 'j': right,
                                   'reliability': reliability})
    return candidates, counts


def maximum_weight_matching(nodes: Sequence[int], edges: Sequence[dict]) -> List[dict]:
    """Exact deterministic maximum-weight matching for the small CWRU graph."""
    nodes = tuple(sorted(map(int, nodes)))
    edge_map = {
        tuple(sorted((int(edge['i']), int(edge['j'])))): edge
        for edge in edges
    }

    @lru_cache(maxsize=None)
    def solve(remaining: Tuple[int, ...]):
        if not remaining:
            return 0.0, ()
        first = remaining[0]
        best_weight, best_pairs = solve(remaining[1:])
        for position, other in enumerate(remaining[1:], start=1):
            pair = (first, other)
            if pair not in edge_map:
                continue
            reduced = remaining[1:position] + remaining[position + 1:]
            weight, pairs = solve(reduced)
            candidate_weight = weight + float(edge_map[pair]['reliability'])
            candidate_pairs = tuple(sorted((pair,) + pairs))
            if (candidate_weight > best_weight + 1e-15 or
                    (abs(candidate_weight - best_weight) <= 1e-15
                     and candidate_pairs < best_pairs)):
                best_weight, best_pairs = candidate_weight, candidate_pairs
        return best_weight, best_pairs

    _, pairs = solve(nodes)
    return [dict(edge_map[pair]) for pair in pairs]


def classifier_difference_basis(classifier_weight: Tensor,
                                class_ids: Sequence[int],
                                selected_edges: Sequence[dict]) -> Tuple[Tensor, List[float], float]:
    feature_dim = classifier_weight.shape[1]
    if not selected_edges:
        return classifier_weight.new_zeros((feature_dim, 0)), [], 0.0
    positions = {label: index for index, label in enumerate(class_ids)}
    weight = classifier_weight.detach().cpu().double()
    difference = torch.stack([
        weight[positions[int(edge['i'])]] - weight[positions[int(edge['j'])]]
        for edge in selected_edges
    ], dim=1)
    u, singular, _ = torch.linalg.svd(difference, full_matrices=False)
    threshold = (torch.finfo(torch.float64).eps
                 * max(difference.shape) * singular.max())
    rank = int((singular > threshold).sum())
    basis = u[:, :rank]
    reconstruction = basis @ (basis.transpose(0, 1) @ difference)
    residual = float(torch.linalg.matrix_norm(difference - reconstruction)
                     / torch.linalg.matrix_norm(difference).clamp_min(1e-12))
    return basis.float().to(classifier_weight.device), singular.tolist(), residual


def build_protection(logits: Tensor, labels: Tensor, class_ids: Sequence[int],
                     classifier_weight: Tensor, min_samples: int = 5,
                     lower_quantile: float = 0.10) -> ProtectionResult:
    candidates, counts = reliable_edges(
        logits, labels, class_ids, classifier_weight,
        min_samples=min_samples, lower_quantile=lower_quantile)
    selected = maximum_weight_matching(class_ids, candidates)
    basis, singular, residual = classifier_difference_basis(
        classifier_weight, class_ids, selected)
    return ProtectionResult(
        basis=basis, candidates=candidates, selected=selected,
        singular_values=[float(value) for value in singular],
        rank=basis.shape[1], relative_residual=residual,
        valid_per_class=counts,
    )


@torch.no_grad()
def projection_error(basis: Tensor, classifier_weight: Tensor,
                     class_ids: Sequence[int], selected_edges: Sequence[dict],
                     residuals: Tensor) -> float:
    if not selected_edges or residuals.numel() == 0:
        return 0.0
    positions = {label: index for index, label in enumerate(class_ids)}
    projected = residuals
    if basis.numel():
        projected = residuals - (residuals @ basis) @ basis.transpose(0, 1)
    errors = []
    for edge in selected_edges:
        difference = (classifier_weight[positions[int(edge['i'])]]
                      - classifier_weight[positions[int(edge['j'])]])
        errors.append((projected @ difference).abs().max())
    return float(torch.stack(errors).max())
