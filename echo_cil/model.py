from __future__ import annotations

import copy
import math
from typing import Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .physics import (
    FS_TARGET, HARMONIC_NUMBERS, HARMONIC_ORDERS, HARMONIC_TYPES,
    ORDER_MAX, ORDER_POINTS, WINDOW,
)


class ConvGNAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, kernel: int,
                 stride: int = 1, padding: int = 0):
        super().__init__(
            nn.Conv1d(in_channels, out_channels, kernel, stride=stride,
                      padding=padding, bias=False),
            nn.GroupNorm(8, out_channels), nn.SiLU(),
        )


class QualityGate(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(9, 16), nn.SiLU(), nn.Linear(16, 1))

    def forward(self, quality: Tensor) -> Tensor:
        return torch.sigmoid(self.net(quality))


class HarmonicEvidence(nn.Module):
    """Three-band order encoder plus 21 distance-biased harmonic queries."""

    def __init__(self, dim: int = 128, heads: int = 4):
        super().__init__()
        if dim % heads:
            raise ValueError('feature dimension must be divisible by heads')
        self.dim, self.heads, self.head_dim = dim, heads, dim // heads
        self.encoder = nn.Sequential(
            ConvGNAct(3, 32, 7, padding=3),
            ConvGNAct(32, 64, 5, stride=2, padding=2),
            ConvGNAct(64, 128, 5, stride=2, padding=2),
        )
        self.type_embedding = nn.Embedding(7, dim)
        self.harmonic_embedding = nn.Embedding(3, dim)
        self.position_projection = nn.Linear(dim, dim)
        self.query_projection = nn.Linear(dim, dim)
        self.key_projection = nn.Linear(dim, dim)
        self.value_projection = nn.Linear(dim, dim)
        self.output_projection = nn.Linear(dim, dim)
        self.offset_raw = nn.Parameter(torch.zeros(21))
        self.width_raw = nn.Parameter(torch.full(
            (21,), math.log(math.expm1(0.2))))
        self.register_buffer('anchor_orders', torch.from_numpy(HARMONIC_ORDERS))
        self.register_buffer('anchor_types', torch.from_numpy(HARMONIC_TYPES))
        self.register_buffer('harmonic_numbers', torch.from_numpy(HARMONIC_NUMBERS))
        token_orders = torch.arange(64, dtype=torch.float32) * 4.0 * ORDER_MAX / (ORDER_POINTS - 1)
        self.register_buffer('token_orders', token_orders)

    def _position_encoding(self) -> Tensor:
        frequencies = torch.exp(torch.linspace(
            0.0, math.log(1000.0), self.dim // 2,
            device=self.anchor_orders.device))
        phase = self.anchor_orders.unsqueeze(1) / frequencies.unsqueeze(0)
        return torch.cat((torch.sin(phase), torch.cos(phase)), dim=1)

    def forward(self, order: Tensor, rpm: Tensor):
        tokens = self.encoder(order).transpose(1, 2)  # [B, 64, 128]
        queries = (self.type_embedding(self.anchor_types)
                   + self.harmonic_embedding(self.harmonic_numbers)
                   + self.position_projection(self._position_encoding()))
        queries = self.query_projection(queries).view(21, self.heads, self.head_dim)
        keys = self.key_projection(tokens).view(
            tokens.shape[0], 64, self.heads, self.head_dim)
        values = self.value_projection(tokens).view(
            tokens.shape[0], 64, self.heads, self.head_dim)
        logits = torch.einsum('qhd,bkhd->bhqk', queries, keys)
        logits = logits / math.sqrt(self.head_dim)

        shaft_hz = rpm.float().clamp_min(1.0) / 60.0
        spectral_floor = FS_TARGET / (WINDOW * shaft_hz)
        receptive_floor = order.new_tensor(9.0 * ORDER_MAX / (ORDER_POINTS - 1))
        sigma_min = torch.maximum(spectral_floor, receptive_floor)
        sigma = sigma_min[:, None] + F.softplus(self.width_raw)[None, :]
        centers = self.anchor_orders + 0.25 * torch.tanh(self.offset_raw)
        distance = self.token_orders[None, None, :] - centers[None, :, None]
        bias = -(distance ** 2) / (2.0 * sigma[:, :, None] ** 2)
        attention = torch.softmax(logits + bias[:, None, :, :], dim=-1)
        evidence = torch.einsum('bhqk,bkhd->bqhd', attention, values)
        evidence = self.output_projection(evidence.reshape(order.shape[0], 21, self.dim))
        return evidence, attention


class ConditionalScaleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.downsample = ConvGNAct(in_channels, out_channels, 3,
                                    stride=2, padding=1)
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(out_channels, out_channels, 5, padding=2 * dilation,
                          dilation=dilation, groups=out_channels, bias=False),
                nn.GroupNorm(8, out_channels), nn.SiLU(),
            ) for dilation in (1, 3, 7)
        ])
        self.controller = nn.Sequential(
            nn.Linear(128, 32), nn.SiLU(), nn.Linear(32, 3))
        self.pointwise = nn.Conv1d(out_channels, out_channels, 1, bias=False)
        self.output_norm = nn.GroupNorm(8, out_channels)

    def forward(self, x: Tensor, evidence_mean: Tensor, gamma: Tensor):
        trunk = self.downsample(x)
        conditional = torch.softmax(self.controller(evidence_mean), dim=-1)
        alpha = (1.0 - gamma) / 3.0 + gamma * conditional
        branches = torch.stack([branch(trunk) for branch in self.branches], dim=1)
        mixed = (branches * alpha[:, :, None, None]).sum(dim=1)
        output = F.silu(self.output_norm(trunk + self.pointwise(mixed)))
        return output, alpha


class TimeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = ConvGNAct(1, 32, 15, stride=4, padding=7)
        self.blocks = nn.ModuleList([
            ConditionalScaleBlock(32, 64),
            ConditionalScaleBlock(64, 96),
            ConditionalScaleBlock(96, 128),
        ])

    def forward(self, x: Tensor, evidence: Tensor, gamma: Tensor):
        x = self.stem(x.unsqueeze(1))
        evidence_mean = evidence.mean(dim=1)
        weights = []
        for block in self.blocks:
            x, alpha = block(x, evidence_mean, gamma)
            weights.append(alpha)
        return x.transpose(1, 2), weights  # [B, 64, 128]


class CrossDomainFusion(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.time_norm = nn.LayerNorm(dim)
        self.evidence_norm = nn.LayerNorm(dim)
        self.attention = nn.MultiheadAttention(dim, 4, batch_first=True)
        self.ffn_norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 256), nn.SiLU(), nn.Linear(256, dim))
        self.output = nn.Linear(dim, dim)

    def forward(self, time_tokens: Tensor, evidence: Tensor, gamma: Tensor):
        interaction, cross_attention = self.attention(
            self.time_norm(time_tokens), self.evidence_norm(evidence),
            self.evidence_norm(evidence), need_weights=True,
            average_attn_weights=False)
        fused = time_tokens + gamma[:, None, :] * interaction
        fused = fused + self.ffn(self.ffn_norm(fused))
        return self.output(fused.mean(dim=1)), cross_attention


class ProtectedExpert(nn.Module):
    def __init__(self, basis: Tensor, stage: int, group: Sequence[int],
                 dim: int = 128, rank: int = 8):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.gate = nn.Sequential(
            nn.Linear(dim + 1, 16), nn.SiLU(), nn.Linear(16, 1))
        nn.init.zeros_(self.up.weight)
        self.register_buffer('basis', basis.detach().float().clone())
        self.stage = int(stage)
        self.group = tuple(int(value) for value in group)

    def forward(self, h: Tensor, gamma: Tensor):
        # The technical plan requires the projection itself to remain FP32
        # even when the surrounding training step uses CUDA autocast.
        with torch.autocast(device_type=h.device.type, enabled=False):
            h32, gamma32 = h.float(), gamma.float()
            raw = self.up(F.silu(self.down(h32)))
            if self.basis.numel():
                projected = raw - (raw @ self.basis) @ self.basis.transpose(0, 1)
            else:
                projected = raw
            gate = torch.sigmoid(self.gate(torch.cat((h32, gamma32), dim=1)))
        return gate * projected, gate, raw


class IncrementalClassifier(nn.Module):
    def __init__(self, feature_dim: int, base_classes: Sequence[int],
                 initial_scale: float = 16.0):
        super().__init__()
        self.feature_dim = feature_dim
        self.heads = nn.ModuleList([nn.Linear(feature_dim, len(base_classes))])
        self.class_chunks: List[List[int]] = [list(map(int, base_classes))]
        self.logit_scale = nn.Parameter(torch.tensor(math.log(initial_scale)))

    @property
    def class_ids(self) -> List[int]:
        return [label for chunk in self.class_chunks for label in chunk]

    def forward(self, z: Tensor) -> Tensor:
        features = F.normalize(z.float(), dim=1)
        scale = self.logit_scale.exp().clamp(max=100.0)
        chunks = [scale * F.linear(features, F.normalize(head.weight.float(), dim=1))
                  + head.bias.float() for head in self.heads]
        return torch.cat(chunks, dim=1)

    def add_classes(self, classes: Sequence[int]) -> nn.Linear:
        for head in self.heads:
            head.requires_grad_(False)
        head = nn.Linear(self.feature_dim, len(classes)).to(self.heads[0].weight.device)
        self.heads.append(head)
        self.class_chunks.append(list(map(int, classes)))
        return head

    def freeze_all(self) -> None:
        self.requires_grad_(False)

    def trainable_scale(self) -> nn.Parameter:
        return self.logit_scale

    def weight_bias(self) -> tuple[Tensor, Tensor]:
        return (torch.cat([head.weight for head in self.heads], dim=0),
                torch.cat([head.bias for head in self.heads], dim=0))

    def target_indices(self, labels: Tensor) -> Tensor:
        mapping = {label: index for index, label in enumerate(self.class_ids)}
        values = [mapping[int(label)] for label in labels.detach().cpu().tolist()]
        return torch.as_tensor(values, device=labels.device, dtype=torch.long)


class ECHOModel(nn.Module):
    """Strict ECHO-CIL network: conditional encoder plus protected experts."""

    def __init__(self, base_classes: Sequence[int], feature_dim: int = 128,
                 expert_rank: int = 8, expert_limit: int = 4):
        super().__init__()
        self.evidence = HarmonicEvidence(feature_dim, heads=4)
        self.quality_gate = QualityGate()
        self.time = TimeEncoder()
        self.fusion = CrossDomainFusion(feature_dim)
        self.experts = nn.ModuleList()
        self.classifier = IncrementalClassifier(feature_dim, base_classes)
        self.register_buffer('prototype_vectors', torch.empty(0, feature_dim))
        self.register_buffer('prototype_labels', torch.empty(0, dtype=torch.long))
        self.register_buffer('prototype_rpms', torch.empty(0))
        self.register_buffer('prototype_weight', torch.tensor(0.0))
        self.register_buffer('prototype_neighbors', torch.tensor(1, dtype=torch.long))
        self.register_buffer('prototype_rpm_penalty', torch.tensor(0.0))
        self.feature_dim = feature_dim
        self.expert_rank = expert_rank
        self.expert_limit = expert_limit
        self.current_stage = 0

    @property
    def class_ids(self) -> List[int]:
        return self.classifier.class_ids

    @property
    def num_classes(self) -> int:
        return len(self.class_ids)

    def base_features(self, x: Tensor, order: Tensor, quality: Tensor, rpm: Tensor):
        gamma = self.quality_gate(quality)
        evidence, harmonic_attention = self.evidence(order, rpm)
        time_tokens, scale_weights = self.time(x, evidence, gamma)
        h, cross_attention = self.fusion(time_tokens, evidence, gamma)
        diagnostics = {
            'gamma': gamma, 'harmonic_attention': harmonic_attention,
            'scale_weights': scale_weights, 'cross_attention': cross_attention,
        }
        return h, gamma, diagnostics

    def representation(self, x: Tensor, order: Tensor, quality: Tensor,
                       rpm: Tensor, enabled_current: Optional[Iterable[int]] = None):
        h, gamma, diagnostics = self.base_features(x, order, quality, rpm)
        z = h
        current_delta = torch.zeros_like(h)
        gates, raw_residuals = [], []
        enabled = None if enabled_current is None else set(enabled_current)
        for index, expert in enumerate(self.experts):
            is_current = expert.stage == self.current_stage
            if is_current and enabled is not None and index not in enabled:
                continue
            delta, gate, raw = expert(h, gamma)
            z = z + delta
            if is_current:
                current_delta = current_delta + delta
            gates.append(gate); raw_residuals.append(raw)
        diagnostics['expert_gates'] = gates
        diagnostics['raw_residuals'] = raw_residuals
        diagnostics['current_delta'] = current_delta
        diagnostics['h'] = h
        return z, diagnostics

    def forward(self, x: Tensor, order: Tensor, quality: Tensor, rpm: Tensor,
                enabled_current: Optional[Iterable[int]] = None):
        z, diagnostics = self.representation(
            x, order, quality, rpm, enabled_current=enabled_current)
        logits = self.classifier(z)
        if (not self.training and self.prototype_vectors.numel()
                and float(self.prototype_weight) > 0.0):
            similarities = (F.normalize(z.float(), dim=1)
                            @ F.normalize(self.prototype_vectors.float(), dim=1).t())
            if self.prototype_rpms.numel():
                rpm_distance = ((rpm.float()[:, None]
                                 - self.prototype_rpms.float()[None, :]).abs()
                                / rpm.float()[:, None].clamp_min(1.0))
                similarities = (similarities
                                - self.prototype_rpm_penalty * rpm_distance)
            scale = self.classifier.logit_scale.exp().clamp(max=100.0)
            prototype_logits = logits.new_full(logits.shape, -1e4)
            positions = {label: index for index, label in enumerate(self.class_ids)}
            for label in self.prototype_labels.unique().tolist():
                if int(label) in positions:
                    mask = self.prototype_labels == int(label)
                    class_similarities = similarities[:, mask]
                    neighbors = min(
                        int(self.prototype_neighbors), class_similarities.shape[1])
                    prototype_logits[:, positions[int(label)]] = (
                        scale * class_similarities.topk(
                            neighbors, dim=1).values.mean(dim=1))
            weight = self.prototype_weight.to(logits)
            logits = (1.0 - weight) * logits + weight * prototype_logits
        return logits, z, diagnostics

    def set_prototypes(self, vectors: Tensor, labels: Tensor, rpms: Tensor,
                       weight: float = 0.5) -> None:
        self.prototype_vectors = vectors.detach().float().to(
            next(self.parameters()).device)
        self.prototype_labels = labels.detach().long().to(
            next(self.parameters()).device)
        self.prototype_rpms = rpms.detach().float().to(
            next(self.parameters()).device)
        self.prototype_weight.fill_(float(weight))

    def set_prototype_weight(self, weight: float) -> None:
        self.prototype_weight.fill_(float(weight))

    def set_prototype_options(self, neighbors: int,
                              rpm_penalty: float) -> None:
        self.prototype_neighbors.fill_(int(neighbors))
        self.prototype_rpm_penalty.fill_(float(rpm_penalty))

    def clear_prototypes(self) -> None:
        device = next(self.parameters()).device
        self.prototype_vectors = torch.empty(
            0, self.feature_dim, device=device)
        self.prototype_labels = torch.empty(0, dtype=torch.long, device=device)
        self.prototype_rpms = torch.empty(0, device=device)
        self.prototype_weight.zero_()
        self.prototype_neighbors.fill_(1)
        self.prototype_rpm_penalty.zero_()

    def freeze_base(self) -> None:
        for module in (self.evidence, self.quality_gate, self.time, self.fusion):
            module.requires_grad_(False)
            module.eval()

    def enable_incremental_finetuning(self) -> List[nn.Parameter]:
        """Unfreeze only high-level feature layers for incremental plasticity."""
        modules = (self.time.blocks[-1], self.fusion)
        parameters: List[nn.Parameter] = []
        for module in modules:
            module.requires_grad_(True)
            module.train()
            parameters.extend(module.parameters())
        return parameters

    def begin_increment(self, new_classes: Sequence[int], stage: int) -> nn.Linear:
        self.current_stage = int(stage)
        self.clear_prototypes()
        self.freeze_base()
        for expert in self.experts:
            expert.requires_grad_(False)
            expert.eval()
        return self.classifier.add_classes(new_classes)

    def add_expert(self, basis: Tensor, group: Sequence[int]) -> int:
        if len(self.experts) >= self.expert_limit:
            raise RuntimeError('expert budget exhausted')
        expert = ProtectedExpert(
            basis, self.current_stage, group, self.feature_dim, self.expert_rank)
        expert = expert.to(next(self.parameters()).device)
        self.experts.append(expert)
        return len(self.experts) - 1

    def frozen_copy(self) -> 'ECHOModel':
        model = copy.deepcopy(self)
        model.eval()
        model.requires_grad_(False)
        return model

    def architecture_metadata(self) -> Dict[str, object]:
        return {
            'class_chunks': self.classifier.class_chunks,
            'expert_stages': [expert.stage for expert in self.experts],
            'expert_groups': [list(expert.group) for expert in self.experts],
            'expert_limit': self.expert_limit,
            'feature_dim': self.feature_dim,
            'expert_rank': self.expert_rank,
            'prototype_count': int(self.prototype_vectors.shape[0]),
            'prototype_weight': float(self.prototype_weight),
            'prototype_neighbors': int(self.prototype_neighbors),
            'prototype_rpm_penalty': float(self.prototype_rpm_penalty),
        }
