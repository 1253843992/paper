from __future__ import annotations

import copy
from typing import Iterable, List

import torch
from torch import Tensor, nn
import torch.nn.functional as F


class DilatedResidualBlock(nn.Module):
    def __init__(self, channels: int, dilation: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, 7, padding=3 * dilation, dilation=dilation),
            nn.BatchNorm1d(channels), nn.GELU(), nn.Dropout(0.1),
            nn.Conv1d(channels, channels, 5, padding=2 * dilation, dilation=dilation),
            nn.BatchNorm1d(channels))
        self.act = nn.GELU()

    def forward(self, x: Tensor) -> Tensor:
        return self.act(x + self.net(x))


class SignalEncoder(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(1, 32, 15, stride=2, padding=7), nn.BatchNorm1d(32), nn.GELU())
        self.blocks = nn.Sequential(*[DilatedResidualBlock(32, d) for d in (1, 3, 7)])
        self.down = nn.Sequential(nn.Conv1d(32, 64, 5, stride=2, padding=2), nn.BatchNorm1d(64), nn.GELU(),
                                  nn.Conv1d(64, 96, 5, stride=2, padding=2), nn.BatchNorm1d(96), nn.GELU())
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(96, out_dim)

    def forward(self, x: Tensor) -> Tensor:
        x = self.down(self.blocks(self.stem(x.unsqueeze(1))))
        return self.fc(self.pool(x).flatten(1))


class PhysicsEncoder(nn.Module):
    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.spec = nn.Sequential(nn.Conv1d(3, 32, 9, padding=4), nn.BatchNorm1d(32), nn.GELU(),
                                  nn.Conv1d(32, 64, 7, padding=3), nn.GELU(), nn.AdaptiveAvgPool1d(1))
        self.quality = nn.Sequential(nn.Linear(9, 32), nn.LayerNorm(32), nn.GELU(), nn.Dropout(0.1))
        self.fc = nn.Sequential(nn.Linear(96, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def forward(self, order: Tensor, quality: Tensor) -> Tensor:
        a = self.spec(order).flatten(1)
        return self.fc(torch.cat([a, self.quality(quality)], dim=1))


class LowRankExpert(nn.Module):
    def __init__(self, dim: int = 128, rank: int = 8):
        super().__init__()
        self.down = nn.Linear(dim, rank, bias=False)
        self.up = nn.Linear(rank, dim, bias=False)
        self.norm = nn.LayerNorm(dim)
        # A newly appended expert must be a strict no-op.  This preserves the
        # previous-stage function until the new expert starts learning.
        nn.init.zeros_(self.up.weight)

    def forward(self, z: Tensor) -> Tensor:
        return self.up(F.gelu(self.down(self.norm(z))))


class ECHOModel(nn.Module):
    """ECHO-CIL encoder, gated low-rank experts and expandable classifier."""
    def __init__(self, num_classes: int = 4, feature_dim: int = 128, rank: int = 8):
        super().__init__()
        self.signal = SignalEncoder(feature_dim)
        self.physics = PhysicsEncoder(feature_dim)
        self.fusion = nn.Sequential(nn.Linear(feature_dim * 2, feature_dim), nn.LayerNorm(feature_dim), nn.GELU())
        self.gate = nn.Linear(feature_dim, 1)
        self.experts = nn.ModuleList([LowRankExpert(feature_dim, rank)])
        self.expert_gate = nn.Linear(feature_dim, 1)
        self.classifier = nn.Linear(feature_dim, num_classes)
        self.feature_dim = feature_dim
        self.rank = rank

    @property
    def num_classes(self) -> int:
        return self.classifier.out_features

    def encode(self, x: Tensor, order: Tensor, quality: Tensor) -> Tensor:
        zs, zp = self.signal(x), self.physics(order, quality)
        # A learned scalar gate chooses how much physical information enters.
        alpha = torch.sigmoid(self.gate(zs))
        return self.fusion(torch.cat([zs, alpha * zp], dim=1))

    def forward(self, x: Tensor, order: Tensor, quality: Tensor):
        z = self.encode(x, order, quality)
        deltas = torch.stack([e(z) for e in self.experts], dim=1)
        # Experts produce residual deltas only.  Do not divide by the number
        # of experts: adding a zero-initialized expert must not rescale old
        # expert contributions or change old predictions.
        scores = torch.sigmoid(self.expert_gate(z))
        z = z + (deltas * scores.unsqueeze(-1)).sum(1)
        return self.classifier(z), z

    def add_expert(self) -> None:
        for p in self.experts.parameters():
            p.requires_grad_(False)
        self.experts.append(LowRankExpert(self.feature_dim, self.rank).to(next(self.parameters()).device))
        old = self.expert_gate
        new = nn.Linear(self.feature_dim, old.out_features + 1).to(old.weight.device)
        with torch.no_grad():
            new.weight[:old.out_features].copy_(old.weight)
            new.bias[:old.out_features].copy_(old.bias)
            nn.init.zeros_(new.weight[old.out_features:]); nn.init.zeros_(new.bias[old.out_features:])
        self.expert_gate = new

    def expand_classes(self, new_num_classes: int) -> None:
        if new_num_classes <= self.num_classes:
            return
        old = self.classifier
        new = nn.Linear(self.feature_dim, new_num_classes).to(old.weight.device)
        with torch.no_grad():
            new.weight[:old.out_features].copy_(old.weight)
            new.bias[:old.out_features].copy_(old.bias)
            nn.init.normal_(new.weight[old.out_features:], 0, 0.02)
            nn.init.zeros_(new.bias[old.out_features:])
        self.classifier = new

    def frozen_copy(self) -> "ECHOModel":
        model = copy.deepcopy(self)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        return model
