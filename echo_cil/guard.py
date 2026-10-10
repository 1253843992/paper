"""SVD discriminative-subspace protection for ECHO-CIL.

This module is intentionally standalone so it can be copied directly into
the project's ``echo_cil`` package.
"""

from __future__ import annotations

import torch
from torch import Tensor


class SubspaceGuard:
    """Protect old discriminative feature directions during an increment.

    The basis is estimated with SVD/PCA from old-model features. During the
    next stage, only the component of representation drift lying in this
    basis is penalized. If no basis is available, the penalty is zero.
    """

    def __init__(self, rank: int = 16, strength: float = 1.0):
        self.rank = int(rank)
        self.strength = float(strength)
        self.basis: Tensor | None = None

    @torch.no_grad()
    def fit(self, features: Tensor) -> None:
        """Fit an orthonormal basis from a feature matrix [N, D]."""
        if features.ndim != 2:
            raise ValueError(f"features must have shape [N, D], got {tuple(features.shape)}")

        if features.shape[0] < 2:
            self.basis = torch.eye(features.shape[1], device=features.device,
                                   dtype=features.dtype)
            return

        centered = features - features.mean(dim=0, keepdim=True)
        q = min(self.rank, features.shape[1], max(1, features.shape[0] - 1))
        _, _, v = torch.pca_lowrank(centered, q=q, center=False)
        self.basis = v[:, :q].detach()

    def penalty(self, old_z: Tensor, new_z: Tensor) -> Tensor:
        """Return projected representation-drift penalty."""
        if old_z.shape != new_z.shape:
            raise ValueError("old_z and new_z must have the same shape")
        if self.basis is None:
            return new_z.new_zeros(())

        delta = new_z - old_z
        projected = delta @ self.basis @ self.basis.transpose(0, 1)
        return self.strength * projected.pow(2).mean()

