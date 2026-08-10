"""Tensor and image visualization helpers."""

from __future__ import annotations

import numpy as np
import torch


def to_np(x: torch.Tensor) -> np.ndarray:
    """Detached ``float`` CPU numpy view of a tensor (for matplotlib / wandb)."""
    return x.float().detach().cpu().numpy()


def euler_zyx_from_matrix(rot_mat: torch.Tensor) -> np.ndarray:
    """Extract Euler-ZYX angles from rotation matrices."""
    beta = torch.arcsin(torch.clamp(-rot_mat[2, 0], -1.0, 1.0))
    cb = torch.cos(beta)
    if cb.abs() > 1e-6:
        alpha = torch.arctan2(rot_mat[1, 0] / cb, rot_mat[0, 0] / cb)
        gamma = torch.arctan2(rot_mat[2, 1] / cb, rot_mat[2, 2] / cb)
    else:
        alpha = torch.arctan2(-rot_mat[0, 1], rot_mat[1, 1])
        gamma = torch.zeros_like(beta)
    return torch.stack([alpha, beta, gamma]).detach().cpu().numpy()
