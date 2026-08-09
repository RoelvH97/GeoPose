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
        gamma = torch.zeros(1)
    return torch.stack([alpha, beta, gamma]).numpy()


def seg_overlay_rgb(
    bg: np.ndarray,
    ref_mask: np.ndarray,
    pred_mask: np.ndarray,
) -> np.ndarray:
    """Build the MAP seg-overlay RGB image used by both map panels."""
    bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)
    rgb = np.stack([bg, bg, bg], axis=-1)
    rgb[ref_mask, 1] = np.clip(rgb[ref_mask, 1] + 0.6, 0, 1)
    rgb[pred_mask, 0] = np.clip(rgb[pred_mask, 0] + 0.6, 0, 1)
    overlap = ref_mask & pred_mask
    rgb[overlap, 0] = 1.0
    rgb[overlap, 1] = 1.0
    rgb[overlap, 2] = 0.0
    return rgb
