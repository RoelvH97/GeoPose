"""Shared, parameter-free helpers reused by GeoPose (:class:`ResNetPose`) and
the xvr baseline (:class:`XVRPose`).

Everything here is a pure function with **no** ``nn.Parameter`` / buffer, so
importing or calling these does not touch any LightningModule's ``state_dict``.
That is what lets the xvr baseline reuse GeoPose's plumbing without changing
either model's checkpoint layout — GeoPose checkpoints keep loading
bit-identically.

These were previously duplicated inline inside both modules' wandb logging
panels (``_to_np``, the Euler-ZYX extraction block, and the green/red/yellow
seg-overlay construction).
"""

from __future__ import annotations

import numpy as np
import torch


def to_np(x: torch.Tensor) -> np.ndarray:
    """Detached ``float`` CPU numpy view of a tensor (for matplotlib / wandb).

    Only ever called from ``@torch.no_grad`` logging hooks, so the ``detach``
    is a formality — it keeps the helper safe to call from any context.
    """
    return x.float().detach().cpu().numpy()


def euler_zyx_from_matrix(rot_mat: torch.Tensor) -> np.ndarray:
    """Extract ``(alpha, beta, gamma)`` Euler-ZYX angles (rad) from a 3x3
    rotation matrix.

    Byte-faithful port of the extraction block shared by ``ResNetPose`` and
    ``XVRPose`` pose panels (including the gimbal-lock ``cb ≈ 0`` fallback).
    Returns a length-3 numpy array.
    """
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
    """Build the MAP seg-overlay RGB image used by both map panels.

    ``bg`` is a 2-D (H, W) background image (e.g. the DSA MAP channel); it is
    min-max normalised to [0, 1] and tinted: reference seg → green, predicted
    artery → red, overlap → yellow. ``ref_mask`` / ``pred_mask`` are boolean
    (H, W) arrays. Returns an (H, W, 3) float array.
    """
    bg = (bg - bg.min()) / (bg.max() - bg.min() + 1e-8)
    rgb = np.stack([bg, bg, bg], axis=-1)
    rgb[ref_mask, 1] = np.clip(rgb[ref_mask, 1] + 0.6, 0, 1)   # green = reference seg
    rgb[pred_mask, 0] = np.clip(rgb[pred_mask, 0] + 0.6, 0, 1)  # red   = predicted artery
    overlap = ref_mask & pred_mask
    rgb[overlap, 0] = 1.0
    rgb[overlap, 1] = 1.0
    rgb[overlap, 2] = 0.0                                       # yellow = overlap
    return rgb
