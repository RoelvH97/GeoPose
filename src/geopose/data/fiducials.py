"""Carotid-skeleton fiducials for the LXPose-style projection (mPD) loss."""

from __future__ import annotations

import os

import nibabel as nib
import numpy as np
import torch
from skimage.morphology import skeletonize


def carotid_skeleton_world(
    mask_path: str,
    labels: tuple[int, ...] = (1, 2),
    max_points: int | None = 128,
    cache_dir: str | None = None,
) -> torch.Tensor | None:
    """Skeletonise the carotid labels in ``mask_path`` → world-coordinate fiducials."""
    pid = os.path.basename(mask_path).split(".")[0]
    key = f"{pid}_lbl{'-'.join(map(str, labels))}_n{max_points}"
    cache_path = os.path.join(cache_dir, f"{key}.npy") if cache_dir else None
    if (
        cache_path
        and os.path.exists(cache_path)
        and os.path.getmtime(cache_path) >= os.path.getmtime(mask_path)
    ):
        world = np.load(cache_path)
        return torch.from_numpy(world)[None].float() if len(world) else None

    data = np.asarray(nib.load(mask_path).dataobj)
    carotid = np.isin(data, list(labels))
    if not carotid.any():
        return None
    ijk = np.argwhere(skeletonize(carotid))
    if len(ijk) == 0:
        return None
    if max_points is not None and len(ijk) > max_points:
        sel = np.linspace(0, len(ijk) - 1, max_points).round().astype(int)
        ijk = ijk[sel]
    homo = np.concatenate([ijk, np.ones((len(ijk), 1))], axis=1)
    affine = nib.load(mask_path).affine.astype(np.float64)
    world = (affine @ homo.T).T[:, :3].astype(np.float32)

    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_path, world)
    return torch.from_numpy(world)[None]
