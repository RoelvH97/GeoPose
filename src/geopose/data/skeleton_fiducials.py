"""Carotid-skeleton fiducials for the LXPose-style projection (mPD) loss.

LXPose supervises pose with a *projection loss*: implanted fiducial markers are
projected under the GT and predicted poses and the mean 2D reprojection error
(mm at the detector) is minimised (``LXPose/train.py`` ``mpd1``/``mpd2``). The
carotid task has no markers, so we synthesise a fiducial set by **skeletonising
the carotid arteries** and treating the centreline voxels as distributed
landmarks.

The returned points are in the volume's *original* world frame (RAS mm, i.e.
``affine @ voxel``). Pass them straight to :func:`diffdrr.data.read` as
``fiducials=``; ``read``'s ``canonicalize`` step reorients them into the same
isocentre-centred frame the DRR renders in, after which
``drr.perspective_projection(pose, subject.fiducials)`` yields detector pixels.
Verified end-to-end on real CTA (identical pose → mPD ≈ 0, gradients flow).
"""

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
    """Skeletonise the carotid labels in ``mask_path`` → world-coordinate fiducials.

    Args:
        mask_path:  co-registered carotid label volume (same grid as the CT).
        labels:     label values unioned before skeletonising (CTA carotid = 1, 2).
        max_points: even-subsample the skeleton to at most this many points
                    (``None`` = keep all). Bounds the projection loss's point count.
        cache_dir:  optional directory for a per-patient ``.npy`` cache of the
                    world points (keyed by patient id + labels + max_points, and
                    invalidated when the mask is newer). ``None`` = recompute.

    Returns:
        ``[1, N, 3]`` float32 world coordinates (RAS mm, original volume frame),
        or ``None`` if the carotid mask / skeleton is empty.
    """
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
    ijk = np.argwhere(skeletonize(carotid))                     # [M, 3] voxel indices
    if len(ijk) == 0:
        return None
    if max_points is not None and len(ijk) > max_points:
        sel = np.linspace(0, len(ijk) - 1, max_points).round().astype(int)
        ijk = ijk[sel]
    homo = np.concatenate([ijk, np.ones((len(ijk), 1))], axis=1)          # [N, 4]
    affine = nib.load(mask_path).affine.astype(np.float64)
    world = (affine @ homo.T).T[:, :3].astype(np.float32)                 # [N, 3] RAS mm

    if cache_path:
        os.makedirs(cache_dir, exist_ok=True)
        np.save(cache_path, world)
    return torch.from_numpy(world)[None]                                  # [1, N, 3]
