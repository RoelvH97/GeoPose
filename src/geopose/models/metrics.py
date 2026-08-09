"""Shared map-vs-prediction metrics used by both ResNetPose and the xvr baseline."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


def mpcd(
    pred_arts: torch.Tensor,          # [K, C, H, W] bool — C candidate vessels (2 carotids for CTA)
    valid_segs: torch.Tensor,         # [K, 1, H, W] float (>0.5 = artery)
    delx: float,
    dely: float,
    side: torch.Tensor | None = None, # [K] long — channel of the injected vessel, if known
) -> float | None:
    """Mean Projected Centerline Distance (mm), matching GeoReg Eq. 9.

    For each sample k: skeletonize the reference DSA seg and the projected
    artery, then average over the *projected* centerline the distance to the
    nearest *reference* point, using physical pixel spacing so the result is mm.

    Direction (projection → reference) is deliberate and is the direction
    reported in the GeoReg paper: DSA resolves more distal vessel than CTA can,
    so averaging over the reference would charge the pose for anatomy the CTA
    cannot represent.

    Side selection: contrast injection is unilateral, so the DSA holds ONE
    carotid while the projection holds both. Averaging over both projected
    carotids at once would evaluate the unopacified side, which has no reference
    counterpart by construction. Each candidate vessel in `pred_arts` is scored
    separately against the reference and only one is kept:

      * `side` given  — use that channel (the injected vessel, identified
        externally). Preferred: it cannot be fooled by a mirrored pose.
      * `side` None   — best correspondence (minimum over channels). NOTE this
        can MASK a left-right laterality error: a mirrored pose still matches the
        contralateral carotid and scores well. Pass `side` when scoring anything
        laterality-sensitive (e.g. the view-anchor ablation).

    C == 1 degenerates to a plain single-vessel score.

    Returns None when no sample yields skeletons on both sides.
    """
    distances = []
    for k in range(pred_arts.shape[0]):
        seg_mask = valid_segs[k, 0].cpu().numpy() > 0.5
        seg_skel = skeletonize(seg_mask)
        if seg_skel.sum() == 0:
            continue

        # Distance to the nearest reference point — depends only on the seg, so
        # it is computed once and reused across candidate vessels.
        dist = distance_transform_edt(~seg_skel, sampling=(dely, delx))

        channels = (int(side[k]),) if side is not None else range(pred_arts.shape[1])
        per_channel = []
        for c in channels:
            pred_skel = skeletonize(pred_arts[k, c].cpu().numpy())
            if pred_skel.sum() == 0:
                continue
            per_channel.append(float(dist[pred_skel].mean()))

        if per_channel:
            distances.append(min(per_channel))

    return float(np.mean(distances)) if distances else None
