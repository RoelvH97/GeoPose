"""Shared map-vs-prediction metrics used by both ResNetPose and the xvr baseline."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize


def mpcd(
    pred_arts: torch.Tensor,
    valid_segs: torch.Tensor,
    delx: float,
    dely: float,
    side: torch.Tensor | None = None,
) -> float | None:
    """Mean Projected Centerline Distance (mm), matching GeoReg Eq. 9."""
    distances = []
    for k in range(pred_arts.shape[0]):
        seg_mask = valid_segs[k, 0].cpu().numpy() > 0.5
        seg_skel = skeletonize(seg_mask)
        if seg_skel.sum() == 0:
            continue

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
