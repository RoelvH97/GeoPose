"""Loss primitives shared by GeoPose-Init and GeoPose-Refine."""

import math
import torch
import torch.nn as nn

from diffdrr.metrics import (
    DoubleGeodesicSE3,
    GradientNormalizedCrossCorrelation2d,
    LogGeodesicSE3,
    MultiscaleNormalizedCrossCorrelation2d,
)
from diffdrr.pose import RigidTransform
from hydra.utils import instantiate


def _build(node, default_factory):
    """Instantiate a loss from a ``_target_`` config node, else fall back."""
    if node is not None and "_target_" in node:
        return instantiate(node)
    return default_factory(node)


class SoftCarotidDiceLoss(nn.Module):
    """Soft Dice on carotid vessel *occupancy* (GeoReg-style, fixed for thin vessels)."""

    def __init__(self, k: float = 5.0, eps: float = 1e-5):
        super().__init__()
        self.k = float(k)
        self.eps = float(eps)

    def forward(self, pred_pathlen: torch.Tensor, target_pathlen: torch.Tensor) -> torch.Tensor:
        p = (1.0 - torch.exp(-self.k * pred_pathlen)).flatten(1)
        t = (target_pathlen > 0).float().flatten(1)
        num = 2.0 * (p * t).sum(dim=1)
        den = p.sum(dim=1) + t.sum(dim=1)
        dice = (num + self.eps) / (den + self.eps)
        return (1.0 - dice).mean()


class MultiviewConsistencyLoss(nn.Module):
    """xvr-style pairwise relative-transform consistency."""

    def __init__(self, geo_se3: nn.Module):
        super().__init__()
        self.geo_se3 = geo_se3

    def forward(self, true_pose, pred_pose) -> torch.Tensor:
        true_mat = true_pose.matrix
        pred_mat = pred_pose.matrix
        B = true_mat.shape[0]
        if B < 2:
            return true_mat.new_zeros((0,))
        true_inv = true_pose.inverse().matrix
        pred_inv = pred_pose.inverse().matrix
        idx, jdx = torch.triu_indices(B, B, offset=1, device=true_mat.device)
        true_rel = RigidTransform(true_mat[jdx] @ true_inv[idx])
        pred_rel = RigidTransform(pred_mat[jdx] @ pred_inv[idx])
        _, _, dgeo_rel = self.geo_se3(true_rel, pred_rel)
        return dgeo_rel


def projected_distance_mm(
    pred_pts: torch.Tensor,
    gt_pts: torch.Tensor,
    height: int,
    delx: float,
) -> torch.Tensor:
    """LXPose mean projected distance (mm) over in-frame skeleton fiducials."""
    def _in(p):
        return (p[..., 0] >= 0) & (p[..., 0] <= height) & (p[..., 1] >= 0) & (p[..., 1] <= height)

    valid = (_in(pred_pts) & _in(gt_pts)).float()
    dist = (pred_pts - gt_pts).norm(dim=-1)
    return (dist * valid).sum() / valid.sum().clamp(min=1.0) * float(delx)
