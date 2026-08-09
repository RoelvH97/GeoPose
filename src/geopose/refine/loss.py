"""Publication objective for GeoPose-Refine training."""

import torch
import torch.nn as nn
from diffdrr.metrics import (
    DoubleGeodesicSE3,
    GradientNormalizedCrossCorrelation2d,
    LogGeodesicSE3,
    MultiscaleNormalizedCrossCorrelation2d,
)

from ..shared.losses import SoftCarotidDiceLoss, _build, projected_distance_mm


class RefinePoseCriterion(nn.Module):
    """Objective for the refinement CNN (:class:`RefinePoseModule`)."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        m = cfg.model
        self.log_se3 = _build(m.get("log_se3"), lambda _n: LogGeodesicSE3())
        self.geo_se3 = _build(m.get("geo_se3"), lambda n: DoubleGeodesicSE3(sdd=n.sdd))
        self.ncc = _build(
            m.get("ncc"),
            lambda n: MultiscaleNormalizedCrossCorrelation2d(
                patch_sizes=list(n.patch_sizes), patch_weights=list(n.patch_weights),
            ),
        )
        self.gncc = _build(
            m.get("gncc"),
            lambda n: GradientNormalizedCrossCorrelation2d(
                patch_size=(n.get("patch_size", None) if n is not None else None),
                sigma=float(n.get("sigma", 1.0)) if n is not None else 1.0,
            ),
        )

        self.dice = _build(
            m.get("dice"),
            lambda _n: SoftCarotidDiceLoss(k=float(m.get("dice_k", 5.0))),
        )

    def forward(self, corrected_pose, optimal_pose, drr_corrected=None, drr_optimal=None,
                corrected_art=None, optimal_art=None,
                proj_corr_pts=None, proj_opt_pts=None, proj_height=None, proj_delx=None):
        cfg = self.cfg
        device = corrected_pose.matrix.device

        L_geo_log = self.log_se3(corrected_pose, optimal_pose).mean()
        geo_rot, geo_xyz, L_geo_double = [
            x.mean() for x in self.geo_se3(corrected_pose, optimal_pose)
        ]

        lam_ncc = float(cfg.refine.lambda_ncc)
        lam_gncc = float(cfg.refine.get("lambda_gncc", 0.0))
        if drr_corrected is not None:
            L_ncc = (-self.ncc(drr_corrected, drr_optimal).mean()
                     if lam_ncc > 0.0 else torch.zeros((), device=device))
            L_gncc = (-self.gncc(drr_corrected, drr_optimal).mean()
                      if lam_gncc > 0.0 else torch.zeros((), device=device))
        else:
            L_ncc = torch.zeros((), device=device)
            L_gncc = torch.zeros((), device=device)

        lam_dice = float(cfg.refine.get("lambda_dice", 0.0))
        L_dice = (self.dice(corrected_art, optimal_art)
                  if corrected_art is not None else torch.zeros((), device=device))

        lam_proj = float(cfg.refine.get("lambda_proj", 0.0))
        L_proj = (projected_distance_mm(proj_corr_pts, proj_opt_pts, proj_height, proj_delx)
                  if proj_corr_pts is not None else torch.zeros((), device=device))

        total = (
            float(cfg.refine.lambda_geo_log) * L_geo_log
            + float(cfg.refine.lambda_geo_double) * L_geo_double
            + lam_ncc * L_ncc
            + lam_gncc * L_gncc
            + lam_dice * L_dice
            + lam_proj * L_proj
        )
        return {
            "total": total,
            "geo_log": L_geo_log,
            "geo_double": L_geo_double,
            "geo_rot": geo_rot,
            "geo_xyz": geo_xyz,
            "ncc": L_ncc,
            "gncc": L_gncc,
            "dice": L_dice,
            "proj_mpd": L_proj,
        }
