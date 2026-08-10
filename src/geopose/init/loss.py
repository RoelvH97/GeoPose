"""Publication objective for GeoPose-Init training."""

import torch
import torch.nn as nn
from diffdrr.metrics import (
    DoubleGeodesicSE3,
    LogGeodesicSE3,
    MultiscaleNormalizedCrossCorrelation2d,
)

from ..shared.losses import SoftCarotidDiceLoss, _build, projected_distance_mm


class GeoPoseCriterion(nn.Module):
    """The full GeoPose training objective.

    Total loss is the sum of four terms, with weights from ``configs/init.yaml``:

    ``-mNCC``            multiscale normalized cross-correlation between the
                         rendered and target projection (negated, since DiffDRR
                         reports a similarity)
    ``lambda1 * log_se3``  geodesic SE(3) log-distance to the ground-truth pose
    ``lambda2 * geo_se3``  DiffDRR double-geodesic (rotation + translation) distance
    ``lambda_dice * dice`` soft Dice on rendered carotid occupancy
    ``lambda_proj * mPD``  mean projected distance over carotid skeleton fiducials

    A view-classification cross-entropy on the signed view logits is added with
    weight ``lambda_view_cls_drr``, annealed by the DANN ramp when
    ``view_cls_drr_anneal`` is set.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.ncc = _build(
            cfg.get("ncc"),
            lambda n: MultiscaleNormalizedCrossCorrelation2d(
                patch_sizes=list(n.patch_sizes), patch_weights=list(n.patch_weights),
            ),
        )
        self.log_se3 = _build(cfg.get("log_se3"), lambda _n: LogGeodesicSE3())
        self.geo_se3 = _build(cfg.get("geo_se3"), lambda n: DoubleGeodesicSE3(sdd=n.sdd))
        self.dice = _build(
            cfg.get("dice"),
            lambda _n: SoftCarotidDiceLoss(k=float(cfg.get("dice_k", 5.0))),
        )

    def geodesic_terms(self, pred_poses, poses):
        """(rgeo, tgeo, dgeo) means — also used by the module for logging."""
        return [x.mean() for x in self.geo_se3(pred_poses, poses)]

    def forward(
        self,
        *,
        images,
        pred_images,
        pred_poses,
        poses,
        pred_art=None,
        gt_art=None,
        proj_pred_pts=None,
        proj_gt_pts=None,
        proj_height=None,
        proj_delx=None,
        view_logit_drr=None,
        drr_labels=None,
        da_lam=1.0,
    ):
        cfg = self.cfg
        terms = {}

        ncc_loss = self.ncc(pred_images, images).mean()
        log_loss = self.log_se3(pred_poses, poses).mean()
        geo_rot, geo_xyz, geo_loss = [x.mean() for x in self.geo_se3(pred_poses, poses)]
        # DiffDRR reports NCC similarity, so its negative is minimized.
        total = -ncc_loss + cfg.lambda1 * log_loss + cfg.lambda2 * geo_loss

        lambda_dice = float(cfg.get("lambda_dice", 0.0))
        if lambda_dice > 0.0:
            dice_loss = self.dice(pred_art, gt_art)
            total = total + lambda_dice * dice_loss
            terms["dice"] = dice_loss

        terms["mncc"] = ncc_loss
        terms["log_se3"] = log_loss
        terms["rgeo"] = geo_rot
        terms["tgeo"] = geo_xyz
        terms["dgeo"] = geo_loss

        if proj_pred_pts is not None:
            proj_mpd = projected_distance_mm(proj_pred_pts, proj_gt_pts, proj_height, proj_delx)
            terms["proj_mpd"] = proj_mpd
            lam_proj = float(cfg.get("lambda_proj", 0.0))
            if lam_proj > 0.0:
                total = total + lam_proj * proj_mpd

        view_total = self._view_cls(view_logit_drr, drr_labels, da_lam, terms)
        if view_total is not None:
            total = total + view_total

        return total, terms

    def _view_cls(self, logits, labels, da_lam, terms):
        cfg = self.cfg
        weight = float(cfg.get("lambda_view_cls_drr", 0.0))
        if weight == 0.0:
            return None
        if cfg.get("view_cls_drr_anneal", False):
            weight *= da_lam

        loss = nn.functional.cross_entropy(logits, labels)
        with torch.no_grad():
            accuracy = (logits.argmax(dim=1) == labels).float().mean()
        terms["view_cls_loss_drr"] = loss
        terms["view_cls_acc_drr"] = accuracy
        return weight * loss
