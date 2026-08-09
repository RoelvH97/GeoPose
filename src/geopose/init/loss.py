"""Publication objective for GeoPose-Init training."""

import torch
import torch.nn as nn
from diffdrr.metrics import (
    DoubleGeodesicSE3,
    LogGeodesicSE3,
    MultiscaleNormalizedCrossCorrelation2d,
)

from ..shared.losses import (
    MultiviewConsistencyLoss,
    SoftCarotidDiceLoss,
    _build,
    projected_distance_mm,
)


class GeoPoseCriterion(nn.Module):
    """The full GeoPose training objective."""

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
        self.mvc = MultiviewConsistencyLoss(self.geo_se3)

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
        teacher_poses=None,
        teacher_map_poses=None,
        student_map_poses=None,
        pred_art=None,
        gt_art=None,
        corrected_img=None,
        corrected_art=None,
        corrected_pose=None,
        proj_pred_pts=None,
        proj_gt_pts=None,
        proj_corr_pts=None,
        proj_height=None,
        proj_delx=None,
        domain_drr=None,
        domain_map=None,
        map_ncc_pred=None,
        map_ncc_target=None,
        r0_pred=None,
        r0_target=None,
        view_logit_drr=None,
        drr_labels=None,
        view_logit_map=None,
        map_labels=None,
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
        if cfg.get("dice_anneal", False):
            lambda_dice *= da_lam
        if lambda_dice > 0.0:
            dice_loss = self.dice(pred_art, gt_art)
            total = total + lambda_dice * dice_loss
            terms["dice"] = dice_loss

        lambda_mvc = float(cfg.get("lambda_mvc", 0.0))
        if lambda_mvc > 0.0:
            mvc_per_pair = self.mvc(poses, pred_poses)
            if mvc_per_pair.numel() > 0:
                mvc = mvc_per_pair.mean()
                total = total + lambda_mvc * mvc
                terms["mvc"] = mvc

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

        if teacher_poses is not None:
            ema_cfg = cfg.get("ema", {})
            w = da_lam if ema_cfg.get("anneal", False) else 1.0
            ema_log = self.log_se3(pred_poses, teacher_poses).mean()
            _, _, ema_double = [x.mean() for x in self.geo_se3(pred_poses, teacher_poses)]
            total = total + w * float(ema_cfg.get("lambda_log", 0.0)) * ema_log
            total = total + w * float(ema_cfg.get("lambda_double", 0.0)) * ema_double
            terms["ema_log"]    = ema_log
            terms["ema_double"] = ema_double

        if teacher_map_poses is not None and student_map_poses is not None:
            dsa_cfg = cfg.get("ema", {}).get("dsa", {})
            w = da_lam if dsa_cfg.get("anneal", False) else 1.0
            ema_dsa_log = self.log_se3(student_map_poses, teacher_map_poses).mean()
            _, _, ema_dsa_double = [x.mean() for x in self.geo_se3(student_map_poses, teacher_map_poses)]
            total = total + w * float(dsa_cfg.get("lambda_log", 0.0)) * ema_dsa_log
            total = total + w * float(dsa_cfg.get("lambda_double", 0.0)) * ema_dsa_double
            terms["ema_dsa_log"]    = ema_dsa_log
            terms["ema_dsa_double"] = ema_dsa_double

        if corrected_img is not None:
            w = da_lam if cfg.refine.get("anneal", False) else 1.0
            refine_ncc = self.ncc(corrected_img, images).mean()
            refine_dice = self.dice(corrected_art, gt_art)
            total = total + w * float(cfg.refine.get("lambda_ncc", 1.0)) * (-refine_ncc)
            total = total + w * float(cfg.refine.get("lambda_dice", 0.1)) * refine_dice
            terms["refine_ncc"] = refine_ncc
            terms["refine_dice"] = refine_dice
            terms["refine_w"] = torch.as_tensor(float(w))

            if proj_corr_pts is not None:
                refine_proj_mpd = projected_distance_mm(proj_corr_pts, proj_gt_pts, proj_height, proj_delx)
                terms["refine_proj_mpd"] = refine_proj_mpd
                lam_rproj = float(cfg.refine.get("lambda_proj", 0.0))
                if lam_rproj > 0.0:
                    total = total + w * lam_rproj * refine_proj_mpd

            lam_glog    = float(cfg.refine.get("lambda_geo_log", 0.0))
            lam_gdouble = float(cfg.refine.get("lambda_geo_double", 0.0))
            if corrected_pose is not None and (lam_glog > 0.0 or lam_gdouble > 0.0):
                refine_geo_log = self.log_se3(corrected_pose, poses).mean()
                rg_rot, rg_xyz, refine_geo_double = [
                    x.mean() for x in self.geo_se3(corrected_pose, poses)
                ]
                total = total + w * lam_glog * refine_geo_log
                total = total + w * lam_gdouble * refine_geo_double
                terms["refine_geo_log"]    = refine_geo_log
                terms["refine_geo_double"] = refine_geo_double
                terms["refine_rgeo"]       = rg_rot
                terms["refine_tgeo"]       = rg_xyz

        if cfg.get("lambda_da", 0.0) != 0.0 and domain_map is not None:
            device = domain_drr.device
            B_drr = domain_drr.shape[0]
            K = domain_map.shape[0]
            m = min(B_drr, K)
            d_drr = domain_drr[torch.randperm(B_drr, device=device)[:m]]
            d_map = domain_map[torch.randperm(K, device=device)[:m]]
            logits = torch.cat([d_drr, d_map], dim=0)
            labels = torch.cat([
                torch.zeros(m, 1, device=device),
                torch.ones(m, 1, device=device),
            ], dim=0)
            da_loss = nn.functional.binary_cross_entropy_with_logits(logits, labels)
            with torch.no_grad():
                da_acc = ((logits.sigmoid() > 0.5) == labels.bool()).float().mean()
            total = total + cfg.get("lambda_da", 0.0) * da_loss
            terms["da_loss"] = da_loss
            terms["da_acc"] = da_acc

        if r0_pred is not None:
            r0_penalty = ((r0_pred - r0_target) ** 2).mean()
            terms["map_r0_penalty"] = r0_penalty
            if r0_penalty is not None:
                total = total + cfg.get("lambda_map_r0", 0.0) * r0_penalty
        if map_ncc_pred is not None:
            map_ncc = self.ncc(map_ncc_pred, map_ncc_target).mean()
            terms["map_ncc"] = map_ncc
            if cfg.get("lambda_map_ncc", 0.0) != 0.0:
                total = total + cfg.get("lambda_map_ncc", 0.0) * (-map_ncc)

        view_total = self._view_cls(
            view_logit_drr, drr_labels, view_logit_map, map_labels, da_lam, terms
        )
        if view_total is not None:
            total = total + view_total

        return total, terms

    def _view_cls(self, view_logit_drr, drr_labels, view_logit_map, map_labels, da_lam, terms):
        cfg = self.cfg
        fallback = float(cfg.get("lambda_view_cls", 0.0))
        w_drr = float(cfg.get("lambda_view_cls_drr", fallback))
        w_map = float(cfg.get("lambda_view_cls_map", fallback))
        if w_drr == 0.0 and w_map == 0.0:
            return None
        if cfg.get("view_cls_drr_anneal", False):
            w_drr *= da_lam

        loss = view_logit_drr.new_zeros(())

        drr_loss = nn.functional.cross_entropy(view_logit_drr, drr_labels)
        with torch.no_grad():
            drr_acc = (view_logit_drr.argmax(dim=1) == drr_labels).float().mean()
        terms["view_cls_loss_drr"] = drr_loss
        terms["view_cls_acc_drr"] = drr_acc
        if w_drr != 0.0:
            loss = loss + w_drr * drr_loss

        if view_logit_map is not None:
            map_loss = nn.functional.cross_entropy(view_logit_map, map_labels)
            with torch.no_grad():
                map_acc = (view_logit_map.argmax(dim=1) == map_labels).float().mean()
            terms["view_cls_loss_map"] = map_loss
            terms["view_cls_acc_map"] = map_acc
            if w_map != 0.0:
                loss = loss + w_map * map_loss

        return loss
