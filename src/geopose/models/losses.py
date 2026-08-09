"""Dedicated loss classes for the GeoPose pose-estimation model.

The pose model used to carry its loss math as a pile of methods on the
``ResNetPose`` LightningModule (``_soft_dice_loss``, ``_multiview_consistency``,
``_da_loss``, ``_map_ncc_loss``, ``_view_cls_loss``) plus an inline assembly in
``_shared_step``. That logic lives here now as self-contained, testable classes:

* :class:`SoftCarotidDiceLoss`     — occupancy Dice on carotid path-length maps.
* :class:`MultiviewConsistencyLoss`— xvr-style pairwise relative-transform dGeo.
* :class:`GeoPoseCriterion`        — the full objective. The module computes the
  *operands* (network forwards, DRR renders, label slicing) and hands them to
  the criterion, which owns every reduction, weight, and DANN-style anneal and
  returns ``(total, terms)`` — ``terms`` being the per-term scalars to log.

All losses are stateless (no parameters/buffers), so wrapping the model's
sub-metrics in a criterion adds nothing to the ``state_dict`` and existing
checkpoints keep loading ``strict=True``.
"""

# import necessary libraries
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
    """Instantiate a loss from a ``_target_`` config node, else fall back.

    New configs give loss sub-blocks a ``_target_`` (so they compose like the
    optimizer). Checkpoints predating that carry the legacy flat keys, and
    ``load_from_checkpoint`` rebuilds from the *saved* cfg — so when ``node`` has
    no ``_target_`` we construct from the legacy keys via ``default_factory``.
    This keeps the loadability invariant intact across the refactor.
    """
    if node is not None and "_target_" in node:
        return instantiate(node)
    return default_factory(node)


class SoftCarotidDiceLoss(nn.Module):
    """Soft Dice on carotid vessel *occupancy* (GeoReg-style, fixed for thin vessels).

    Both inputs are non-negative carotid path-length projections ([B, 1, H, W])
    rendered from the same CTA volume at the predicted and ground-truth poses.
    The GT is binarised (path-length > 0 → vessel, as in GeoReg's
    ``(dsa_mask > 0)``); the prediction is mapped to a soft occupancy
    ``1 − exp(−k·x) ∈ [0, 1)`` that sends background (0 mm) → 0 and saturates on
    vessels. (GeoReg uses ``sigmoid(x)`` here, but for our thin carotids —
    path-length ≲3 mm — that floors at sigmoid(0)=0.5 and the Dice goes nearly
    flat; the occupancy map fixes that.) Both operands then live in [0, 1], so
    the standard linear Sørensen Dice measures spatial vessel *overlap*
    independent of thickness. Returns ``1 − Dice`` averaged over the batch
    (0 = perfect overlap).

    ``k`` (per-mm rate) sets occupancy saturation: k≈5 maps a 1 mm path-length
    to ~0.99. Validated on real carotid DRRs — aligned Dice loss ≈ 0.016, rising
    monotonically to ~1.0 by ~20 mm translation error.
    """

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
    """xvr-style pairwise relative-transform consistency.

    For each unique pair (i, j) with i < j in the in-patient batch, compute
    ``dgeo(T_j ∘ T_i⁻¹, T̂_j ∘ T̂_i⁻¹)``. Returns the per-pair tensor (empty when
    B < 2); the caller averages. Uses a :class:`DoubleGeodesicSE3` for the
    geodesic, sharing the model's ``geo_se3`` so the metric matches.
    """

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
    """LXPose mean projected distance (mm) over in-frame skeleton fiducials.

    Args:
        pred_pts, gt_pts: ``[B, N, 2]`` detector-pixel projections of the SAME
            fiducials under the predicted and GT poses (from
            ``drr.perspective_projection``).
        height: detector size (px). Points outside ``[0, height]`` in EITHER
            projection are excluded — LXPose masks both (mask1 & mask2).
        delx: detector pixel spacing (mm/px), converting the pixel error to mm.

    A differentiable masked mean (out-of-frame points get zero weight, sum over
    valid / count) — avoids the NaN-poisoning of LXPose's ``nanmean`` while being
    numerically identical. Returns a scalar (mm at the detector plane).
    """
    def _in(p):
        return (p[..., 0] >= 0) & (p[..., 0] <= height) & (p[..., 1] >= 0) & (p[..., 1] <= height)

    valid = (_in(pred_pts) & _in(gt_pts)).float()          # [B, N]
    dist = (pred_pts - gt_pts).norm(dim=-1)                # [B, N]  pixels
    return (dist * valid).sum() / valid.sum().clamp(min=1.0) * float(delx)


class GeoPoseCriterion(nn.Module):
    """The full GeoPose training objective.

    Built from the model config (so weights live in one place), it owns the
    primary photometric + geodesic terms, the carotid Dice, multiview
    consistency, the end-to-end refinement Dice/NCC, the domain-adaptation BCE,
    the MAP NCC + r0 penalty, and the 3-class view classification — including
    every λ weight and the shared DANN ``da_lam`` anneal.

    The module supplies operands (forwards/renders/label slices) as keyword
    args; absent operands (e.g. no MAP batch, refinement disabled) simply skip
    their term, mirroring the original ``_shared_step`` guards exactly.

    Returns ``(total, terms)``. ``terms`` maps each metric name (without the
    ``{stage}/`` prefix) to its scalar tensor; the module logs them with
    ``on_step`` per stage. ``{stage}/loss`` and the val-only ``mpcd`` metric are
    logged by the module itself.
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
        self.mvc = MultiviewConsistencyLoss(self.geo_se3)

    # ── Geodesic helpers exposed for metrics/logging reuse ────────────────────
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

        # ── Primary photometric + geodesic ───────────────────────────────────
        ncc_loss = self.ncc(pred_images, images).mean()
        log_loss = self.log_se3(pred_poses, poses).mean()
        geo_rot, geo_xyz, geo_loss = [x.mean() for x in self.geo_se3(pred_poses, poses)]
        total = -ncc_loss + cfg.lambda1 * log_loss + cfg.lambda2 * geo_loss

        # ── Carotid Dice (pred-pose vs GT-pose occupancy) ─────────────────────
        lambda_dice = float(cfg.get("lambda_dice", 0.0))
        if cfg.get("dice_anneal", False):
            lambda_dice *= da_lam   # DANN-style 0→1 ramp (shared with GRL)
        if lambda_dice > 0.0:
            dice_loss = self.dice(pred_art, gt_art)
            total = total + lambda_dice * dice_loss
            terms["dice"] = dice_loss

        # ── Multiview consistency ─────────────────────────────────────────────
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

        # ── Projection loss (LXPose mPD on net1's pose) ───────────────────────
        # Reprojection error (mm) of the carotid-skeleton fiducials under the pred
        # vs GT pose — LXPose's projection loss, with centreline points as the
        # fiducials. Logged whenever fiducials are present (a val metric even at
        # weight 0); added to the objective when lambda_proj > 0.
        if proj_pred_pts is not None:
            proj_mpd = projected_distance_mm(proj_pred_pts, proj_gt_pts, proj_height, proj_delx)
            terms["proj_mpd"] = proj_mpd
            lam_proj = float(cfg.get("lambda_proj", 0.0))
            if lam_proj > 0.0:
                total = total + lam_proj * proj_mpd

        # ── EMA mean-teacher consistency (student ↔ teacher pose agreement) ────
        # ``teacher_poses`` are decoded from the EMA-of-student network run on the
        # *un-augmented* DRR with no grad (see ResNetPose._shared_step). Pull the
        # student's augmented-DRR pose toward that target with the same log +
        # double geodesics as the primary objective; the teacher is detached so
        # gradients flow only into the student. Present only in training and only
        # when the EMA teacher is enabled.
        if teacher_poses is not None:
            ema_cfg = cfg.get("ema", {})
            w = da_lam if ema_cfg.get("anneal", False) else 1.0
            ema_log = self.log_se3(pred_poses, teacher_poses).mean()
            _, _, ema_double = [x.mean() for x in self.geo_se3(pred_poses, teacher_poses)]
            total = total + w * float(ema_cfg.get("lambda_log", 0.0)) * ema_log
            total = total + w * float(ema_cfg.get("lambda_double", 0.0)) * ema_double
            terms["ema_log"]    = ema_log
            terms["ema_double"] = ema_double

        # ── Semi-supervised EMA mean-teacher on the unlabeled real DSA pool ───
        # No GT pose exists for real DSA, so the teacher's weak-aug read IS the
        # target: pull the student's strong-aug MAP pose toward the teacher's
        # weak-aug MAP pose (teacher detached). Same geodesics, separate weights
        # (real DSA is noisier than synthetic). Operands come from the module's
        # student/teacher forwards on the shared-geometry weak/strong MAP pair.
        if teacher_map_poses is not None and student_map_poses is not None:
            dsa_cfg = cfg.get("ema", {}).get("dsa", {})
            w = da_lam if dsa_cfg.get("anneal", False) else 1.0
            ema_dsa_log = self.log_se3(student_map_poses, teacher_map_poses).mean()
            _, _, ema_dsa_double = [x.mean() for x in self.geo_se3(student_map_poses, teacher_map_poses)]
            total = total + w * float(dsa_cfg.get("lambda_log", 0.0)) * ema_dsa_log
            total = total + w * float(dsa_cfg.get("lambda_double", 0.0)) * ema_dsa_double
            terms["ema_dsa_log"]    = ema_dsa_log
            terms["ema_dsa_double"] = ema_dsa_double

        # ── End-to-end refinement (Dice + NCC + geodesic δ penalty) ───────────
        if corrected_img is not None:
            w = da_lam if cfg.refine.get("anneal", False) else 1.0
            refine_ncc = self.ncc(corrected_img, images).mean()
            refine_dice = self.dice(corrected_art, gt_art)
            total = total + w * float(cfg.refine.get("lambda_ncc", 1.0)) * (-refine_ncc)
            total = total + w * float(cfg.refine.get("lambda_dice", 0.1)) * refine_dice
            terms["refine_ncc"] = refine_ncc
            terms["refine_dice"] = refine_dice
            terms["refine_w"] = torch.as_tensor(float(w))

            # LXPose stage-2 projection loss: reprojection error (mm) of the
            # corrected (net2) pose. No detach ⇒ also backprops into net1, exactly
            # like LXPose's ncc2 + 0.1·mpd2. Logged always; added when weight > 0.
            if proj_corr_pts is not None:
                refine_proj_mpd = projected_distance_mm(proj_corr_pts, proj_gt_pts, proj_height, proj_delx)
                terms["refine_proj_mpd"] = refine_proj_mpd
                lam_rproj = float(cfg.refine.get("lambda_proj", 0.0))
                if lam_rproj > 0.0:
                    total = total + w * lam_rproj * refine_proj_mpd

            # Geodesic penalty on net2's correction: supervise the corrected pose
            # directly against the GT pose — equivalently, pull net2's δ toward the
            # ideal δ* = P_gt⁻¹∘P_pred (geodesic(corrected, GT)=0 ⟺ δ=δ*). Unlike
            # NCC/Dice this is an exact pose target (synthetic-only — needs GT) and
            # mirrors the two terms RefinePoseCriterion uses for the standalone
            # refiner. No detach ⇒ it also sharpens net1.
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

        # ── Domain adaptation (DANN) ──────────────────────────────────────────
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

        # ── MAP r0 penalty + MAP NCC ──────────────────────────────────────────
        # The module renders MAP DRRs (and skips that render in train when NCC is
        # off), so map_ncc_pred is present only when a render happened.
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

        # ── 3-class view classification ───────────────────────────────────────
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


class RefinePoseCriterion(nn.Module):
    """Objective for the refinement CNN (:class:`RefinePoseModule`).

    The corrected pose is supervised against the optimal pose with two geodesic
    terms (always on) plus optional same-modality NCC / Gradient-NCC between
    DRR(corrected) and DRR(optimal), the carotid occupancy Dice, and the LXPose
    projection loss (mPD over the skeleton fiducials). The module renders those
    two DRRs (only when a photometric weight is active) and projects the
    fiducials (only when they are in the batch) and hands them in; this owns the
    reductions, the λ weights, and the total. Returns a dict of the computed scalars so the
    module can keep its per-view / diagnostic logging unchanged.

    Built from the full cfg (``cfg.model.{ncc,log_se3,geo_se3,gncc}`` for the
    metrics, ``cfg.refine.lambda_*`` for the weights). Metrics prefer a
    ``_target_`` block and fall back to the legacy flat keys so refiner
    checkpoints predating the _target_ wiring still load.
    """

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
        # Carotid occupancy Dice — same loss the GeoPose cascade uses. Active as a
        # loss when cfg.refine.lambda_dice > 0; always returned for logging.
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

        # LXPose projection loss (mPD): mm reprojection error of the carotid-skeleton
        # fiducials under the corrected pose vs the optimal pose — the same term (and
        # the same 0.1 weight) the cascade applies to net2 via cfg.refine.lambda_proj.
        # The module supplies the projections only when data.fiducials.enabled; zero
        # otherwise, so runs without fiducials are unchanged.
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
