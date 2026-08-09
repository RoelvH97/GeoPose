"""Lightning wrapper around :class:`RefineResNetPose`.

Training objective per step (K poses for one manifest entry):

1. Forward: ``(dR_hat, dt_hat) = net(MAP, DRR_noisy, view_label)``, where
   ``view_label`` is signed ``LAT-/PA/LAT+ = 0/1/2``.
2. Materialise the predicted delta as an SE(3) transform via
   :func:`delta_to_pose` (axis-angle rad + translation mm), and apply it on
   the manifold via inverse right-composition::

       corrected_pose = noisy_pose ∘ delta_pose_pred^{-1}

   This is the **matched** inverse of the dataloader's noise generator
   (``noisy_pose = optimal_pose ∘ delta_pose``), so for ``δ_pred = δ`` the
   corrected pose equals the optimal pose exactly to float tolerance — see
   :func:`geopose.models.pose_utils._roundtrip_check`. The dataloader provides
   ``noisy_pose`` and ``optimal_pose`` pre-materialised so this step is just
   one ``compose`` + one ``inverse`` on existing ``RigidTransform`` objects.
3. Compose up to four losses (all between corrected_pose and optimal_pose):

   * ``L_geo_log    = LogGeodesicSE3(corrected_pose, optimal_pose)`` — SE(3)
     log-map magnitude (combines R and t into one scalar).
   * ``L_geo_double = DoubleGeodesicSE3(corrected_pose, optimal_pose)`` —
     rotation geodesic + SDD-scaled translation distance, decoupled.
     More stable per-component gradient for small perturbations.
   * ``L_ncc = -MultiscaleNCC(DRR(corrected_pose), DRR(optimal_pose))`` —
     image-space alignment, same-modality (both DRRs rendered from the same
     CTA at slightly different poses). NCC ceilings at 1.0; the cleaner
     pose-driven gradient flows through DiffDRR's ``grid_sample`` via the
     pose tensors. Mirrors the primary ``ncc_loss`` in
     :mod:`geopose.models.resnet_pose`. The cross-modality DRR-vs-DSA NCC is logged
     separately as ``val/ncc_dsa`` for visibility into inference-time
     alignment.
   * ``L_gncc = -GradientNCC(DRR(corrected_pose), DRR(optimal_pose))`` —
     Sobel-filtered NCC (Gradient Correlation, Penney et al. 1998). Same
     operands as ``L_ncc`` so we can match exactly at convergence; gradient
     maps emphasise edge geometry and sharpen the loss landscape near the
     optimum where intensity NCC plateaus. Off by default
     (``lambda_gncc = 0``). When both NCC terms are active the two renders
     are computed once and shared.

   * ``L_proj = mPD(corrected_pose, optimal_pose)`` — LXPose's projection loss:
     the mm reprojection error of the carotid-skeleton fiducials at the detector
     (:func:`geopose.models.losses.projected_distance_mm`). Needs
     ``data.fiducials.enabled`` (the dataset then emits ``batch["fiducials"]``);
     off by default (``lambda_proj = 0``). The distance-based alternative to the
     carotid Dice — see docs/baselines.html#followup.

4. ``total = λ_geo_log · L_geo_log + λ_geo_double · L_geo_double``
   ``       + λ_ncc · L_ncc + λ_gncc · L_gncc + λ_dice · L_dice + λ_proj · L_proj``.

Each batch is one manifest entry's K-amortised poses (DataLoader batch_size=1
+ passthrough collate, see :mod:`geopose.data.refine_dataloader`). The DRR object
arrives in the batch so we can re-render at the corrected pose without a
second CTA load.
"""

from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pytorch_lightning as pl
import torch
from diffdrr.pose import RigidTransform
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger

from .losses import RefinePoseCriterion
from .pose_utils import delta_to_pose
from .refine_fusion_pose import build_refine_pose_net, refiner_view_index


class RefinePoseModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters({"cfg": cfg})
        self.cfg = cfg
        self.net = build_refine_pose_net(cfg.model)

        # Geodesic (log-SE(3) + double) + optional NCC / Gradient-NCC, with the
        # λ weights and the total assembly — all in the criterion. The module
        # below only renders DRR(corrected)/DRR(optimal) and logs.
        self.criterion = RefinePoseCriterion(cfg)

        # GradientNCC carries a fixed Sobel-kernel buffer, so relocating it under
        # `criterion` renamed `gncc.sobel.*` → `criterion.gncc.sobel.*`. Remap the
        # legacy top-level loss-metric keys on load so checkpoints saved before the
        # loss extraction still load strict=True (weights identical). No-op on new
        # checkpoints, which already store the criterion.* keys.
        self._register_load_state_dict_pre_hook(self._remap_legacy_loss_keys)

    @staticmethod
    def _remap_legacy_loss_keys(state_dict, prefix, local_metadata, strict,
                                missing_keys, unexpected_keys, error_msgs):
        for legacy in ("ncc.", "gncc.", "geo_se3.", "log_se3."):
            src = prefix + legacy
            for k in [k for k in state_dict if k.startswith(src)]:
                state_dict[prefix + "criterion." + k[len(prefix):]] = state_dict.pop(k)

    # ── Loss assembly ────────────────────────────────────────────────────────

    def _shared_step(self, batch, prefix: str):
        map_img      = batch["map"]            # [K, 1, H, W]
        drr_noisy    = batch["drr_noisy"]      # [K, 1, H, W]
        noisy_pose   = batch["noisy_pose"]     # RigidTransform [K, 4, 4]
        optimal_pose = batch["optimal_pose"]   # RigidTransform [K, 4, 4]
        true_dR      = batch["true_dR"]        # [K, 3] axis-angle (rad)
        true_dt      = batch["true_dt"]        # [K, 3] translation (mm)
        lateral_mask = batch["lateral_mask"]   # [K] bool
        drr          = batch["drr"]            # patient's DRR (CPU)
        K = map_img.shape[0]

        # 1. Network forward
        view_label = refiner_view_index(
            self.net, batch.get("view_label"), K
        ).to(map_img.device)
        dR_pred, dt_pred = self.net(map_img, drr_noisy, view_label)

        # 2. SE(3) correction — matched inverse of the dataloader's
        # ``noisy_pose = optimal_pose ∘ delta_pose`` construction. With
        # δ_pred = δ, corrected_pose equals optimal_pose to float tolerance.
        delta_pose_pred = delta_to_pose(dR_pred, dt_pred)
        corrected_pose = noisy_pose.compose(delta_pose_pred.inverse())

        # 3. Render DRR(corrected) / DRR(optimal) with mask_to_channels so one
        # render yields both the full projection (summed → NCC operands) and the
        # carotid path-length (channels 1:3 → Dice). The corrected render carries
        # grad when a photometric OR the Dice weight is active.
        lam_ncc = float(self.cfg.refine.lambda_ncc)
        lam_gncc = float(self.cfg.refine.get("lambda_gncc", 0.0))
        lam_dice = float(self.cfg.refine.get("lambda_dice", 0.0))
        want_photo = lam_ncc > 0.0 or lam_gncc > 0.0
        need_corr_grad = want_photo or lam_dice > 0.0

        drr.to(self.device)
        if need_corr_grad:
            mc_corr = drr(corrected_pose, mask_to_channels=True)   # [K, C, H, W] (grad)
        else:
            with torch.no_grad():
                mc_corr = drr(corrected_pose, mask_to_channels=True)
        with torch.no_grad():
            mc_opt = drr(optimal_pose, mask_to_channels=True)

        # ── Skeleton-fiducial projections for the LXPose mPD term ─────────────
        # Project the carotid-skeleton fiducials (batch["fiducials"], already in
        # the DRR's centered frame) under the corrected and optimal poses →
        # detector pixels. Pure geometry, no render: differentiable through
        # corrected_pose regardless of whether the renders carry grad. Absent when
        # data.fiducials.enabled is false ⇒ the criterion skips the term. Mirrors
        # the cascade's proj_corr_pts block in resnet_pose._shared_step.
        proj_corr_pts = proj_opt_pts = proj_height = proj_delx = None
        fid = batch.get("fiducials")
        if fid is not None:
            fid = fid.to(self.device).expand(K, -1, -1)          # [K, N, 3]
            proj_corr_pts = drr.perspective_projection(corrected_pose, fid)
            with torch.no_grad():
                proj_opt_pts = drr.perspective_projection(optimal_pose, fid)
            proj_height = int(drr.detector.height)
            proj_delx = float(drr.detector.delx)
        drr.cpu()

        drr_corrected = drr_optimal_img = None
        if want_photo:
            drr_corrected = self._normalize(mc_corr.sum(dim=1, keepdim=True))
            # NCC target: the corrupted channel-1 input (matches the GeoPose
            # cascade's refine NCC — corrected-vs-input, deployment-faithful) when
            # ncc_vs_input, else the clean GT render (default). map_img is already
            # normalised in the dataloader.
            drr_optimal_img = (map_img if self.cfg.refine.get("ncc_vs_input", False)
                               else self._normalize(mc_opt.sum(dim=1, keepdim=True)))

        # Artery path-length (channels 1:art_end) → Dice in the criterion. art_end
        # rides in the batch (carotid 1:3 for CTA, all vessels for topbrain/topcow;
        # default 3 ⇒ CTA bit-identical). Corrected carries grad (when active);
        # optimal is the detached target.
        art_end = batch.get("art_end", 3)
        corrected_art = mc_corr[:, 1:art_end].sum(dim=1, keepdim=True)
        optimal_art   = mc_opt[:, 1:art_end].sum(dim=1, keepdim=True)

        res          = self.criterion(corrected_pose, optimal_pose, drr_corrected,
                                      drr_optimal_img, corrected_art, optimal_art,
                                      proj_corr_pts, proj_opt_pts, proj_height, proj_delx)
        total        = res["total"]
        L_geo_log    = res["geo_log"]
        L_geo_double = res["geo_double"]
        geo_rot      = res["geo_rot"]
        geo_xyz      = res["geo_xyz"]
        L_ncc        = res["ncc"]
        L_gncc       = res["gncc"]
        refine_dice  = res["dice"]

        # ── Logging ──────────────────────────────────────────────────────
        self.log(f"{prefix}/loss", total, prog_bar=(prefix == "train"), batch_size=K)
        self.log(f"{prefix}/geo_log", L_geo_log, batch_size=K)
        self.log(f"{prefix}/geo_double", L_geo_double, batch_size=K)
        self.log(f"{prefix}/geo_double_rot", geo_rot, batch_size=K)  # SDD-scaled rotation geodesic
        self.log(f"{prefix}/geo_double_xyz", geo_xyz, batch_size=K)  # translation distance
        self.log(f"{prefix}/ncc", -L_ncc, batch_size=K)  # positive NCC = aligned
        if lam_gncc > 0.0:
            self.log(f"{prefix}/gncc", -L_gncc, batch_size=K)  # positive GradientNCC = edge-aligned

        # ── GeoPose-cascade-named aliases — for direct comparison with the e2e run.
        # Same names/definitions as GeoPoseCriterion's refine terms (refine_dice,
        # refine_ncc, refine_geo_log/double, refine_rgeo/tgeo), so the standalone
        # refiner's curves overlay the cascade's. refine_w=1 (no anneal here).
        self.log(f"{prefix}/refine_dice", refine_dice, batch_size=K)
        if proj_corr_pts is not None:
            # mm reprojection error of the skeleton fiducials at the corrected
            # pose — same name/definition as GeoPoseCriterion's refine term.
            self.log(f"{prefix}/refine_proj_mpd", res["proj_mpd"], batch_size=K)
        self.log(f"{prefix}/refine_ncc", -L_ncc, batch_size=K)
        self.log(f"{prefix}/refine_geo_log", L_geo_log, batch_size=K)
        self.log(f"{prefix}/refine_geo_double", L_geo_double, batch_size=K)
        self.log(f"{prefix}/refine_rgeo", geo_rot, batch_size=K)
        self.log(f"{prefix}/refine_tgeo", geo_xyz, batch_size=K)
        self.log(f"{prefix}/refine_w", torch.ones((), device=total.device), batch_size=K)

        # Per-axis MAE on δ (averaged over K) — diagnostic
        per_axis_R = (dR_pred.detach() - true_dR).abs().mean(dim=0)  # [3]
        per_axis_t = (dt_pred.detach() - true_dt).abs().mean(dim=0)  # [3]
        for i in range(3):
            self.log(f"{prefix}/dR_mae_{i}", per_axis_R[i], batch_size=K)
            self.log(f"{prefix}/dt_mae_{i}", per_axis_t[i], batch_size=K)
        # Alias for the along-beam depth axis (== dt_mae_1) under the bi-plane's
        # headline name, so the two refiners overlay on one wandb panel.
        self.log(f"{prefix}/dt_mae_depth", per_axis_t[1], prog_bar=(prefix == "train"), batch_size=K)

        # Interpretable pose errors after correction:
        # DoubleGeodesicSE3's rot_geo returns SDR * ‖so3_log_map(R1ᵀR2)‖ — divide
        # by SDR and convert to degrees for direct readability.
        sdr = float(self.cfg.model.geo_se3.sdd) / 2.0
        rot_err_deg = (geo_rot.detach() / sdr) * (180.0 / math.pi)
        trans_err_mm = geo_xyz.detach()
        self.log(f"{prefix}/rot_err_deg", rot_err_deg, batch_size=K)
        self.log(f"{prefix}/trans_err_mm", trans_err_mm, batch_size=K)

        # Per-view breakdown — split by lateral_mask on per-element metrics. The
        # synthetic batch mixes PA/LAT within one step (GeoPose-style batching);
        # the single-view manifest batch populates only one group per step.
        # Lightning averages across the epoch either way.
        gr_pe, gx_pe, _ = self.criterion.geo_se3(corrected_pose, optimal_pose)  # per-element [K]
        rot_err_deg_pe = (gr_pe.detach() / sdr) * (180.0 / math.pi)
        trans_err_mm_pe = gx_pe.detach()
        ncc_pe = (self.criterion.ncc(drr_corrected, drr_optimal_img).detach()
                  if want_photo else None)
        abs_R_pe = (dR_pred.detach() - true_dR).abs()   # [K, 3] per-axis δ residual
        abs_t_pe = (dt_pred.detach() - true_dt).abs()   # [K, 3]
        lat_m = lateral_mask.bool()
        for vname, vm in (("lat", lat_m), ("pa", ~lat_m)):
            n_v = int(vm.sum())
            if n_v == 0:
                continue
            self.log(f"{prefix}/{vname}/rot_err_deg", rot_err_deg_pe[vm].mean(), batch_size=n_v)
            self.log(f"{prefix}/{vname}/trans_err_mm", trans_err_mm_pe[vm].mean(), batch_size=n_v)
            if ncc_pe is not None:
                self.log(f"{prefix}/{vname}/ncc", ncc_pe[vm].mean(), batch_size=n_v)
            # Per-view per-axis δ MAE — 1-to-1 with the bi-plane module's
            # {v}/dt_mae_i (esp. dt_mae_1 = along-beam depth), so synth and
            # bi-plane PA/LAT depth recovery are directly comparable.
            for i in range(3):
                self.log(f"{prefix}/{vname}/dR_mae_{i}", abs_R_pe[vm, i].mean(), batch_size=n_v)
                self.log(f"{prefix}/{vname}/dt_mae_{i}", abs_t_pe[vm, i].mean(), batch_size=n_v)

        return total

    # ── Lightning hooks ──────────────────────────────────────────────────────

    def _want_images(self) -> bool:
        log_every = int(self.cfg.refine.get("log_images_every_n_epochs", 5))
        return self.current_epoch % log_every == 0

    def on_train_epoch_start(self):
        self._train_log_buffer: list[dict] | None = [] if self._want_images() else None

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "train")
        # Mirror ResNetPose: also log a train sample panel (first N entries) on the
        # logged epochs, not just val.
        n_log = int(self.cfg.refine.get("n_log_entries", 4))
        if self._train_log_buffer is not None and batch_idx < n_log:
            self._capture_sample(batch, self._train_log_buffer)
        return loss

    def on_train_epoch_end(self):
        self._log_panel(getattr(self, "_train_log_buffer", None), "train")

    def on_validation_epoch_start(self):
        self._val_log_buffer: list[dict] | None = [] if self._want_images() else None

    def validation_step(self, batch, batch_idx):
        # Inference-time alignment metric: NCC of the rendered DRR against the
        # real DSA MAP (cross-modality, ceiling ~0.6). The training loss uses
        # DRR-vs-DRR NCC; this one is logged separately for visibility.
        with torch.no_grad():
            map_img = batch["map"]
            drr_noisy_img = batch["drr_noisy"]
            ncc_noisy_dsa = self.criterion.ncc(drr_noisy_img, map_img).mean()
        loss = self._shared_step(batch, "val")
        K = map_img.shape[0]
        self.log("val/ncc_noisy_dsa", ncc_noisy_dsa, batch_size=K)

        # Same metric at the corrected pose — fresh forward pass + render.
        with torch.no_grad():
            view_label = refiner_view_index(
                self.net, batch.get("view_label"), map_img.shape[0]
            ).to(map_img.device)
            dR_p, dt_p = self.net(map_img, drr_noisy_img, view_label)
            delta_p = delta_to_pose(dR_p, dt_p)
            corrected_pose = batch["noisy_pose"].compose(delta_p.inverse())
            drr = batch["drr"]
            drr.to(self.device)
            drr_corr = drr(corrected_pose)
            if drr_corr.shape[1] > 1:
                drr_corr = drr_corr.sum(dim=1, keepdim=True)
            drr_corr = self._normalize(drr_corr)
            drr.cpu()
            ncc_corr_dsa = self.criterion.ncc(drr_corr, map_img).mean()
        self.log("val/ncc_dsa", ncc_corr_dsa, batch_size=K)
        self.log("val/delta_ncc_dsa", ncc_corr_dsa - ncc_noisy_dsa, batch_size=K)

        # Capture data for image logging — first N val entries this epoch.
        n_log = int(self.cfg.refine.get("n_log_entries", 4))
        if self._val_log_buffer is not None and batch_idx < n_log:
            self._capture_sample(batch, self._val_log_buffer)

        return loss

    def on_validation_epoch_end(self):
        self._log_panel(getattr(self, "_val_log_buffer", None), "val")

    def _log_panel(self, buffer, stage: str) -> None:
        """Log the sample panel (rows = pose stages, cols = samples) to wandb —
        analogous to ResNetPose._log_drr_panel. Called for both train and val."""
        if not buffer:
            return
        wandb_logger = next((l for l in self.loggers if isinstance(l, WandbLogger)), None)
        if wandb_logger is None:
            return
        try:
            import wandb
        except ImportError:
            return
        fig = self._build_panel(buffer, stage)
        wandb_logger.log_image(
            key=f"{stage}/samples", images=[wandb.Image(fig)],
            caption=[f"epoch {self.current_epoch}"],
        )
        plt.close(fig)
        if stage == "train":
            self._train_log_buffer = None
        else:
            self._val_log_buffer = None

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    # ── Image logging helpers ────────────────────────────────────────────────

    @torch.no_grad()
    def _capture_sample(self, batch, buffer: list) -> None:
        """Render DRR(corrected) and DRR(optimal) for the first K=0 element and
        append everything the panel needs to ``buffer``."""
        map_img = batch["map"]
        drr_noisy = batch["drr_noisy"]
        lateral_mask = batch["lateral_mask"]
        drr = batch["drr"]

        view_label = refiner_view_index(
            self.net, batch.get("view_label"), map_img.shape[0]
        ).to(map_img.device)
        dR_pred, dt_pred = self.net(map_img, drr_noisy, view_label)
        # Pick the first batch element (index 0) — one sample to draw per column.
        # Slice the pre-materialised SE(3) matrices and re-wrap as RigidTransform.
        noisy_pose_k0 = RigidTransform(batch["noisy_pose"].matrix[:1])
        optimal_pose = RigidTransform(batch["optimal_pose"].matrix[:1])
        delta_pose_pred = delta_to_pose(dR_pred[:1], dt_pred[:1])

        drr.to(self.device)
        corrected_pose = noisy_pose_k0.compose(delta_pose_pred.inverse())
        img_corr = drr(corrected_pose)
        img_opt = drr(optimal_pose)
        drr.cpu()
        if img_corr.shape[1] > 1:
            img_corr = img_corr.sum(dim=1, keepdim=True)
            img_opt = img_opt.sum(dim=1, keepdim=True)
        img_corr = self._normalize(img_corr)
        img_opt = self._normalize(img_opt)

        buffer.append({
            "patient_id": batch["patient_id"],
            "view": "lat" if bool(lateral_mask[0]) else "pa",  # k=0 element's own view
            "timestamp": batch["timestamp"],
            "channel": batch["channel"],
            "map":       map_img[0, 0].detach().cpu().numpy(),
            "drr_noisy": drr_noisy[0, 0].detach().cpu().numpy(),
            "drr_corr":  img_corr[0, 0].detach().cpu().numpy(),
            "drr_opt":   img_opt[0, 0].detach().cpu().numpy(),
            "true_dR":   batch["true_dR"][0].detach().cpu().numpy(),
            "true_dt":   batch["true_dt"][0].detach().cpu().numpy(),
            "pred_dR":   dR_pred[0].detach().cpu().numpy(),
            "pred_dt":   dt_pred[0].detach().cpu().numpy(),
        })

    @staticmethod
    def _row_label(ax, text: str) -> None:
        """Bold rotated row label at the left edge (matches ResNetPose's panels)."""
        ax.text(-0.14, 0.5, text, transform=ax.transAxes, rotation=90,
                ha="center", va="center", fontsize=11, fontweight="bold",
                color="#1f2328", clip_on=False)

    def _build_panel(self, buffer, stage: str):
        """4 rows (pose stages) × N columns (samples), analogous to
        ResNetPose._log_drr_panel — transposed from the old layout.

        Rows: Input (ch1) | Noisy δ (ch2) | Corrected (pred δ) | Optimal (GT).
        If the network is doing its job, ``pred ≈ true`` so the Corrected row
        matches the Optimal row. Used for both train and val (batch element 0).
        """
        row_labels = ["Input", "Noisy (δ)", "Corrected", "Optimal (GT)"]
        n_rows, N = len(row_labels), len(buffer)
        fig, axes = plt.subplots(n_rows, N, figsize=(3.4 * N, 3.4 * n_rows), squeeze=False)
        fig.subplots_adjust(left=0.10, bottom=0.05, top=0.92, hspace=0.4)
        fig.suptitle(f"{stage} samples · epoch {self.current_epoch} · sample 0", fontsize=11)

        for r, lbl in enumerate(row_labels):
            self._row_label(axes[r, 0], lbl)

        for c, item in enumerate(buffer):
            tdR, tdt = item["true_dR"], item["true_dt"]
            pdR, pdt = item["pred_dR"], item["pred_dt"]
            imgs = [item["map"], item["drr_noisy"], item["drr_corr"], item["drr_opt"]]
            caps = [
                f"{item['patient_id']}  {item['view']}",
                f"true δR=[{tdR[0]:+.2f},{tdR[1]:+.2f},{tdR[2]:+.2f}]\n"
                f"true δt=[{tdt[0]:+.0f},{tdt[1]:+.0f},{tdt[2]:+.0f}] mm",
                f"pred δR=[{pdR[0]:+.2f},{pdR[1]:+.2f},{pdR[2]:+.2f}]\n"
                f"pred δt=[{pdt[0]:+.0f},{pdt[1]:+.0f},{pdt[2]:+.0f}] mm",
                "δ = 0 (target)",
            ]
            for r in range(n_rows):
                ax = axes[r, c]
                ax.imshow(imgs[r], cmap="gray")
                if r == 0:
                    ax.set_title(f"sample {c}", fontsize=9, fontweight="bold", pad=4)
                ax.text(0.5, -0.04, caps[r], transform=ax.transAxes, ha="center",
                        va="top", fontsize=7, family="monospace")
                ax.set_xticks([]); ax.set_yticks([])
        return fig

    # ── Optimiser ────────────────────────────────────────────────────────────

    def configure_optimizers(self):
        opt = instantiate(self.cfg.model.optimizer, params=self.parameters())
        sched = instantiate(
            self.cfg.model.scheduler, optimizer=opt, T_max=int(self.cfg.trainer.max_epochs),
        )
        return {"optimizer": opt, "lr_scheduler": sched}

    # ── Helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize(images: torch.Tensor) -> torch.Tensor:
        """Per-image min-max to [0, 1]. Matches CTAPoseDataset._normalize."""
        N, C, H, W = images.shape
        flat = images.reshape(N * C, -1)
        vmin = flat.min(dim=1).values.view(N, C, 1, 1)
        vmax = flat.max(dim=1).values.view(N, C, 1, 1)
        return (images - vmin) / (vmax - vmin).clamp(min=1e-8)
