"""Lightning wrapper around :class:`RefineResNetPose`."""

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

from .loss import RefinePoseCriterion
from ..shared.pose import delta_to_pose
from .network import build_refine_pose_net, refiner_view_index


class RefinePoseModule(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters({"cfg": cfg})
        self.cfg = cfg
        self.net = build_refine_pose_net(cfg.model)

        self.criterion = RefinePoseCriterion(cfg)

        self._register_load_state_dict_pre_hook(self._remap_legacy_loss_keys)

    @staticmethod
    def _remap_legacy_loss_keys(state_dict, prefix, local_metadata, strict,
                                missing_keys, unexpected_keys, error_msgs):
        """Map pre-extraction loss keys into the criterion namespace."""
        for legacy in ("ncc.", "gncc.", "geo_se3.", "log_se3."):
            src = prefix + legacy
            for k in [k for k in state_dict if k.startswith(src)]:
                state_dict[prefix + "criterion." + k[len(prefix):]] = state_dict.pop(k)

    def _shared_step(self, batch, prefix: str):
        map_img      = batch["map"]
        drr_noisy    = batch["drr_noisy"]
        noisy_pose   = batch["noisy_pose"]
        optimal_pose = batch["optimal_pose"]
        true_dR      = batch["true_dR"]
        true_dt      = batch["true_dt"]
        lateral_mask = batch["lateral_mask"]
        drr          = batch["drr"]
        K = map_img.shape[0]

        view_label = refiner_view_index(
            self.net, batch.get("view_label"), K
        ).to(map_img.device)
        dR_pred, dt_pred = self.net(map_img, drr_noisy, view_label)

        delta_pose_pred = delta_to_pose(dR_pred, dt_pred)
        # Training defines noisy = target ∘ delta, hence the right-composed inverse.
        corrected_pose = noisy_pose.compose(delta_pose_pred.inverse())

        lam_ncc = float(self.cfg.refine.lambda_ncc)
        lam_gncc = float(self.cfg.refine.get("lambda_gncc", 0.0))
        lam_dice = float(self.cfg.refine.get("lambda_dice", 0.0))
        want_photo = lam_ncc > 0.0 or lam_gncc > 0.0
        need_corr_grad = want_photo or lam_dice > 0.0

        drr.to(self.device)
        if need_corr_grad:
            mc_corr = drr(corrected_pose, mask_to_channels=True)
        else:
            with torch.no_grad():
                mc_corr = drr(corrected_pose, mask_to_channels=True)
        with torch.no_grad():
            mc_opt = drr(optimal_pose, mask_to_channels=True)

        proj_corr_pts = proj_opt_pts = proj_height = proj_delx = None
        fid = batch.get("fiducials")
        if fid is not None:
            fid = fid.to(self.device).expand(K, -1, -1)
            proj_corr_pts = drr.perspective_projection(corrected_pose, fid)
            with torch.no_grad():
                proj_opt_pts = drr.perspective_projection(optimal_pose, fid)
            proj_height = int(drr.detector.height)
            proj_delx = float(drr.detector.delx)
        drr.cpu()

        drr_corrected = drr_optimal_img = None
        if want_photo:
            drr_corrected = self._normalize(mc_corr.sum(dim=1, keepdim=True))

            drr_optimal_img = (map_img if self.cfg.refine.get("ncc_vs_input", False)
                               else self._normalize(mc_opt.sum(dim=1, keepdim=True)))

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

        self.log(f"{prefix}/loss", total, prog_bar=(prefix == "train"), batch_size=K)
        self.log(f"{prefix}/geo_log", L_geo_log, batch_size=K)
        self.log(f"{prefix}/geo_double", L_geo_double, batch_size=K)
        self.log(f"{prefix}/geo_double_rot", geo_rot, batch_size=K)
        self.log(f"{prefix}/geo_double_xyz", geo_xyz, batch_size=K)
        self.log(f"{prefix}/ncc", -L_ncc, batch_size=K)
        if lam_gncc > 0.0:
            self.log(f"{prefix}/gncc", -L_gncc, batch_size=K)

        self.log(f"{prefix}/refine_dice", refine_dice, batch_size=K)
        if proj_corr_pts is not None:

            self.log(f"{prefix}/refine_proj_mpd", res["proj_mpd"], batch_size=K)
        self.log(f"{prefix}/refine_ncc", -L_ncc, batch_size=K)
        self.log(f"{prefix}/refine_geo_log", L_geo_log, batch_size=K)
        self.log(f"{prefix}/refine_geo_double", L_geo_double, batch_size=K)
        self.log(f"{prefix}/refine_rgeo", geo_rot, batch_size=K)
        self.log(f"{prefix}/refine_tgeo", geo_xyz, batch_size=K)
        self.log(f"{prefix}/refine_w", torch.ones((), device=total.device), batch_size=K)

        per_axis_R = (dR_pred.detach() - true_dR).abs().mean(dim=0)
        per_axis_t = (dt_pred.detach() - true_dt).abs().mean(dim=0)
        for i in range(3):
            self.log(f"{prefix}/dR_mae_{i}", per_axis_R[i], batch_size=K)
            self.log(f"{prefix}/dt_mae_{i}", per_axis_t[i], batch_size=K)

        self.log(f"{prefix}/dt_mae_depth", per_axis_t[1], prog_bar=(prefix == "train"), batch_size=K)

        sdr = float(self.cfg.model.geo_se3.sdd) / 2.0
        rot_err_deg = (geo_rot.detach() / sdr) * (180.0 / math.pi)
        trans_err_mm = geo_xyz.detach()
        self.log(f"{prefix}/rot_err_deg", rot_err_deg, batch_size=K)
        self.log(f"{prefix}/trans_err_mm", trans_err_mm, batch_size=K)

        gr_pe, gx_pe, _ = self.criterion.geo_se3(corrected_pose, optimal_pose)
        rot_err_deg_pe = (gr_pe.detach() / sdr) * (180.0 / math.pi)
        trans_err_mm_pe = gx_pe.detach()
        ncc_pe = (self.criterion.ncc(drr_corrected, drr_optimal_img).detach()
                  if want_photo else None)
        abs_R_pe = (dR_pred.detach() - true_dR).abs()
        abs_t_pe = (dt_pred.detach() - true_dt).abs()
        lat_m = lateral_mask.bool()
        for vname, vm in (("lat", lat_m), ("pa", ~lat_m)):
            n_v = int(vm.sum())
            if n_v == 0:
                continue
            self.log(f"{prefix}/{vname}/rot_err_deg", rot_err_deg_pe[vm].mean(), batch_size=n_v)
            self.log(f"{prefix}/{vname}/trans_err_mm", trans_err_mm_pe[vm].mean(), batch_size=n_v)
            if ncc_pe is not None:
                self.log(f"{prefix}/{vname}/ncc", ncc_pe[vm].mean(), batch_size=n_v)

            for i in range(3):
                self.log(f"{prefix}/{vname}/dR_mae_{i}", abs_R_pe[vm, i].mean(), batch_size=n_v)
                self.log(f"{prefix}/{vname}/dt_mae_{i}", abs_t_pe[vm, i].mean(), batch_size=n_v)

        return total

    def _want_images(self) -> bool:
        log_every = int(self.cfg.refine.get("log_images_every_n_epochs", 5))
        return self.current_epoch % log_every == 0

    def on_train_epoch_start(self):
        self._train_log_buffer: list[dict] | None = [] if self._want_images() else None

    def training_step(self, batch, batch_idx):
        loss = self._shared_step(batch, "train")

        n_log = int(self.cfg.refine.get("n_log_entries", 4))
        if self._train_log_buffer is not None and batch_idx < n_log:
            self._capture_sample(batch, self._train_log_buffer)
        return loss

    def on_train_epoch_end(self):
        self._log_panel(getattr(self, "_train_log_buffer", None), "train")

    def on_validation_epoch_start(self):
        self._val_log_buffer: list[dict] | None = [] if self._want_images() else None

    def validation_step(self, batch, batch_idx):

        with torch.no_grad():
            map_img = batch["map"]
            drr_noisy_img = batch["drr_noisy"]
            ncc_noisy_dsa = self.criterion.ncc(drr_noisy_img, map_img).mean()
        loss = self._shared_step(batch, "val")
        K = map_img.shape[0]
        self.log("val/ncc_noisy_dsa", ncc_noisy_dsa, batch_size=K)

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

        n_log = int(self.cfg.refine.get("n_log_entries", 4))
        if self._val_log_buffer is not None and batch_idx < n_log:
            self._capture_sample(batch, self._val_log_buffer)

        return loss

    def on_validation_epoch_end(self):
        self._log_panel(getattr(self, "_val_log_buffer", None), "val")

    def _log_panel(self, buffer, stage: str) -> None:
        """Log the buffered pose-stage comparison panel."""
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

    @torch.no_grad()
    def _capture_sample(self, batch, buffer: list) -> None:
        """Render and buffer one corrected and target sample."""
        map_img = batch["map"]
        drr_noisy = batch["drr_noisy"]
        lateral_mask = batch["lateral_mask"]
        drr = batch["drr"]

        view_label = refiner_view_index(
            self.net, batch.get("view_label"), map_img.shape[0]
        ).to(map_img.device)
        dR_pred, dt_pred = self.net(map_img, drr_noisy, view_label)

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
            "view": "lat" if bool(lateral_mask[0]) else "pa",
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
        """Build a pose-stage comparison panel from buffered samples."""
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

    def configure_optimizers(self):
        opt = instantiate(self.cfg.model.optimizer, params=self.parameters())
        sched = instantiate(
            self.cfg.model.scheduler, optimizer=opt, T_max=int(self.cfg.trainer.max_epochs),
        )
        return {"optimizer": opt, "lr_scheduler": sched}

    @staticmethod
    def _normalize(images: torch.Tensor) -> torch.Tensor:
        """Per-image min-max to [0, 1]. Matches CTAPoseDataset._normalize."""
        N, C, H, W = images.shape
        flat = images.reshape(N * C, -1)
        vmin = flat.min(dim=1).values.view(N, C, 1, 1)
        vmax = flat.max(dim=1).values.view(N, C, 1, 1)
        return (images - vmin) / (vmax - vmin).clamp(min=1e-8)
