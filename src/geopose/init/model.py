"""GeoPose-Init model, training loop, and pose decoder."""

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb

from diffdrr.pose import convert
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger

from ..shared.blocks import build_resnet_backbone, _PoseDomainHead
from .loss import GeoPoseCriterion
from ..shared.visualization import euler_zyx_from_matrix
from ..shared.visualization import to_np as _to_np


class ResNetPose(pl.LightningModule):
    """Single-backbone GeoPose-Init: one DRR or MAP view in, one 6-DOF pose out.

    The network is a torchvision ResNet whose ``fc`` is replaced by
    :class:`~geopose.shared.blocks._PoseDomainHead`, which emits three groups of
    outputs concatenated along dim 1:

    ``[0:num_outputs]``   pose parameters, decoded by :meth:`_decode_pose`
    ``[num_outputs]``     a domain logit behind a gradient-reversal layer
    ``[num_outputs+1:+4]``  signed view logits over {LAT-, PA, LAT+}

    Poses are Euler-ZYX rotations in radians plus a translation in millimetres,
    expressed in the DiffDRR camera convention. The pose branch is conditioned on
    a signed view role through a learned embedding, so the same backbone serves
    both projections; see :meth:`_decode_pose` for the output normalization.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        net, in_features = build_resnet_backbone(cfg.backbone, 1, cfg.pretrained)
        self.view_role_emb_dim = int(cfg.get("view_role_emb_dim", 0))
        net.fc = _PoseDomainHead(
            in_features,
            cfg.num_outputs,
            cfg.get("da_alpha", 1.0),
            role_emb_dim=self.view_role_emb_dim,
            role_classes=3,
        )
        self.net = net

        self.criterion = GeoPoseCriterion(cfg)
        self._da_lam = 0.0

    def forward(
        self,
        x: torch.Tensor,
        view_label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pose params [B, num_outputs], optionally conditioned on known view role."""
        return self._net_forward(self.net, x, view_label)[:, :self.cfg.num_outputs]

    @staticmethod
    def _backbone_features(net: nn.Module, images: torch.Tensor) -> torch.Tensor:
        """Return pooled ResNet features before the prediction head."""
        features = net.conv1(images)
        features = net.bn1(features)
        features = net.relu(features)
        features = net.maxpool(features)
        features = net.layer1(features)
        features = net.layer2(features)
        features = net.layer3(features)
        features = net.layer4(features)
        features = net.avgpool(features)
        return torch.flatten(features, 1)

    def _net_forward(
        self,
        net: nn.Module,
        images: torch.Tensor,
        view_label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward a GeoPose ResNet, injecting a known role when enabled."""
        if self.view_role_emb_dim == 0:
            return net(images)
        if view_label is None:
            raise ValueError(
                "view_label is required when model.view_role_emb_dim is non-zero"
            )
        features = self._backbone_features(net, images)
        return net.fc(features, view_label.long())

    def _net_forward_predicted_role(
        self,
        net: nn.Module,
        images: torch.Tensor,
        metadata_label: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Two-phase forward for the real-DSA eval path.

        On real DSA the acquisition metadata gives the view *role* (lateral vs PA)
        reliably, but not which lateral side the C-arm was on. Phase one reads the
        view logits, which resolve the side; phase two feeds the resulting signed
        label to the pose branch. Returns the full head output and the signed label
        that :meth:`_decode_pose` must be given.
        """
        n = self.cfg.num_outputs
        if self.view_role_emb_dim == 0:
            out = net(images)
            decode_view_label = self._dsa_decode_view_labels(
                out[:, n + 1:n + 4], metadata_label
            )
            return out, decode_view_label

        features = self._backbone_features(net, images)
        domain_logit, view_logits = net.fc.forward_aux(features)
        decode_view_label = self._dsa_decode_view_labels(view_logits, metadata_label)
        pose_params = net.fc.forward_pose(features, decode_view_label)
        return torch.cat([pose_params, domain_logit, view_logits], dim=1), decode_view_label

    # Signed view roles: 0 = LAT-, 1 = PA, 2 = LAT+. The rotation anchor adds
    # -pi/2, 0, +pi/2 to the first Euler angle respectively.
    _VIEW_SIGN = (-1.0, 0.0, 1.0)

    def _decode_pose(self, raw: torch.Tensor, view_label: torch.Tensor | None = None):
        """Split raw network output into (R [B,3] radians, t [B,3] millimetres).

        The network regresses residuals about a per-view anchor rather than an
        absolute pose. Rotation: the first Euler-ZYX angle is offset by the signed
        view anchor above, so the network only has to explain the deviation from a
        canonical LAT/PA orientation. Translation: outputs are scaled by
        ``_T_SCALE_MM`` and shifted by ``_T_OFFSET_MM``, which places the origin at
        the isocentre used throughout training (``data.pose.t_y_offset``). Both
        constants are part of the frozen contract: changing either invalidates the
        published checkpoints.
        """
        R = raw[:, :3]
        if view_label is not None and self.cfg.get("view_anchor", True):
            sign_lookup = torch.tensor(self._VIEW_SIGN, device=R.device, dtype=R.dtype)
            sign = sign_lookup[view_label.long()]
            offset = torch.zeros_like(R)
            offset[:, 0] = sign * (math.pi / 2)
            R = R + offset
        t = raw[:, 3:6] * self._T_SCALE_MM + torch.tensor(
            self._T_OFFSET_MM, device=raw.device
        )
        return R, t

    _T_SCALE_MM = 100.0
    _T_OFFSET_MM = (0.0, 650.0, 0.0)

    def _dsa_decode_view_labels(
        self,
        view_logits: torch.Tensor,
        metadata_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Keep the metadata view role but take the lateral side from the network."""
        labels = metadata_labels.long()
        is_lateral = labels != 1
        side = view_logits[:, (0, 2)].argmax(dim=1)
        lateral_labels = side * 2
        return torch.where(is_lateral, lateral_labels, torch.ones_like(labels))

    def _dann_lambda(self) -> float:
        """DANN sigmoid ramp λ = 2/(1+exp(−γ·p)) − 1, p = epoch progress in [0, 1]."""
        p     = self.current_epoch / max(self.trainer.max_epochs - 1, 1)
        gamma = self.cfg.get("da_gamma", 10.0)
        return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0

    def on_train_epoch_start(self):
        """Update epoch-dependent training weights."""
        self._da_lam = self._dann_lambda()

        self.net.fc.grl.alpha = self.cfg.get("da_alpha", 1.0) * self._da_lam
        self.log("train/da_alpha",   self.net.fc.grl.alpha)
        self.log("train/anneal_lam", self._da_lam)

        n = self.cfg.get("log_images_every_n_train_epochs", 10)
        self._train_log_buffer = [] if (self.current_epoch % n == 0) else None

    def training_step(self, batch, batch_idx):
        loss, R, t, pred_poses = self._shared_step(batch, "train")
        if self._train_log_buffer is not None:
            n_log = self.cfg.get("n_log_patients", 4)
            if batch_idx < n_log:
                self._train_log_buffer.append(
                    (batch, R.detach().cpu(), t.detach().cpu(), pred_poses)
                )
        return loss

    def on_train_epoch_end(self):
        if self._train_log_buffer and isinstance(self.logger, WandbLogger):
            self._log_drr_panel(self._train_log_buffer, "train")

    def on_validation_epoch_start(self):
        self._val_log_buffer = []

    def validation_step(self, batch, batch_idx):
        n_log = self.cfg.get("n_log_patients", 4)
        loss, R, t, pred_poses = self._shared_step(batch, "val")
        if batch_idx < n_log:
            self._val_log_buffer.append(
                (batch, R.detach().cpu(), t.detach().cpu(), pred_poses)
            )

    def on_validation_epoch_end(self):
        if self._val_log_buffer and isinstance(self.logger, WandbLogger):
            self._log_drr_panel(self._val_log_buffer, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def _shared_step(self, batch, stage: str):
        drr    = batch["drr"]
        images = batch["images"]
        poses  = batch["poses"]
        on_step = (stage == "train")

        out         = self._net_forward(
            self.net, images, batch["view_label"]
        )
        n           = self.cfg.num_outputs
        pred_params = out[:, :n]
        view_drr    = out[:, n+1:n+4]

        R, t = self._decode_pose(pred_params, view_label=batch["view_label"])
        pred_poses = convert(
            R, t,
            parameterization=self.cfg.parameterization,
            convention=self.cfg.convention,
        )

        art_end = batch.get("art_end", 3)
        drr.to(self.device)
        pred_mc     = drr(pred_poses, mask_to_channels=True)
        pred_images = pred_mc.sum(dim=1, keepdim=True)

        proj_gt_pts = proj_pred_pts = proj_height = proj_delx = None
        fid = batch.get("fiducials")
        if fid is not None:
            fid = fid.to(self.device).expand(images.shape[0], -1, -1)
            proj_gt_pts   = drr.perspective_projection(poses, fid)
            proj_pred_pts = drr.perspective_projection(pred_poses, fid)
            proj_height = int(drr.detector.height)
            proj_delx   = float(drr.detector.delx)
        drr.cpu()

        gt_art   = batch["art_gt"].to(self.device) if "art_gt" in batch else None
        pred_art = pred_mc[:, 1:art_end].sum(dim=1, keepdim=True)

        loss, terms = self.criterion(
            images=images, pred_images=pred_images, pred_poses=pred_poses, poses=poses,
            pred_art=pred_art, gt_art=gt_art,
            proj_pred_pts=proj_pred_pts, proj_gt_pts=proj_gt_pts,
            proj_height=proj_height, proj_delx=proj_delx,
            view_logit_drr=view_drr, drr_labels=batch["view_label"].long(),
            da_lam=self._da_lam,
        )
        for name, value in terms.items():
            self.log(f"{stage}/{name}", value, on_step=on_step, on_epoch=True)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=on_step, on_epoch=True)

        return loss, R.detach(), t.detach(), pred_poses

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = instantiate(self.cfg.optimizer, params=params)
        scheduler = instantiate(
            self.cfg.scheduler, optimizer=optimizer, T_max=self.trainer.max_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @staticmethod
    def _row_label(ax, text: str) -> None:
        """Bold rotated row label at the left edge of `ax`."""
        ax.text(-0.12, 0.5, text, transform=ax.transAxes, rotation=90,
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="#1f2328", clip_on=False)

    @torch.no_grad()
    def _log_drr_panel(self, log_buffer, stage):
        """Render one column per patient (first pose of each patient's batch)."""
        def mat_to_euler_t(pose):
            mat = pose.matrix.squeeze(0).cpu()

            trans = (mat[:3, :3].T @ mat[:3, 3]).numpy()
            return euler_zyx_from_matrix(mat[:3, :3]), trans

        def fmt_pose(pose):
            r, tr = mat_to_euler_t(pose)
            return (f"R ZYX (rad): {r[0]:+.3f}  {r[1]:+.3f}  {r[2]:+.3f}\n"
                    f"t     (mm):  {tr[0]:+.1f}  {tr[1]:+.1f}  {tr[2]:+.1f}")

        def fmt_raw(r, tr):
            return (f"R: {r[0]:+.3f}  {r[1]:+.3f}  {r[2]:+.3f}\n"
                    f"t (mm):  {tr[0]:+.1f}  {tr[1]:+.1f}  {tr[2]:+.1f}")

        N = len(log_buffer)
        row_labels = ["Iso Pose", "Random Pose (GT)", "Predicted Pose"]
        n_rows = len(row_labels)
        fig, axes = plt.subplots(n_rows, N, figsize=(5 * N, 4 * n_rows), squeeze=False)
        fig.subplots_adjust(bottom=0.08, top=0.93, hspace=0.5)
        fig.suptitle(f"Epoch {self.current_epoch}  —  one column per patient", fontsize=13)

        for row, label in enumerate(row_labels):
            self._row_label(axes[row, 0], label)

        for col, (batch, R, t, pred_poses) in enumerate(log_buffer):
            drr       = batch["drr"]
            images    = batch["images"]
            image_iso = batch["image_iso"]
            pose_iso  = batch["pose_iso"]
            R_gt      = batch["R"]
            t_gt      = batch["t"]

            pred_pose_0 = convert(
                R[:1].to(self.device), t[:1].to(self.device),
                parameterization=self.cfg.parameterization,
                convention=self.cfg.convention,
            )
            drr.to(self.device)
            pred_img = drr(pred_pose_0).squeeze()
            drr.cpu()

            imgs = [image_iso[0].squeeze(), images[0].squeeze(), pred_img]
            captions = [
                fmt_pose(pose_iso),
                fmt_raw(R_gt[0].cpu().numpy(), t_gt[0].cpu().numpy()),
                fmt_raw(R[0].numpy(), t[0].numpy()),
            ]
            for row, (img, caption) in enumerate(zip(imgs, captions)):
                ax = axes[row, col]
                ax.imshow(_to_np(img), cmap="gray")
                if row == 0:
                    ax.set_title(f"Patient {col}", fontsize=10, fontweight="bold", pad=6)
                ax.axis("off")
                ax.text(0.5, -0.04, caption, transform=ax.transAxes,
                        ha="center", va="top", fontsize=8, family="monospace")

        self.logger.experiment.log(
            {f"{stage}/pose_images": wandb.Image(fig), "trainer/global_step": self.global_step}
        )
        plt.close(fig)
