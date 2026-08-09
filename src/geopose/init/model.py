import copy
import math
import re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
import wandb

from pathlib import Path

from diffdrr.pose import RigidTransform, convert
from hydra.utils import instantiate
from omegaconf import DictConfig
from pytorch_lightning.loggers import WandbLogger

from ..shared.blocks import build_resnet_backbone, _PoseDomainHead
from .loss import GeoPoseCriterion
from ..shared.metrics import mpcd as _mpcd_metric
from ..shared.pose import delta_to_pose
from ..refine.network import (
    build_refine_pose_net,
    load_refine_pose_checkpoint,
    refiner_view_index,
)
from ..shared.visualization import euler_zyx_from_matrix, seg_overlay_rgb
from ..shared.visualization import to_np as _to_np


# ── Model ────────────────────────────────────────────────────────────────────

class ResNetPose(pl.LightningModule):

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.save_hyperparameters()
        self.cfg = cfg

        # 1-channel (grayscale) backbone; the torchvision fc is swapped for the
        # combined pose + domain + view head (see geopose.shared.blocks).
        net, in_features = build_resnet_backbone(cfg.backbone, 1, cfg.pretrained)
        self.view_role_emb_dim = int(cfg.get("view_role_emb_dim", 0))
        # 2 (default) = binary PA/LAT role, byte-identical to every checkpoint
        # predating this option. 3 = signed LAT-/PA/LAT+ role — closes the gap
        # where the pose head was less informed than the decode anchor sitting
        # right after it (see _net_forward_predicted_role).
        self.view_role_emb_classes = int(cfg.get("view_role_emb_classes", 2))
        if self.view_role_emb_classes not in (2, 3):
            raise ValueError(
                "model.view_role_emb_classes must be 2 (binary PA/LAT) or "
                f"3 (signed LAT-/PA/LAT+), got {self.view_role_emb_classes}"
            )
        net.fc = _PoseDomainHead(
            in_features,
            cfg.num_outputs,
            cfg.get("da_alpha", 1.0),
            role_emb_dim=self.view_role_emb_dim,
            role_classes=self.view_role_emb_classes,
        )
        self.net = net

        # ── LXPose-style end-to-end refinement network (net2) ─────────────────
        # Optional second CNN that refines net1's prediction. Gated by
        # cfg.refine.enabled so that, when off, no extra parameters are
        # registered and the state_dict stays byte-identical (existing
        # checkpoints load unchanged). When on, it consumes
        # [input DRR | render-at-prediction] and regresses a 6-DoF pose delta.
        refine_cfg = cfg.get("refine", {})
        self.refine_net = None
        if refine_cfg.get("enabled", False):
            refine_ckpt = refine_cfg.get("init_ckpt", None)
            self.refine_net = (
                load_refine_pose_checkpoint(refine_ckpt)
                if refine_ckpt
                else build_refine_pose_net(refine_cfg)
            )
            if refine_cfg.get("freeze", bool(refine_ckpt)):
                self.refine_net.requires_grad_(False)
                self.refine_net.eval()

        # Optional warm-start for net1 from a prior GeoPose checkpoint. Loads only
        # the ``net.*`` tensors (the primary CNN); refine_net and the criterion
        # start fresh. Useful for the refine-cascade run: init net1 from a
        # converged refine-off checkpoint, then train the cascade on top.
        init_ckpt = cfg.get("init_net_ckpt", None)
        if init_ckpt:
            self._init_net_from_ckpt(init_ckpt)

        # Optionally freeze net1 so only the refine cascade (net2) trains. Done
        # after the warm-start load; params get requires_grad=False and net1 is
        # held in eval so its BatchNorm running stats stay fixed (see `train`
        # below, which re-asserts eval each time Lightning toggles train mode).
        # Pairs with init_net_ckpt to train net2 on top of a frozen net1.
        if cfg.get("freeze_net1", False):
            for p in self.net.parameters():
                p.requires_grad_(False)
            self.net.eval()

        # All loss math (primary NCC/geodesic, Dice, MVC, refine, DA, MAP NCC/r0,
        # view-CE) + weights/anneals live in the criterion; the step below just
        # builds operands and logs the returned per-term scalars.
        self.criterion = GeoPoseCriterion(cfg)

        # ── EMA mean-teacher (optional) ───────────────────────────────────────
        # One EMA-of-student teacher shared by two consistency objectives:
        #   • ema.enabled     — synthetic DRR: teacher sees the clean DRR, student
        #                       the augmented one (criterion's L_ema).
        #   • ema.dsa.enabled — UNLABELED real DSA pool: teacher sees a weakly-
        #                       augmented MAP, student a strongly-augmented one
        #                       (semi-supervised; criterion's L_ema_dsa).
        # Built whenever either is on. Gated like the refine cascade so that, when
        # both are off, no extra parameters are registered and existing checkpoints
        # load bit-identically. Built AFTER the optional warm-start / freeze so the
        # teacher starts byte-identical to the student. Held in eval and
        # requires_grad=False — the EMA update (on_train_batch_end) is the only thing
        # that moves it; its BatchNorm uses the EMA'd running stats.
        ema_cfg = cfg.get("ema", {})
        self.teacher_net = None
        if ema_cfg.get("enabled", False) or ema_cfg.get("dsa", {}).get("enabled", False):
            self.teacher_net = copy.deepcopy(self.net)
            self.teacher_net.requires_grad_(False)
            self.teacher_net.eval()

        # Current DANN ramp value (0→1). Updated each train epoch; shared by the
        # GRL alpha anneal and the optional Dice / DRR-view-CE weight anneals.
        # Starts at 0 so the pre-train validation sanity check matches epoch-0.
        self._da_lam = 0.0

    # ── Warm-start ─────────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_ckpt(path: str) -> Path:
        """Resolve `path` to a single .ckpt file.

        Accepts either a checkpoint file or a run/output directory. For a
        directory, globs recursively and returns the checkpoint with the highest
        ``epoch=N`` in its filename (falling back to alphabetical when no run is
        encoded), so pointing at a Hydra output dir picks the latest checkpoint.
        """
        p = Path(path)
        if p.is_file():
            return p
        if p.is_dir():
            ckpts = sorted(p.rglob("*.ckpt"))
            if not ckpts:
                raise FileNotFoundError(f"No .ckpt under {p}")
            def _epoch(c: Path) -> int:
                m = re.search(r"epoch=(\d+)", c.name)
                return int(m.group(1)) if m else -1
            return max(ckpts, key=_epoch)
        raise FileNotFoundError(f"init_net_ckpt path does not exist: {p}")

    def _init_net_from_ckpt(self, path: str) -> None:
        """Load only the net1 (`net.*`) weights from a GeoPose checkpoint.

        Uses ``strict=False`` so the load is robust to head/criterion changes;
        the refine network and criterion are left at their fresh init.
        """
        ckpt = self._resolve_ckpt(path)
        state = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
        net_sd = {k[len("net."):]: v for k, v in state.items() if k.startswith("net.")}
        if not net_sd:
            raise RuntimeError(f"No 'net.*' tensors found in {ckpt}")
        missing, unexpected = self.net.load_state_dict(net_sd, strict=False)
        print(f"[ResNetPose] warm-started net1 from {ckpt} "
              f"({len(net_sd)} tensors; {len(missing)} missing, "
              f"{len(unexpected)} unexpected)")

    def train(self, mode: bool = True):
        """Keep net1 in eval when frozen, even as Lightning toggles train mode.

        Lightning calls ``model.train()`` at the start of each training epoch,
        which would otherwise re-enable BatchNorm running-stat updates on the
        frozen net1. Re-asserting ``self.net.eval()`` here pins those stats.
        """
        super().train(mode)
        if self.cfg.get("freeze_net1", False):
            self.net.eval()
        if self.refine_net is not None and self.cfg.get("refine", {}).get("freeze", False):
            self.refine_net.eval()
        if self.teacher_net is not None:
            # EMA teacher is always in eval: its BatchNorm consumes the EMA'd
            # running stats, and the no-grad teacher forward must not update them.
            self.teacher_net.eval()
        return self

    # ── Forward ──────────────────────────────────────────────────────────────

    def forward(
        self,
        x: torch.Tensor,
        view_label: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Pose params [B, num_outputs], optionally conditioned on known view role."""
        return self._net_forward(self.net, x, view_label)[:, :self.cfg.num_outputs]

    @staticmethod
    def _view_role(view_label: torch.Tensor) -> torch.Tensor:
        """Map signed view labels {LAT-=0, PA=1, LAT+=2} to {PA=0, LAT=1}."""
        return (view_label.long() != 1).long()

    def _role_from_signed(self, view_label: torch.Tensor) -> torch.Tensor:
        """Role fed to the pose-branch embedding, from a signed {0,1,2} label.

        view_role_emb_classes=2 (default): collapse to binary PA/LAT via
        _view_role. =3: pass the signed label through unchanged — the pose
        head then sees the same LAT- vs LAT+ distinction the decode anchor
        already conditions on.
        """
        if self.view_role_emb_classes == 3:
            return view_label.long()
        return self._view_role(view_label)

    @staticmethod
    def _backbone_features(net: nn.Module, images: torch.Tensor) -> torch.Tensor:
        """Pooled backbone features [B, in_features] (torchvision's ResNet up
        to but not including ``fc``). Reused by both the role-independent aux
        heads (domain/view) and the role-conditioned pose head so the backbone
        runs exactly once regardless of how many downstream heads are called.
        """
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
        """Forward a GeoPose ResNet, injecting a known role when enabled.

        For callers where the role-conditioning label is already known BEFORE
        any forward pass (synthetic training's teacher-forced GT view_label —
        there is no prediction to sequence around). The disabled path remains
        the original ``net(images)`` call byte-for-byte. Real-DSA eval, where
        the label may itself depend on this net's own view logits, uses
        ``_net_forward_predicted_role`` instead.
        """
        if self.view_role_emb_dim == 0:
            return net(images)
        if view_label is None:
            raise ValueError(
                "view_label is required when model.view_role_emb_dim is non-zero"
            )
        features = self._backbone_features(net, images)
        return net.fc(features, self._role_from_signed(view_label))

    def _net_forward_predicted_role(
        self,
        net: nn.Module,
        images: torch.Tensor,
        metadata_label: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Two-phase forward for the real-DSA eval path.

        The signed label the pose branch (and the decode anchor) condition on
        may depend on this net's OWN view logits
        (``model.dsa_view_anchor_mode=known_role_predicted_side``): known
        acquisition order supplies PA-vs-LAT, but the LAT- vs LAT+ side comes
        from the image-only view classifier. That is a real dependency, not a
        detail — feeding the pose branch the metadata label instead (as if the
        side were known) is exactly the inconsistency this method closes.

        Runs the backbone ONCE: (1) role-independent aux heads (domain/view)
        resolve ``decode_view_label`` via ``_dsa_decode_view_labels``; (2) the
        pose head is conditioned on that resolved label. Returns
        ``(full_out, decode_view_label)`` where ``full_out`` matches
        ``_net_forward``'s ``[pose | domain | view]`` layout, so every
        existing caller-side slice (``out[:, :n]`` / ``out[:, n:n+1]`` /
        ``out[:, n+1:n+4]``) is unchanged.

        Equivalent to the old metadata-conditioned path when
        ``view_role_emb_classes=2`` — ``_view_role(decode_view_label) ==
        _view_role(metadata_label)`` always, because ``_dsa_decode_view_labels``
        never flips PA<->LAT, it only resolves the LAT side — so this is a
        unification of the two real-DSA call sites, not a behaviour change,
        for the existing arm. Verified in
        ``scripts/smoke/smoke_test_geopose_role_embedding.py``.
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
        pose_params = net.fc.forward_pose(
            features, self._role_from_signed(decode_view_label)
        )
        return torch.cat([pose_params, domain_logit, view_logits], dim=1), decode_view_label

    # Sign mapping for 3-class view labels — matches geopose.init.data.
    # 0 → LAT(-π/2), 1 → PA (0), 2 → LAT(+π/2)
    _VIEW_SIGN = (-1.0, 0.0, 1.0)

    def _decode_pose(self, raw: torch.Tensor, view_label: torch.Tensor | None = None):
        """Split raw network output into (R [B,3], t_mm [B,3]).

        Network outputs are interpreted as *residuals* from a view-conditional
        isopose. If `view_label` is provided (long, in {0,1,2}) AND
        ``cfg.view_anchor`` is True (default), a signed π/2 anchor is added to
        R[:,0]: LAT(-π/2) → -π/2, PA → 0, LAT(+π/2) → +π/2. With ``view_anchor``
        False (or no view_label), the raw rotation is returned unchanged and the
        network must regress the full rotation itself (LXPose-style, no view-label
        dependence). Translation always carries the fixed [0, 650, 0] mm offset.
        """
        R = raw[:, :3]
        if view_label is not None and self.cfg.get("view_anchor", True):
            sign_lookup = torch.tensor(self._VIEW_SIGN, device=R.device, dtype=R.dtype)
            sign = sign_lookup[view_label.long()]                 # [B]
            offset = torch.zeros_like(R)
            offset[:, 0] = sign * (math.pi / 2)
            R = R + offset
        t = raw[:, 3:6] * 100 + torch.tensor([0, 650, 0], device=raw.device)
        return R, t

    def _dsa_decode_view_labels(
        self,
        view_logits: torch.Tensor,
        metadata_labels: torch.Tensor,
    ) -> torch.Tensor:
        """Choose the anchor label used to decode real DSA pose predictions.

        ``metadata`` preserves the historical path: use the signed C-arm label
        directly. ``known_role_predicted_side`` assumes acquisition ordering
        tells us PA versus LAT, but makes the image head choose LAT- versus LAT+.
        The PA class is masked from that choice, so a known LAT can never be
        decoded as frontal. ``known_role_fixed_lat_plus`` is the protocol-prior
        control for this heavily LAT+-skewed cohort.
        """
        mode = str(self.cfg.get("dsa_view_anchor_mode", "metadata"))
        labels = metadata_labels.long()
        if mode == "metadata":
            return labels

        is_lateral = labels != 1
        if mode == "known_role_predicted_side":
            # Conditional binary decision: column 0=LAT-, column 1=LAT+.
            side = view_logits[:, (0, 2)].argmax(dim=1)
            lateral_labels = side * 2
        elif mode == "known_role_fixed_lat_plus":
            lateral_labels = torch.full_like(labels, 2)
        else:
            raise ValueError(
                "model.dsa_view_anchor_mode must be metadata, "
                "known_role_predicted_side, or known_role_fixed_lat_plus; "
                f"got {mode!r}"
            )
        return torch.where(is_lateral, lateral_labels, torch.ones_like(labels))

    # ── Training / validation / test steps ───────────────────────────────────

    def _dann_lambda(self) -> float:
        """DANN sigmoid ramp λ = 2/(1+exp(−γ·p)) − 1, p = epoch progress in [0, 1].

        Shared by the GRL alpha anneal and the optional Dice / DRR-view-CE weight
        anneals (``dice_anneal`` / ``view_cls_drr_anneal``) so all three follow
        the same 0→1 schedule over training, governed by ``da_gamma``.
        """
        p     = self.current_epoch / max(self.trainer.max_epochs - 1, 1)
        gamma = self.cfg.get("da_gamma", 10.0)
        return 2.0 / (1.0 + math.exp(-gamma * p)) - 1.0

    def on_train_epoch_start(self):
        """Anneal GRL alpha (and, when enabled, the Dice / DRR-view-CE weights)
        via the shared DANN schedule: λ = 2/(1+exp(−γ·p)) − 1."""
        self._da_lam = self._dann_lambda()

        self.net.fc.grl.alpha = self.cfg.get("da_alpha", 1.0) * self._da_lam
        self.log("train/da_alpha",   self.net.fc.grl.alpha)
        self.log("train/anneal_lam", self._da_lam)

        n = self.cfg.get("log_images_every_n_train_epochs", 10)
        self._train_log_buffer = [] if (self.current_epoch % n == 0) else None

    def training_step(self, batch, batch_idx):
        loss, R, t, pred_poses, corrected = self._shared_step(batch, "train")
        if self._train_log_buffer is not None:
            n_log = self.cfg.get("n_log_patients", 4)
            if batch_idx < n_log:
                self._train_log_buffer.append(
                    (batch, R.detach().cpu(), t.detach().cpu(), pred_poses, corrected)
                )
        return loss

    @torch.no_grad()
    def _ema_update(self):
        """θ_teacher ← m·θ_teacher + (1−m)·θ_student for the EMA mean-teacher.

        Covers parameters and float buffers (e.g. BatchNorm running stats);
        integer buffers (``num_batches_tracked``) are copied straight over. Relies
        on the teacher being a deepcopy of net1, so the two iterators line up
        positionally. Called once per optimizer step from ``on_train_batch_end``.
        """
        m = float(self.cfg.get("ema", {}).get("decay", 0.999))
        for tp, sp in zip(self.teacher_net.parameters(), self.net.parameters()):
            tp.mul_(m).add_(sp.detach(), alpha=1.0 - m)
        for tb, sb in zip(self.teacher_net.buffers(), self.net.buffers()):
            if tb.dtype.is_floating_point:
                tb.mul_(m).add_(sb.detach(), alpha=1.0 - m)
            else:
                tb.copy_(sb)

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # EMA the teacher after the optimizer has stepped the student.
        if self.teacher_net is not None:
            self._ema_update()

    def on_train_epoch_end(self):
        if self._train_log_buffer and isinstance(self.logger, WandbLogger):
            self._log_drr_panel(self._train_log_buffer, "train")
            self._log_map_panel(self._train_log_buffer[0][0], "train")

    def on_validation_epoch_start(self):
        self._val_log_buffer = []  # list of (batch, R, t, pred_poses) per patient

    def validation_step(self, batch, batch_idx):
        n_log = self.cfg.get("n_log_patients", 4)
        loss, R, t, pred_poses, corrected = self._shared_step(batch, "val")
        if batch_idx < n_log:
            self._val_log_buffer.append(
                (batch, R.detach().cpu(), t.detach().cpu(), pred_poses, corrected)
            )

    def on_validation_epoch_end(self):
        if self._val_log_buffer and isinstance(self.logger, WandbLogger):
            self._log_drr_panel(self._val_log_buffer, "val")
            self._log_map_panel(self._val_log_buffer[0][0], "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def _shared_step(self, batch, stage: str):
        drr    = batch["drr"]
        images = batch["images"]   # [B, C, H, W] — B random poses, single patient
        poses  = batch["poses"]    # RigidTransform [B, 4, 4]
        on_step = (stage == "train")

        # ── DRR forward pass ─────────────────────────────────────────────────
        out         = self._net_forward(
            self.net, images, batch["view_label"]
        )                                            # [B, num_outputs + 4]
        n           = self.cfg.num_outputs
        pred_params = out[:, :n]                     # [B, 6]
        domain_drr  = out[:, n:n+1]                  # [B, 1]  label = 0
        view_drr    = out[:, n+1:n+4]                # [B, 3]  view logits (LAT-, PA, LAT+)

        R, t = self._decode_pose(pred_params, view_label=batch["view_label"])
        pred_poses = convert(
            R, t,
            parameterization=self.cfg.parameterization,
            convention=self.cfg.convention,
        )

        # ── EMA mean-teacher targets (train only) ────────────────────────────
        # The teacher (an EMA of the student) provides no-grad consistency targets.
        # view_label is geometry (augmentation-invariant), so the same decode anchor
        # applies to both teacher and student.
        ema_cfg = self.cfg.get("ema", {})

        # (1) Synthetic DRR: teacher views the UN-augmented DRR; its pose is the
        # target for the student's augmented-DRR pose (criterion's L_ema). One extra
        # CNN forward, no render.
        teacher_poses = None
        if self.teacher_net is not None and stage == "train" and ema_cfg.get("enabled", False):
            with torch.no_grad():
                teacher_out = self._net_forward(
                    self.teacher_net, batch["images_clean"], batch["view_label"]
                )
                R_te, t_te  = self._decode_pose(teacher_out[:, :n], view_label=batch["view_label"])
                teacher_poses = convert(
                    R_te, t_te,
                    parameterization=self.cfg.parameterization,
                    convention=self.cfg.convention,
                )

        # (2) Unlabeled real DSA: teacher views the WEAKLY-augmented MAP, student the
        # STRONGLY-augmented one (shared geometry, pre-flip labels). No GT pose ⇒ the
        # teacher's read is the only target (criterion's L_ema_dsa). Valid (non-empty)
        # MAP channels only — CTA patients without DSA are all-zero and skipped.
        teacher_map_poses = student_map_poses = None
        if (self.teacher_net is not None and stage == "train"
                and ema_cfg.get("dsa", {}).get("enabled", False) and "maps_strong" in batch):
            ms = batch["maps_strong"].unsqueeze(1).float().to(self.device)  # [4, 1, H, W]
            mw = batch["maps_weak"].unsqueeze(1).float().to(self.device)
            vmask = ms.any(dim=(1, 2, 3))                                   # [4] bool
            if vmask.any():
                vlbl = batch["map_view_labels_clean"].to(self.device).long()[vmask]
                student_map_out = self._net_forward(self.net, ms[vmask], vlbl)
                R_sm, t_sm = self._decode_pose(student_map_out[:, :n], view_label=vlbl)
                student_map_poses = convert(R_sm, t_sm,
                                            parameterization=self.cfg.parameterization,
                                            convention=self.cfg.convention)
                with torch.no_grad():
                    teacher_map_out = self._net_forward(self.teacher_net, mw[vmask], vlbl)
                    R_tm, t_tm = self._decode_pose(teacher_map_out[:, :n], view_label=vlbl)
                    teacher_map_poses = convert(R_tm, t_tm,
                                                parameterization=self.cfg.parameterization,
                                                convention=self.cfg.convention)

        # Render per-label so we can both sum to the full projection (for NCC)
        # and pull the artery channels out (for Dice). mask_to_channels puts label
        # i in channel i; the artery target is channels 1:art_end — carotid (1:3)
        # for CTA, the union of all vessels (1:N) for TopBrain. art_end rides in
        # the batch from the dataloader (default 3). The GT-pose artery target
        # (batch["art_gt"]) is precomputed there with the same slice.
        art_end = batch.get("art_end", 3)
        drr.to(self.device)
        pred_mc     = drr(pred_poses, mask_to_channels=True)  # [B, n_labels, H, W]
        pred_images = pred_mc.sum(dim=1, keepdim=True)        # [B, 1, H, W] full projection

        # ── End-to-end refinement (net2) ─────────────────────────────────────
        # Predict a 6-DoF pose delta from [input DRR | render-at-prediction],
        # correct on the SE(3) manifold (pred ∘ δ⁻¹, matching RefinePoseModule's
        # convention), and re-render in this same on-device window. No detach
        # anywhere (LXPose parity): the refine loss backprops into net1 through
        # both the corrected pose and the render fed to net2. The small-init head
        # gives δ≈0 at start ⇒ corrected≈pred, so the cascade begins as a near
        # no-op and pairs cleanly with the loss-weight warmup.
        corrected_pose = corrected_mc = None
        if self.refine_net is not None:
            refine_view = refiner_view_index(
                self.refine_net, batch["view_label"], images.shape[0]
            ).to(images.device)
            dR, dt = self.refine_net(images, self._normalize(pred_images), refine_view)
            corrected_pose = pred_poses.compose(delta_to_pose(dR, dt).inverse())
            corrected_mc   = drr(corrected_pose, mask_to_channels=True)  # 2nd masked render

        # ── Skeleton-fiducial projections for the LXPose projection loss ──────
        # Project the carotid-skeleton fiducials (batch["fiducials"], world coords
        # already in the DRR's centered frame) under the GT, net1, and corrected
        # poses → detector pixels. Pure geometry (no render), fully differentiable,
        # so the reprojection error backprops into net1 (and net2, no detach).
        # Done in this on-device window; None when fiducials are disabled.
        proj_gt_pts = proj_pred_pts = proj_corr_pts = proj_height = proj_delx = None
        fid = batch.get("fiducials")
        if fid is not None:
            fid = fid.to(self.device).expand(images.shape[0], -1, -1)   # [B, N, 3]
            proj_gt_pts   = drr.perspective_projection(poses, fid)
            proj_pred_pts = drr.perspective_projection(pred_poses, fid)
            if corrected_pose is not None:
                proj_corr_pts = drr.perspective_projection(corrected_pose, fid)
            proj_height = int(drr.detector.height)
            proj_delx   = float(drr.detector.delx)
        drr.cpu()

        # ── Loss operands (reductions + weights + anneals live in the criterion) ──
        gt_art   = batch["art_gt"].to(self.device) if "art_gt" in batch else None
        pred_art = pred_mc[:, 1:art_end].sum(dim=1, keepdim=True)   # [B, 1, H, W] artery path-length

        corrected_img = corrected_art = None
        if corrected_mc is not None:
            corrected_img = corrected_mc.sum(dim=1, keepdim=True)
            corrected_art = corrected_mc[:, 1:art_end].sum(dim=1, keepdim=True)  # RAW path-length (matches art_gt scale)

        # Single MAP forward pass shared by all MAP losses.
        valid_maps = map_out = valid_mask = valid_map_view_labels = decode_view_label = None
        if "maps" in batch:
            maps_batch = batch["maps"].unsqueeze(1).float().to(self.device)  # [4, 1, H, W]
            valid_mask = maps_batch.any(dim=(1, 2, 3))                       # [4]
            if valid_mask.any():
                valid_maps = maps_batch[valid_mask]                          # [K, 1, H, W]
                valid_map_view_labels = (
                    batch["map_view_labels"].to(self.device).long()[valid_mask]
                )
                # Two-phase: aux heads resolve decode_view_label (the predicted
                # LAT side under known_role_predicted_side) BEFORE the pose
                # branch runs, so the pose head sees the same label the
                # decode anchor conditions on — see _net_forward_predicted_role.
                map_out, decode_view_label = self._net_forward_predicted_role(
                    self.net, valid_maps, valid_map_view_labels
                )                                                           # [K, num_outputs+4]

        domain_map = view_logit_map = map_labels = None
        r0_pred = r0_target = map_ncc_pred = map_ncc_target = None
        mpcd_val = mpcd_refined_val = None
        if map_out is not None:
            domain_map     = map_out[:, n:n+1]
            view_logit_map = map_out[:, n+1:n+4]
            view_label     = valid_map_view_labels                           # [K]
            map_labels     = view_label

            lateral_mask_map = view_label != 1
            if lateral_mask_map.any():
                dsa_lat_side_acc = (
                    decode_view_label[lateral_mask_map]
                    == view_label[lateral_mask_map]
                ).float().mean()
                self.log(
                    f"{stage}/dsa_lat_side_acc",
                    dsa_lat_side_acc,
                    on_step=on_step,
                    on_epoch=True,
                    batch_size=int(lateral_mask_map.sum()),
                )

            # r0 penalty pulls the rotation *residual* toward the view-conditional
            # isopose; computed whenever MAP data is present (matches old path).
            R_p, t_p    = self._decode_pose(
                map_out[:, :n], view_label=decode_view_label
            )
            sign_lookup = torch.tensor(self._VIEW_SIGN, device=self.device, dtype=R_p.dtype)
            r0_target   = sign_lookup[view_label] * (math.pi / 2)
            r0_pred     = R_p[:, 0]

            # Render MAP-predicted DRRs for the NCC term — skipped during training
            # when NCC is off; still rendered in val/test to log map_ncc + mpcd.
            if not (stage == "train" and self.cfg.get("lambda_map_ncc", 0.0) == 0.0):
                pred_poses_map = convert(R_p, t_p,
                                         parameterization=self.cfg.parameterization,
                                         convention=self.cfg.convention)
                delx, dely = drr.detector.delx, drr.detector.dely
                drr.to(self.device)
                pred_drrs = drr(pred_poses_map, mask_to_channels=True)   # [K, C, H, W]
                # Vessels kept PER-CHANNEL (not summed): mpcd scores each carotid
                # separately against the one-sided DSA seg and keeps the best —
                # summing would average over the unopacified side (see metrics.mpcd).
                pred_arts = pred_drrs[:, 1:art_end] > 0                  # [K, C-1, H, W]
                pred_drrs = pred_drrs.sum(dim=1, keepdim=True)           # [K, 1, H, W] render-at-pred

                # Inference-time refinement on the REAL DSA path (val/test only):
                # net2 sees [MAP | render-at-prediction], corrects net1's MAP pose
                # (pred ∘ δ⁻¹ — same convention as the synthetic path and the MAP
                # panel), re-renders, and we score mpcd on the corrected arteries.
                # Metric only (no grad, no train cost) — the read on whether the
                # cascade actually improves real-angiography alignment, since the
                # plain `mpcd` above sees net1 only.
                refined_arts = None
                if self.refine_net is not None and stage != "train":
                    refine_view = refiner_view_index(
                        self.refine_net, decode_view_label, valid_maps.shape[0]
                    ).to(valid_maps.device)
                    dR_m, dt_m = self.refine_net(self._normalize(valid_maps),
                                                 self._normalize(pred_drrs),
                                                 refine_view)
                    refined_poses = pred_poses_map.compose(delta_to_pose(dR_m, dt_m).inverse())
                    refined_arts  = drr(refined_poses, mask_to_channels=True)[:, 1:art_end] > 0
                drr.cpu()

                valid_segs = (batch["segs"].to(self.device)[valid_mask].unsqueeze(1)
                              if ("segs" in batch and valid_mask is not None and valid_mask.any())
                              else None)
                if stage != "train" and valid_segs is not None:
                    with torch.no_grad():
                        mpcd_val = self._mpcd(pred_arts, valid_segs, delx=delx, dely=dely)
                        if refined_arts is not None:
                            mpcd_refined_val = self._mpcd(refined_arts, valid_segs, delx=delx, dely=dely)
                map_ncc_pred, map_ncc_target = pred_drrs, valid_maps

        # ── Assemble the loss + per-term scalars to log ───────────────────────
        loss, terms = self.criterion(
            images=images, pred_images=pred_images, pred_poses=pred_poses, poses=poses,
            teacher_poses=teacher_poses,
            teacher_map_poses=teacher_map_poses, student_map_poses=student_map_poses,
            pred_art=pred_art, gt_art=gt_art,
            corrected_img=corrected_img, corrected_art=corrected_art,
            corrected_pose=corrected_pose,
            proj_pred_pts=proj_pred_pts, proj_gt_pts=proj_gt_pts,
            proj_corr_pts=proj_corr_pts, proj_height=proj_height, proj_delx=proj_delx,
            domain_drr=domain_drr, domain_map=domain_map,
            map_ncc_pred=map_ncc_pred, map_ncc_target=map_ncc_target,
            r0_pred=r0_pred, r0_target=r0_target,
            view_logit_drr=view_drr, drr_labels=batch["view_label"].long(),
            view_logit_map=view_logit_map, map_labels=map_labels,
            da_lam=self._da_lam,
        )
        for name, value in terms.items():
            self.log(f"{stage}/{name}", value, on_step=on_step, on_epoch=True)
        if mpcd_val is not None:
            self.log(f"{stage}/mpcd", mpcd_val, on_step=False, on_epoch=True)
        if mpcd_refined_val is not None:
            self.log(f"{stage}/mpcd_refined", mpcd_refined_val, on_step=False, on_epoch=True)
        self.log(f"{stage}/loss", loss, prog_bar=True, on_step=on_step, on_epoch=True)

        # Hand the corrected pose to the logger as a detached, CPU-resident
        # RigidTransform so the per-patient log buffer never pins the GPU
        # autograd graph across the epoch (R/t are detached/cpu'd for the same
        # reason). None when refinement is disabled — the panel skips its row.
        corrected_cpu = (
            None if corrected_pose is None
            else RigidTransform(corrected_pose.matrix.detach().cpu())
        )
        return loss, R.detach(), t.detach(), pred_poses, corrected_cpu

    @staticmethod
    def _normalize(images: torch.Tensor) -> torch.Tensor:
        """Per-image min-max normalise a [N, C, H, W] tensor to [0, 1].

        Identical to CTAPoseDataset._normalize / RefinePoseModule._normalize, so
        the render-at-prediction fed to the refinement network sits on the same
        intensity scale as the dataloader's inputs.
        """
        N, C, H, W = images.shape
        flat = images.reshape(N * C, -1)
        vmin = flat.min(dim=1).values.view(N, C, 1, 1)
        vmax = flat.max(dim=1).values.view(N, C, 1, 1)
        return (images - vmin) / (vmax - vmin).clamp(min=1e-8)

    _mpcd = staticmethod(_mpcd_metric)

    # ── Optimizers ───────────────────────────────────────────────────────────

    def configure_optimizers(self):
        # Only optimise trainable params — when net1 is frozen this excludes it,
        # so AdamW's state isn't allocated for parameters that never get grads.
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = instantiate(self.cfg.optimizer, params=params)
        scheduler = instantiate(
            self.cfg.scheduler, optimizer=optimizer, T_max=self.trainer.max_epochs,
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    # ── Logging ──────────────────────────────────────────────────────────────

    @staticmethod
    def _row_label(ax, text: str) -> None:
        """Bold rotated row label at the left edge of `ax`.

        Uses a text artist rather than ``set_ylabel`` because the panels call
        ``ax.axis("off")``, which hides the y-axis (and thus any ylabel); a text
        artist still renders. ``clip_on=False`` keeps it visible in the left
        margin. Callers leave room via ``subplots_adjust(left=…)``.
        """
        ax.text(-0.12, 0.5, text, transform=ax.transAxes, rotation=90,
                ha="center", va="center", fontsize=12, fontweight="bold",
                color="#1f2328", clip_on=False)

    @torch.no_grad()
    def _log_drr_panel(self, log_buffer, stage):
        """Render one column per patient (first pose of each patient's batch).

        log_buffer entries are
        ``(batch, R [B,3], t [B,3], pred_poses RigidTransform [B], corrected)`` —
        one per patient. ``corrected`` is the refined-pose RigidTransform (CPU)
        when the refinement cascade is enabled, else ``None``; when present it
        adds a 4th "Refined Pose" row.
        """
        def mat_to_euler_t(pose):
            mat = pose.matrix.squeeze(0).cpu()
            # diffdrr builds the matrix as rotate∘translate, so mat[:3,3] = R·t,
            # NOT the translation *parameter*. Undo the rotation (t = Rᵀ·mat[:3,3])
            # so this matches the raw-param t shown for the GT/predicted rows
            # (otherwise the refined row's t is in a different frame — same mm, but
            # rotated, which reads as "wrong units" for the ~±π/2 lateral poses).
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
        if self.refine_net is not None:
            row_labels.append("Refined Pose")
        n_rows = len(row_labels)
        fig, axes = plt.subplots(n_rows, N, figsize=(5 * N, 4 * n_rows), squeeze=False)
        fig.subplots_adjust(bottom=0.08, top=0.93, hspace=0.5)
        fig.suptitle(f"Epoch {self.current_epoch}  —  one column per patient", fontsize=13)

        for row, label in enumerate(row_labels):
            self._row_label(axes[row, 0], label)

        for col, (batch, R, t, pred_poses, corrected) in enumerate(log_buffer):
            drr       = batch["drr"]
            images    = batch["images"]     # [B, C, H, W]
            image_iso = batch["image_iso"]  # [1, C, H, W]
            pose_iso  = batch["pose_iso"]   # RigidTransform [1, 4, 4]
            R_gt      = batch["R"]          # [B, 3]
            t_gt      = batch["t"]          # [B, 3]

            # Render the first predicted pose (and the refined pose, when the
            # cascade is on) for this patient in a single on-device window.
            pred_pose_0 = convert(
                R[:1].to(self.device), t[:1].to(self.device),
                parameterization=self.cfg.parameterization,
                convention=self.cfg.convention,
            )
            drr.to(self.device)
            pred_img = drr(pred_pose_0).squeeze()
            refined_img = refined_caption = None
            if corrected is not None:
                corrected_pose_0 = RigidTransform(corrected.matrix[:1].to(self.device))
                refined_img      = drr(corrected_pose_0).squeeze()
                refined_caption  = fmt_pose(corrected_pose_0)
            drr.cpu()

            imgs = [image_iso[0].squeeze(), images[0].squeeze(), pred_img]
            captions = [
                fmt_pose(pose_iso),
                fmt_raw(R_gt[0].cpu().numpy(), t_gt[0].cpu().numpy()),
                fmt_raw(R[0].numpy(), t[0].numpy()),
            ]
            if refined_img is not None:
                imgs.append(refined_img)
                captions.append(refined_caption)
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

    @torch.no_grad()
    def _log_map_panel(self, batch, stage):
        if "maps" not in batch:
            return
        drr         = batch["drr"]
        maps        = batch["maps"]   # [4, H, W]
        segs        = batch.get("segs")        # [4, H, W] or None
        chan_labels = list("abcd")
        refine_on   = self.refine_net is not None
        art_end     = batch.get("art_end", 3)  # artery channel slice end (cta=3; topbrain=N)

        # Batch all 4 channels through net1; render the predicted CTA in one call.
        maps_batch = maps.unsqueeze(1).float().to(self.device)  # [4, 1, H, W]
        view_label = batch["map_view_labels"].to(self.device).long()
        map_out, decode_view_label = self._net_forward_predicted_role(
            self.net, maps_batch, view_label
        )
        n = self.cfg.num_outputs
        R_p, t_p = self._decode_pose(
            map_out[:, :n], view_label=decode_view_label
        )  # [4, 3] each
        poses_p    = convert(R_p, t_p,
                             parameterization=self.cfg.parameterization,
                             convention=self.cfg.convention)
        drr.to(self.device)
        cta_projs_mc = drr(poses_p, mask_to_channels=True)  # [4, C, H, W]
        cta_projs    = cta_projs_mc.sum(dim=1)              # [4, H, W] full projection

        # Inference-time refinement on the REAL DSA path: net2 sees
        # [MAP | render-at-prediction] and corrects net1's MAP pose, same
        # convention as the synthetic path in _shared_step (pred ∘ δ⁻¹). Both
        # channels are min-max normalised to match net2's training-input scale.
        # Logging only (under @torch.no_grad); rendered in the same on-device window.
        dR = dt = cta_refined = refined_arts = None
        if refine_on:
            refine_view = refiner_view_index(
                self.refine_net, decode_view_label, maps_batch.shape[0]
            ).to(maps_batch.device)
            dR, dt         = self.refine_net(self._normalize(maps_batch),
                                             self._normalize(cta_projs.unsqueeze(1)),
                                             refine_view)
            refined_poses  = poses_p.compose(delta_to_pose(dR, dt).inverse())
            cta_refined_mc = drr(refined_poses, mask_to_channels=True)
            cta_refined    = cta_refined_mc.sum(dim=1)                # [4, H, W]
            refined_arts   = (cta_refined_mc[:, 1:art_end].sum(dim=1) > 0).cpu().numpy()
        drr.cpu()

        pred_arts = (cta_projs_mc[:, 1:art_end].sum(dim=1) > 0).cpu().numpy()  # [4, H, W] bool

        # ── Assemble rows dynamically ─────────────────────────────────────────
        # When refinement is on, add a "(refined)" CTA-projection row and (if seg
        # data is present) a second artery overlay so init-vs-refined alignment to
        # the DSA seg is directly comparable. Off ⇒ the original 2/3-row layout.
        # Labels are single-line (clean when rotated); the overlay colour key is a
        # footer (below) rather than crammed into the row label.
        rows = [("DSA input", "map"),
                ("CTA proj" + (" (init)" if refine_on else ""), "proj_init")]
        if refine_on:
            rows.append(("CTA proj (refined)", "proj_refined"))
        if segs is not None:
            rows.append(("Artery overlay" + (" (init)" if refine_on else ""), "overlay_init"))
            if refine_on:
                rows.append(("Artery overlay (refined)", "overlay_refined"))

        n_rows = len(rows)
        fig, axes = plt.subplots(n_rows, 4, figsize=(20, 5 * n_rows), squeeze=False)
        fig.suptitle(f"DSA MAP → predicted CTA{'  (+ refinement)' if refine_on else ''}  |  "
                     f"Epoch {self.current_epoch}", fontsize=13)
        fig.subplots_adjust(left=0.13, hspace=0.45)

        def _seg_bool(ch):
            return segs[ch].cpu().numpy() > 0.5

        for r, (_, kind) in enumerate(rows):
            for ch in range(4):
                ax = axes[r, ch]
                if kind == "map":
                    ax.imshow(_to_np(maps[ch]), cmap="gray")
                    ax.set_title(f"MAP [{chan_labels[ch]}]", fontsize=10)
                elif kind == "proj_init":
                    ax.imshow(_to_np(cta_projs[ch]), cmap="gray")
                    ax.set_title(f"R: {_to_np(R_p[ch]).round(3)}\n"
                                 f"t: {_to_np(t_p[ch]).round(1)} mm",
                                 fontsize=8, family="monospace")
                elif kind == "proj_refined":
                    ax.imshow(_to_np(cta_refined[ch]), cmap="gray")
                    ax.set_title(f"δR: {_to_np(dR[ch]).round(3)}\n"
                                 f"δt: {_to_np(dt[ch]).round(1)} mm",
                                 fontsize=8, family="monospace")
                elif kind == "overlay_init":
                    ax.imshow(seg_overlay_rgb(_to_np(maps[ch]), _seg_bool(ch), pred_arts[ch]))
                    ax.set_title(f"[{chan_labels[ch]}]", fontsize=9)
                elif kind == "overlay_refined":
                    ax.imshow(seg_overlay_rgb(_to_np(maps[ch]), _seg_bool(ch), refined_arts[ch]))
                    ax.set_title(f"[{chan_labels[ch]}]", fontsize=9)
                ax.axis("off")

        # Row labels via text artists (set_ylabel is hidden by ax.axis("off")).
        for r, (ylabel, _) in enumerate(rows):
            self._row_label(axes[r, 0], ylabel)

        # Overlay colour key — only meaningful when the overlay rows are present.
        if segs is not None:
            fig.text(0.5, 0.02,
                     "Artery overlay —  green: reference seg   ·   red: predicted artery   ·   yellow: overlap",
                     ha="center", va="bottom", fontsize=11, color="#57606a")

        self.logger.experiment.log(
            {f"{stage}/map_pose_images": wandb.Image(fig), "trainer/global_step": self.global_step}
        )
        plt.close(fig)
