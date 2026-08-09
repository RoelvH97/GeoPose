"""Refinement CNN: 2-channel ResNet predicting a 6-DOF pose delta.

Architecture
------------
* Input channels: DSA MAP (resampled to the GeoPose SDD) stacked with the
  noisy-pose DRR — both shape ``[B, 1, H, W]``, concatenated to ``[B, 2, H, W]``.
* Backbone: torchvision ResNet / ResNeXt / Wide-ResNet, configurable via
  ``cfg.backbone``. Same registry as :class:`geopose.models.resnet_pose.ResNetPose`.
  When ``cfg.pretrained`` is True, the pretrained 3-channel ``conv1`` weights
  are averaged into 1 channel and replicated across the two input channels;
  gradients differentiate them during training.
* View conditioning: a 2-row :class:`torch.nn.Embedding` indexed by
  ``lateral_mask`` (0 = PA, 1 = LAT), concatenated to the pooled backbone
  features before the regression head. Initialised to zeros so the network
  starts view-agnostic and learns the asymmetry that the
  :ref:`noise study <noise-study>` exposed.
* Output: 6 raw values interpreted as ``(δR [3], δt [3])`` in Euler ZYX
  (rad) + translation (mm). No centring offset — the delta is centred at zero
  by construction (training samples zero-mean deltas).
* Output scaling: the raw head outputs are multiplied by
  ``cfg.delta_R_scale`` (default 0.1 rad) and ``cfg.delta_t_scale``
  (default 20 mm) so the regression target stays in O(1) for both rotation
  and translation. Mirrors the ``t = raw * 100`` trick in the existing
  :class:`ResNetPose._decode_pose` (geopose/models/resnet_pose.py): without
  it, the head would have to learn ~200× larger weights for translation
  outputs than for rotation outputs, hurting optimisation stability.
* Head init: small Gaussian weights + zero bias so the network's first
  prediction is a near-zero correction. Otherwise a randomly initialised head
  outputs huge ``δ`` at step 0 and the geodesic loss explodes on the first
  batch.

Forward signature
-----------------
::

    dR_pred, dt_pred = net(map_img, drr_noisy, lateral_mask)
    #   map_img:      [B, 1, H, W]
    #   drr_noisy:    [B, 1, H, W]
    #   lateral_mask: [B] bool
    #   dR_pred:      [B, 3]  rad
    #   dt_pred:      [B, 3]  mm
"""

from __future__ import annotations

import torch
import torch.nn as nn

from omegaconf import DictConfig
from .blocks import build_resnet_backbone


def _as3(v) -> list[float]:
    """Coerce a scalar or length-3 config value into a 3-list (per-axis scale)."""
    if isinstance(v, (int, float)):
        return [float(v)] * 3
    v = list(v)
    if len(v) == 1:
        return [float(v[0])] * 3
    if len(v) != 3:
        raise ValueError(f"expected a scalar or length-3 scale, got {v!r}")
    return [float(x) for x in v]


class RefineResNetPose(nn.Module):
    """2-channel ResNet that regresses a 6-DOF pose delta with view conditioning."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        # 2-channel backbone ([MAP | noisy-DRR]); drop the torchvision fc and
        # keep the pooled features (see geopose.models.blocks).
        net, in_features = build_resnet_backbone(cfg.backbone, 2, cfg.pretrained)
        net.fc = nn.Identity()
        self.backbone = net

        # ── View conditioning ───────────────────────────────────────────────
        # `view_classes` rows in the embedding. 2 (default): binary PA/LAT, indexed
        # by lateral_mask (0=PA, 1=LAT) — the original behaviour, kept for the
        # cascade/manifest refiners and existing checkpoints. 3: the GeoPose
        # `view_label` convention (0=LAT-, 1=PA, 2=LAT+), so the net can tell the
        # two lateral sides apart — a left-lateral is a L-R mirror of a right-lateral,
        # so collapsing them confounds the in-plane→depth mapping (see
        # docs/refinement.html, bi-plane open architectural note).
        view_emb_dim = int(cfg.get("view_emb_dim", 16))
        self.view_classes = int(cfg.get("view_classes", 2))
        self.view_emb = nn.Embedding(self.view_classes, view_emb_dim)
        nn.init.zeros_(self.view_emb.weight)

        # ── Residual regression head ────────────────────────────────────────
        self.head = nn.Linear(in_features + view_emb_dim, 6)
        nn.init.normal_(self.head.weight, std=1e-4)
        nn.init.zeros_(self.head.bias)

        # ── Output scaling ──────────────────────────────────────────────────
        # Make the head's raw outputs O(1) by multiplying through fixed scales
        # (≈ typical δ magnitudes). Accepts a scalar (broadcast to all 3 axes) or
        # a per-axis length-3 list — net1's translation error is strongly
        # anisotropic (t_y ≫ t_x,t_z), so a per-axis t-scale keeps each axis's
        # effective learning rate matched. Stored as plain tensors (NOT buffers)
        # so they stay out of the state_dict — the loadability invariant holds and
        # existing refine checkpoints still load strict.
        self.delta_R_scale = torch.tensor(_as3(cfg.get("delta_R_scale", 0.1)))   # [3] rad
        self.delta_t_scale = torch.tensor(_as3(cfg.get("delta_t_scale", 20.0)))  # [3] mm

    def forward(
        self,
        map_img: torch.Tensor,
        drr_noisy: torch.Tensor,
        view_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict ``(dR, dt)`` from a paired MAP + noisy-DRR batch.

        Args:
            map_img:    ``[B, 1, H, W]`` DSA MAP, intensity-normalised.
            drr_noisy:  ``[B, 1, H, W]`` noisy-pose DRR, intensity-normalised.
            view_idx:   ``[B]`` view index for the embedding. Binary mode
                        (``view_classes=2``): a bool ``lateral_mask`` (False=PA,
                        True=LAT). 3-way mode (``view_classes=3``): the GeoPose
                        ``view_label`` in {0=LAT-, 1=PA, 2=LAT+}.

        Returns:
            ``(dR_pred, dt_pred)`` each ``[B, 3]``: Euler ZYX (rad) / mm.
        """
        x = torch.cat([map_img, drr_noisy], dim=1)   # [B, 2, H, W]
        feat = self.backbone(x)                      # [B, in_features]
        view_e = self.view_emb(view_idx.long())      # [B, view_emb_dim]
        h = torch.cat([feat, view_e], dim=1)
        raw = self.head(h)                           # [B, 6]
        dR = raw[:, :3] * self.delta_R_scale.to(raw.device)   # [3] broadcast, rad
        dt = raw[:, 3:6] * self.delta_t_scale.to(raw.device)  # [3] broadcast, mm
        return dR, dt


def _smoke_test() -> None:
    """Quick forward-pass check: ``python -m geopose.models.refine_resnet_pose``.

    Verifies the model builds with the smallest backbone (no pretrained
    weights, so no network access needed), forward returns the right shapes,
    and the view embedding actually moves the output when the mask flips.
    Initial predictions should be tiny (small-init head + zero-init view
    embedding) so the noisy pose isn't blown apart on step 0.
    """
    from omegaconf import OmegaConf

    torch.manual_seed(0)
    cfg = OmegaConf.create({"backbone": "resnet18", "pretrained": False, "view_emb_dim": 16})
    net = RefineResNetPose(cfg).eval()

    B, H, W = 4, 256, 256
    map_img = torch.rand(B, 1, H, W)
    drr_noisy = torch.rand(B, 1, H, W)
    lateral_mask_pa = torch.zeros(B, dtype=torch.bool)
    lateral_mask_lat = torch.ones(B, dtype=torch.bool)

    with torch.no_grad():
        dR_pa, dt_pa = net(map_img, drr_noisy, lateral_mask_pa)
        dR_lat, dt_lat = net(map_img, drr_noisy, lateral_mask_lat)

    assert dR_pa.shape == (B, 3), f"dR shape {dR_pa.shape}"
    assert dt_pa.shape == (B, 3), f"dt shape {dt_pa.shape}"

    # With the small-init head and output scaling, expected init magnitudes are
    # ~1e-3 rad and ~1e-1 mm (well below any plausible δ → corrected ≈ noisy).
    init_R_max = max(dR_pa.abs().max().item(), dR_lat.abs().max().item())
    init_t_max = max(dt_pa.abs().max().item(), dt_lat.abs().max().item())
    assert init_R_max < 1e-2, f"initial dR too large: {init_R_max}"
    assert init_t_max < 1.0,  f"initial dt too large: {init_t_max}"

    # View embedding is zero-init, so PA and LAT outputs should match exactly.
    assert torch.allclose(dR_pa, dR_lat), "PA vs LAT differ pre-training (view_emb not zero?)"

    # After perturbing the view embedding, the outputs should diverge.
    with torch.no_grad():
        net.view_emb.weight.data[1].fill_(1.0)
        dR_lat2, _ = net(map_img, drr_noisy, lateral_mask_lat)
    assert not torch.allclose(dR_pa, dR_lat2), "view_emb perturbation had no effect"

    n_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("refine_resnet_pose smoke test OK")
    print(f"  backbone: {cfg.backbone}  trainable params: {n_params/1e6:.2f}M")
    print(f"  output shapes: dR={tuple(dR_pa.shape)}  dt={tuple(dt_pa.shape)}")
    print(f"  initial |dR|_max = {init_R_max:.2e} rad  (expect << 0.1 = typical δR)")
    print(f"  initial |dt|_max = {init_t_max:.2e} mm   (expect << 20  = typical δt)")
    print(f"  output scales:   R_scale={net.delta_R_scale}  t_scale={net.delta_t_scale}")


if __name__ == "__main__":
    _smoke_test()
