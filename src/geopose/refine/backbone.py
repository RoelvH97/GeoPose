"""Refinement CNN: 2-channel ResNet predicting a 6-DOF pose delta."""

from __future__ import annotations

import torch
import torch.nn as nn

from omegaconf import DictConfig
from ..shared.blocks import build_resnet_backbone


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

        net, in_features = build_resnet_backbone(cfg.backbone, 2, cfg.pretrained)
        net.fc = nn.Identity()
        self.backbone = net

        view_emb_dim = int(cfg.get("view_emb_dim", 16))
        # New models use signed LAT-/PA/LAT+ labels; legacy models use PA/LAT.
        self.view_classes = int(cfg.get("view_classes", 2))
        self.view_emb = nn.Embedding(self.view_classes, view_emb_dim)
        nn.init.zeros_(self.view_emb.weight)

        self.head = nn.Linear(in_features + view_emb_dim, 6)
        nn.init.normal_(self.head.weight, std=1e-4)
        nn.init.zeros_(self.head.bias)

        # Plain tensors keep fixed scales out of checkpoint state dictionaries.
        self.delta_R_scale = torch.tensor(_as3(cfg.get("delta_R_scale", 0.1)))
        self.delta_t_scale = torch.tensor(_as3(cfg.get("delta_t_scale", 20.0)))

    def forward(
        self,
        map_img: torch.Tensor,
        drr_noisy: torch.Tensor,
        view_idx: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict ``(dR, dt)`` from a paired MAP + noisy-DRR batch."""
        x = torch.cat([map_img, drr_noisy], dim=1)
        feat = self.backbone(x)
        view_e = self.view_emb(view_idx.long())
        h = torch.cat([feat, view_e], dim=1)
        raw = self.head(h)
        dR = raw[:, :3] * self.delta_R_scale.to(raw.device)
        dt = raw[:, 3:6] * self.delta_t_scale.to(raw.device)
        return dR, dt


def _smoke_test() -> None:
    """Quick forward-pass check: ``python -m geopose.refine.backbone``."""
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

    init_R_max = max(dR_pa.abs().max().item(), dR_lat.abs().max().item())
    init_t_max = max(dt_pa.abs().max().item(), dt_lat.abs().max().item())
    assert init_R_max < 1e-2, f"initial dR too large: {init_R_max}"
    assert init_t_max < 1.0,  f"initial dt too large: {init_t_max}"

    assert torch.allclose(dR_pa, dR_lat), "PA vs LAT differ pre-training (view_emb not zero?)"

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
