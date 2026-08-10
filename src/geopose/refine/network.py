"""Warm-started single-view refiner architecture."""

from __future__ import annotations

from pathlib import Path

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


def _view_features(
    embedding: nn.Embedding,
    view_label: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Validate and embed signed view labels for a refiner batch."""
    if view_label is None:
        raise ValueError("view_label is required for refiner view conditioning")
    view_label = view_label.to(device=embedding.weight.device, dtype=torch.long)
    if view_label.ndim != 1 or view_label.shape[0] != batch_size:
        raise ValueError(
            f"Expected view_label [B] with B={batch_size}, got {tuple(view_label.shape)}"
        )
    n = embedding.num_embeddings
    if torch.any((view_label < 0) | (view_label >= n)):
        raise ValueError(f"view_label values must be in [0, {n - 1}]")
    return embedding(view_label)


def refiner_view_index(net: nn.Module, view_label: torch.Tensor, batch_size: int):
    """Validate the signed view labels {0=LAT-, 1=PA, 2=LAT+} the refiner expects."""
    if view_label is None:
        raise ValueError("Refiner batch is missing required signed view_label")
    view_label = view_label.long()
    if view_label.ndim != 1 or view_label.shape[0] != batch_size:
        raise ValueError(
            f"Expected signed view_label [B] with B={batch_size}, got {tuple(view_label.shape)}"
        )
    if torch.any((view_label < 0) | (view_label > 2)):
        raise ValueError("Signed view_label values must be 0=LAT-, 1=PA, or 2=LAT+")
    return view_label


def _checkpoint_state(path: str | Path) -> tuple[Path, dict, object | None]:
    ckpt = Path(path).expanduser()
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"model.init_encoder_ckpt must name an explicit .ckpt file, got {ckpt}"
        )
    blob = torch.load(ckpt, map_location="cpu", weights_only=False)
    state = blob.get("state_dict", blob)
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint {ckpt} has no state_dict mapping")
    saved_cfg = blob.get("hyper_parameters", {}).get("cfg")
    return ckpt, state, saved_cfg


def load_geopose_encoder(
    backbone: nn.Module,
    checkpoint: str | Path,
    *,
    backbone_name: str,
) -> Path:
    """Strictly load GeoPose's encoder, excluding every task-specific head."""
    ckpt, state, saved_cfg = _checkpoint_state(checkpoint)
    source_name = None
    if saved_cfg is not None:
        try:
            source_name = str(saved_cfg.get("backbone"))
        except AttributeError:
            source_name = None
    if source_name and source_name != "None" and source_name != str(backbone_name):
        raise ValueError(
            f"Encoder checkpoint uses {source_name}, target uses {backbone_name}"
        )

    allowed = ("conv1.", "bn1.", "layer1.", "layer2.", "layer3.", "layer4.")
    source = {
        key[len("net."):]: value
        for key, value in state.items()
        if key.startswith("net.") and key[len("net."):].startswith(allowed)
    }
    if not source:
        raise RuntimeError(f"No GeoPose encoder tensors (net.conv1/bn1/layer*) in {ckpt}")

    target = backbone.state_dict()
    adapted: dict[str, torch.Tensor] = {}
    missing: list[str] = []
    for key, target_value in target.items():
        if not key.startswith(allowed):
            continue
        value = source.get(key)
        if value is None:
            missing.append(key)
            continue
        if value.shape != target_value.shape:
            raise ValueError(
                f"Encoder tensor {key} shape mismatch: source {tuple(value.shape)}, "
                f"target {tuple(target_value.shape)}"
            )
        adapted[key] = value

    if missing:
        raise RuntimeError(
            f"Checkpoint {ckpt} is missing {len(missing)} encoder tensors; "
            f"first missing: {missing[:5]}"
        )
    unexpected = sorted(set(source) - set(adapted))
    if unexpected:
        raise RuntimeError(
            f"Checkpoint {ckpt} has unexpected encoder tensors; first: {unexpected[:5]}"
        )

    merged = dict(target)
    merged.update(adapted)
    backbone.load_state_dict(merged, strict=True)
    print(
        f"[refiner] initialized {backbone_name} encoder from {ckpt} "
        f"({len(adapted)} tensors)"
    )
    return ckpt


def _build_refiner_backbone(cfg: DictConfig, in_channels: int):
    init_ckpt = cfg.get("init_encoder_ckpt", None)
    net, in_features = build_resnet_backbone(
        cfg.backbone,
        in_channels,
        bool(cfg.pretrained) and not bool(init_ckpt),
    )
    net.fc = nn.Identity()
    if init_ckpt:
        load_geopose_encoder(
            net,
            init_ckpt,
            backbone_name=str(cfg.backbone),
        )
    return net, in_features


def _scaled_delta(module: nn.Module, raw: torch.Tensor):
    dR = raw[:, :3] * module.delta_R_scale.to(raw.device)
    dt = raw[:, 3:6] * module.delta_t_scale.to(raw.device)
    return dR, dt


def _init_common_head(module: nn.Module, cfg: DictConfig, in_features: int) -> None:
    view_classes = int(cfg.get("view_classes", 3))
    view_emb_dim = int(cfg.get("view_emb_dim", 16))
    if view_classes != 3:
        raise ValueError("Warm-started refiner arms require view_classes=3")
    if view_emb_dim <= 0:
        raise ValueError("Warm-started refiner arms require view_emb_dim > 0")
    module.view_classes = view_classes
    module.view_emb = nn.Embedding(3, view_emb_dim)
    nn.init.zeros_(module.view_emb.weight)
    module.head = nn.Linear(in_features + view_emb_dim, 6)
    nn.init.normal_(module.head.weight, std=1e-4)
    nn.init.zeros_(module.head.bias)
    # Plain tensors keep fixed scales out of checkpoint state dictionaries.
    module.delta_R_scale = torch.tensor(_as3(cfg.get("delta_R_scale", 0.1)))
    module.delta_t_scale = torch.tensor(_as3(cfg.get("delta_t_scale", 20.0)))


class PooledLateFusionRefinePose(nn.Module):
    """Shared one-channel encoder with pooled feature-comparison fusion.

    Both the target projection and the current render pass through the same
    encoder, warm-started from the GeoPose-Init backbone. Their pooled features
    are compared elementwise as ``[a, b, a-b, |a-b|, a*b]``, fused, concatenated
    with a signed view embedding, and mapped to a 6-DOF tangent-space delta.
    The head is initialized with std 1e-4 and the view embedding at exactly
    zero, so views are indistinguishable before training; note that the
    intervening ``fusion`` layer uses default initialization, so an untrained
    net does not predict the identity correction.
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone, in_features = _build_refiner_backbone(
            cfg, in_channels=1
        )
        self.fusion = nn.Sequential(
            nn.Linear(5 * in_features, in_features),
            nn.ReLU(inplace=True),
        )
        _init_common_head(self, cfg, in_features)

    @staticmethod
    def _compare(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b, a - b, (a - b).abs(), a * b], dim=1)

    def forward(self, map_img, drr_noisy, view_label):
        """Predict ``(dR [B,3] radians, dt [B,3] millimetres)`` for one view."""
        if map_img.shape != drr_noisy.shape:
            raise ValueError(
                f"MAP/render shapes must match, got {map_img.shape} and {drr_noisy.shape}"
            )
        B = map_img.shape[0]
        both = self.backbone(torch.cat([map_img, drr_noisy], dim=0))
        map_feat, render_feat = both[:B], both[B:]
        feat = self.fusion(self._compare(map_feat, render_feat))
        view_e = _view_features(self.view_emb, view_label, B)
        return _scaled_delta(self, self.head(torch.cat([feat, view_e], dim=1)))


def build_refine_pose_net(cfg: DictConfig) -> nn.Module:
    """Construct the published refiner named by ``cfg.architecture``."""
    architecture = str(cfg.get("architecture", "pooled_late_fusion"))
    if architecture != "pooled_late_fusion":
        raise ValueError(
            f"Unknown refiner architecture {architecture!r}; "
            "the published contract uses 'pooled_late_fusion'"
        )
    return PooledLateFusionRefinePose(cfg)
