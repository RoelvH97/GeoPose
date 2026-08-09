"""Warm-started single-view refiner architectures.

The legacy :class:`RefineResNetPose` remains untouched so historical
checkpoints keep their exact module tree. New experiments opt into one of
three architectures through :func:`build_refine_pose_net`:

``early_fusion``
    The existing two-channel ResNet, with GeoPose's one-channel ``conv1``
    repeated into both input channels and divided by two, preserving the
    pretrained response when the two inputs are identical.
``pooled_late_fusion``
    A shared one-channel encoder followed by pooled comparison features.
``spatial_late_fusion``
    A shared one-channel encoder with layer2/3/4 spatial comparison before
    global pooling.

Every new arm uses the signed acquisition-view convention
``0=LAT-, 1=PA, 2=LAT+`` and transfers encoder tensors only. Fusion modules,
the three-row view embedding, and the near-zero delta head start fresh.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf

from .blocks import build_resnet_backbone
from .refine_resnet_pose import RefineResNetPose, _as3


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


def require_signed_view_label(view_label: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Enforce the public three-way refiner conditioning contract."""
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


def refiner_view_index(net: nn.Module, view_label: torch.Tensor, batch_size: int):
    """Use signed labels for new nets and an explicit binary map for old nets."""
    signed = require_signed_view_label(view_label, batch_size)
    classes = int(getattr(net, "view_classes", 3))
    if classes == 3:
        return signed
    if classes == 2:
        return (signed != 1).long()
    raise ValueError(f"Unsupported refiner view_classes={classes}; expected 2 or 3")


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
    conv1_mode: str,
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
        if key == "conv1.weight" and value.shape != target_value.shape:
            if (
                conv1_mode == "repeat"
                and value.ndim == 4
                and value.shape[1] == 1
                and target_value.shape[1] == 2
                and value.shape[0] == target_value.shape[0]
                and value.shape[2:] == target_value.shape[2:]
            ):
                value = value.repeat(1, 2, 1, 1).div(2.0)
            else:
                raise ValueError(
                    f"Cannot adapt conv1 {tuple(value.shape)} -> {tuple(target_value.shape)} "
                    f"with conv1_mode={conv1_mode!r}"
                )
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
        f"({len(adapted)} tensors, conv1_mode={conv1_mode})"
    )
    return ckpt


def _build_refiner_backbone(cfg: DictConfig, in_channels: int, conv1_mode: str):
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
            conv1_mode=conv1_mode,
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
    module.delta_R_scale = torch.tensor(_as3(cfg.get("delta_R_scale", 0.1)))
    module.delta_t_scale = torch.tensor(_as3(cfg.get("delta_t_scale", 20.0)))


class WarmStartEarlyFusionRefinePose(RefineResNetPose):
    """Legacy two-channel architecture with audited GeoPose initialization."""

    def __init__(self, cfg: DictConfig):
        init_ckpt = cfg.get("init_encoder_ckpt", None)
        build_cfg = OmegaConf.merge(
            cfg,
            {"pretrained": bool(cfg.pretrained) and not bool(init_ckpt)},
        )
        super().__init__(build_cfg)
        self.cfg = cfg
        if int(cfg.get("view_classes", 3)) != 3 or int(cfg.get("view_emb_dim", 16)) <= 0:
            raise ValueError(
                "Warm-started early_fusion requires a non-empty three-way view embedding"
            )
        if init_ckpt:
            mode = str(cfg.get("conv1_init", "repeat"))
            load_geopose_encoder(
                self.backbone,
                init_ckpt,
                backbone_name=str(cfg.backbone),
                conv1_mode=mode,
            )


class PooledLateFusionRefinePose(nn.Module):
    """Shared one-channel encoder with pooled feature-comparison fusion."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone, in_features = _build_refiner_backbone(
            cfg, in_channels=1, conv1_mode="identity"
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


def _layer_channels(layer: nn.Sequential) -> int:
    block = layer[-1]
    conv = getattr(block, "conv3", None)
    if conv is None:
        conv = block.conv2
    return int(conv.out_channels)


class SpatialLateFusionRefinePose(nn.Module):
    """Shared encoder with multiscale spatial comparison before global pooling."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg
        self.backbone, in_features = _build_refiner_backbone(
            cfg, in_channels=1, conv1_mode="identity"
        )
        stages = tuple(int(s) for s in cfg.get("spatial_stages", [2, 3, 4]))
        if not stages or any(s not in (1, 2, 3, 4) for s in stages):
            raise ValueError(f"spatial_stages must be a non-empty subset of 1..4, got {stages}")
        self.spatial_stages = stages
        proj_dim = int(cfg.get("spatial_proj_dim", 128))
        if proj_dim <= 0:
            raise ValueError("spatial_proj_dim must be positive")
        channels = {
            i: _layer_channels(getattr(self.backbone, f"layer{i}")) for i in stages
        }
        self.stage_projections = nn.ModuleDict({
            str(i): nn.Sequential(
                nn.Conv2d(5 * channels[i], proj_dim, kernel_size=1),
                nn.ReLU(inplace=True),
            )
            for i in stages
        })
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(
                len(stages) * proj_dim,
                in_features,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_features),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        _init_common_head(self, cfg, in_features)

    def _stages(self, x: torch.Tensor) -> dict[int, torch.Tensor]:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        out = {}
        for i in range(1, 5):
            x = getattr(self.backbone, f"layer{i}")(x)
            if i in self.spatial_stages:
                out[i] = x
        return out

    @staticmethod
    def _compare(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([a, b, a - b, (a - b).abs(), a * b], dim=1)

    def forward(self, map_img, drr_noisy, view_label):
        if map_img.shape != drr_noisy.shape:
            raise ValueError(
                f"MAP/render shapes must match, got {map_img.shape} and {drr_noisy.shape}"
            )
        B = map_img.shape[0]
        maps = self._stages(torch.cat([map_img, drr_noisy], dim=0))
        target_size = maps[self.spatial_stages[0]].shape[-2:]
        projected = []
        for stage in self.spatial_stages:
            a, b = maps[stage][:B], maps[stage][B:]
            z = self.stage_projections[str(stage)](self._compare(a, b))
            if z.shape[-2:] != target_size:
                z = F.interpolate(z, size=target_size, mode="bilinear", align_corners=False)
            projected.append(z)
        feat = self.pool(self.spatial_fusion(torch.cat(projected, dim=1))).flatten(1)
        view_e = _view_features(self.view_emb, view_label, B)
        return _scaled_delta(self, self.head(torch.cat([feat, view_e], dim=1)))


def build_refine_pose_net(cfg: DictConfig) -> nn.Module:
    """Construct a legacy or warm-started refiner from ``cfg.architecture``."""
    architecture = cfg.get("architecture", None)
    if architecture is None:
        # Historical checkpoints have no architecture field. Keep their exact
        # model tree and binary/three-way behavior on strict reload.
        return RefineResNetPose(cfg)
    factories = {
        "early_fusion": WarmStartEarlyFusionRefinePose,
        "pooled_late_fusion": PooledLateFusionRefinePose,
        "spatial_late_fusion": SpatialLateFusionRefinePose,
    }
    architecture = str(architecture)
    if architecture not in factories:
        raise ValueError(
            f"Unknown refiner architecture {architecture!r}; choose from {list(factories)}"
        )
    return factories[architecture](cfg)


def load_refine_pose_checkpoint(checkpoint: str | Path) -> nn.Module:
    """Rebuild a standalone refiner from its saved config and load ``net.*``."""
    ckpt, state, saved_cfg = _checkpoint_state(checkpoint)
    if saved_cfg is None:
        raise RuntimeError(f"Standalone refiner checkpoint {ckpt} has no saved cfg")
    try:
        model_cfg = OmegaConf.create(OmegaConf.to_container(saved_cfg.model, resolve=True))
    except (AttributeError, TypeError) as exc:
        raise RuntimeError(f"Checkpoint {ckpt} has no cfg.model refiner config") from exc
    # The checkpoint below is already the trained refiner. Avoid rebuilding its
    # ImageNet/GeoPose initialization as a side effect before loading final weights.
    model_cfg.pretrained = False
    if "init_encoder_ckpt" in model_cfg:
        model_cfg.init_encoder_ckpt = None
    net = build_refine_pose_net(model_cfg)
    net_state = {
        key[len("net."):]: value
        for key, value in state.items()
        if key.startswith("net.")
    }
    if not net_state:
        raise RuntimeError(f"No standalone refiner net.* tensors in {ckpt}")
    missing, unexpected = net.load_state_dict(net_state, strict=True)
    if missing or unexpected:  # strict=True normally raises; retain an explicit audit.
        raise RuntimeError(
            f"Refiner load from {ckpt} was not exact: missing={missing}, unexpected={unexpected}"
        )
    print(f"[refiner] loaded standalone {model_cfg.get('architecture', 'legacy')} from {ckpt}")
    return net
