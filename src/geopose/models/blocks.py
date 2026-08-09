# import necessary libraries
import torch
import torch.nn as nn
import torchvision.models as tv_models


# ── Backbone registry ────────────────────────────────────────────────────────
# Single source of truth for the torchvision backbones both ResNetPose and
# RefineResNetPose build on (was duplicated across both modules).
BACKBONES = {
    "resnet18":          (tv_models.resnet18,          tv_models.ResNet18_Weights.DEFAULT),
    "resnet34":          (tv_models.resnet34,          tv_models.ResNet34_Weights.DEFAULT),
    "resnet50":          (tv_models.resnet50,          tv_models.ResNet50_Weights.DEFAULT),
    "resnet101":         (tv_models.resnet101,         tv_models.ResNet101_Weights.DEFAULT),
    "resnet152":         (tv_models.resnet152,         tv_models.ResNet152_Weights.DEFAULT),
    "resnext50_32x4d":   (tv_models.resnext50_32x4d,   tv_models.ResNeXt50_32X4D_Weights.DEFAULT),
    "resnext101_32x8d":  (tv_models.resnext101_32x8d,  tv_models.ResNeXt101_32X8D_Weights.DEFAULT),
    "wide_resnet50_2":   (tv_models.wide_resnet50_2,   tv_models.Wide_ResNet50_2_Weights.DEFAULT),
    "wide_resnet101_2":  (tv_models.wide_resnet101_2,  tv_models.Wide_ResNet101_2_Weights.DEFAULT),
}


def build_resnet_backbone(name: str, in_channels: int, pretrained: bool):
    """Build a torchvision ResNet adapted to `in_channels` grayscale inputs.

    Replaces conv1 with an `in_channels`-channel conv and, when `pretrained`,
    seeds it by averaging the RGB weights across colour and repeating across the
    new channels (the 1→1 case reduces to a plain channel mean; the 2-channel
    refine net repeats that mean twice — both identical to the prior inline code).
    The torchvision `fc` is left in place so the caller can read `in_features`
    and attach its own head. Returns `(net, in_features)`.
    """
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone '{name}'. Choose from: {list(BACKBONES)}")
    factory, weights = BACKBONES[name]
    net = factory(weights=weights if pretrained else None)

    old = net.conv1
    net.conv1 = nn.Conv2d(
        in_channels, old.out_channels,
        kernel_size=old.kernel_size,
        stride=old.stride,
        padding=old.padding,
        bias=old.bias is not None,
    )
    if pretrained:
        with torch.no_grad():
            avg = old.weight.mean(dim=1, keepdim=True)        # [out, 1, k, k]
            net.conv1.weight.copy_(avg.repeat(1, in_channels, 1, 1))

    in_features = net.fc.in_features
    return net, in_features


# ── Gradient Reversal Layer ──────────────────────────────────────────────────

class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


class GradientReversalLayer(nn.Module):
    """Identity forward; negates and scales gradients in backward."""
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(x, self.alpha)


# ── Combined pose + domain head ──────────────────────────────────────────────

class _PoseDomainHead(nn.Module):
    """Drop-in replacement for ResNet fc.

    Output: [pose_params | domain_logit | view_logits]  shape [B, num_pose + 4]
    - domain_logit (1):  passes through GRL (DANN-style). Labels: DRR = 0, MAP = 1.
    - view_logits (3):   auxiliary 3-class view classifier.
        Labels: 0 = LAT(-π/2), 1 = PA, 2 = LAT(+π/2).
    """
    def __init__(
        self,
        in_features: int,
        num_pose_outputs: int,
        da_alpha: float,
        role_emb_dim: int = 0,
        role_classes: int = 2,
    ):
        super().__init__()
        if role_emb_dim < 0:
            raise ValueError(f"role_emb_dim must be non-negative, got {role_emb_dim}")
        if role_classes < 2:
            raise ValueError(f"role_classes must be >= 2, got {role_classes}")

        # Optional known-role conditioning for the pose branch only. Default
        # role_classes=2 is the binary convention matching DualViewPoseNet
        # (0=PA, 1=LAT) — same shapes/keys as before this option existed.
        # role_classes=3 instead conditions on the SIGNED class (0=LAT-, 1=PA,
        # 2=LAT+), matching the refiners' view_classes:3 and the decode
        # anchor's own convention (see ResNetPose._role_from_signed). Keeping
        # role_emb_dim=0 registers no embedding either way and preserves
        # historical GeoPose checkpoint keys and tensor shapes exactly.
        self.role_classes = role_classes
        self.role_embedding = (
            nn.Embedding(role_classes, role_emb_dim) if role_emb_dim > 0 else None
        )
        pose_in_features = in_features + role_emb_dim
        self.pose   = nn.Linear(pose_in_features, num_pose_outputs)
        self.grl    = GradientReversalLayer(da_alpha)
        self.domain = nn.Linear(in_features, 1)
        self.view   = nn.Linear(in_features, 3)

    def forward_aux(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(domain_logit [B,1], view_logits [B,3]) — independent of view_role.

        Split out from ``forward`` so callers that must resolve view_role FROM
        these view_logits (the real-DSA predicted-side path) can do so before
        running the role-conditioned pose branch, without a second backbone
        pass — see ``ResNetPose._net_forward_predicted_role``.
        """
        return self.domain(self.grl(x)), self.view(x)

    def forward_pose(
        self,
        x: torch.Tensor,
        view_role: torch.Tensor | None = None,
    ) -> torch.Tensor:
        pose_features = x
        if self.role_embedding is not None:
            if view_role is None:
                raise ValueError(
                    "view_role is required when the GeoPose role embedding is enabled"
                )
            view_role = view_role.to(device=x.device, dtype=torch.long)
            if view_role.ndim != 1 or view_role.shape[0] != x.shape[0]:
                raise ValueError(
                    f"Expected view_role [B] for features {tuple(x.shape)}, "
                    f"got {tuple(view_role.shape)}"
                )
            if torch.any((view_role < 0) | (view_role >= self.role_classes)):
                raise ValueError(
                    f"view_role values must be in [0, {self.role_classes})"
                )
            pose_features = torch.cat([x, self.role_embedding(view_role)], dim=1)
        return self.pose(pose_features)

    def forward(
        self,
        x: torch.Tensor,
        view_role: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Both phases in one call — used whenever view_role is already known
        before any forward pass (e.g. synthetic training's teacher-forced GT
        label), so there is no dependency to sequence around.

        Domain and signed-view predictions remain functions of the image
        alone; known-role metadata cannot leak into either auxiliary
        classifier.
        """
        pose = self.forward_pose(x, view_role)
        domain_logit, view_logits = self.forward_aux(x)
        return torch.cat([pose, domain_logit, view_logits], dim=1)
