import kornia.augmentation as K
import torch
import torch.nn.functional as F

try:
    from bilateral_filter_layer import BilateralFilter3d
    HAS_BILATERAL_LAYER = True
except ImportError:
    HAS_BILATERAL_LAYER = False


class DRRAugmentations:
    def __new__(
        cls,
        p=0.333,
        max_crop=10,
        clahe_clip=(1.0, 10.0),
        gamma_range=(0.7, 1.8),
        noise_std=0.01,
        bilateral_sigma_min: float = 1.0,
        bilateral_sigma_max: float = 15.0,
        use_plasma: bool = False,
        plasma_roughness=(0.1, 0.5),
        plasma_brightness_intensity=(-1.0, 1.0),
        plasma_p: float = 1.0,
        same_on_batch=False,
        transformation_matrix_mode="skip",
    ):
        ops = [
            K.RandomClahe(clip_limit=clahe_clip, p=p),
            K.RandomGamma(gamma=gamma_range, p=p),
            K.RandomBoxBlur(p=p),
            K.RandomGaussianNoise(std=noise_std, p=p),
            RandomBilateralFilter(sigma_min=bilateral_sigma_min, sigma_max=bilateral_sigma_max, p=p),
            K.RandomSharpness(p=p),
        ]
        # LXPose-style plasma fractal noise pair (intensity-only). Defaults
        # match https://github.com/fedefacente/LXPose/blob/main/LXPose/train.py
        # where both transforms are applied at p=1.0 at the end of the pipeline.
        if use_plasma:
            ops.extend([
                K.RandomPlasmaContrast(
                    roughness=tuple(plasma_roughness),
                    p=plasma_p,
                    keepdim=True,
                    same_on_batch=same_on_batch,
                ),
                K.RandomPlasmaBrightness(
                    roughness=tuple(plasma_roughness),
                    intensity=tuple(plasma_brightness_intensity),
                    p=plasma_p,
                    keepdim=True,
                    same_on_batch=same_on_batch,
                ),
            ])
        ops.extend([
            K.RandomErasing(p=p),
            RandomCenterCrop(p=p, maxcrop=max_crop),
        ])
        return K.AugmentationSequential(
            *ops,
            keepdim=True,
            same_on_batch=same_on_batch,
            transformation_matrix_mode=transformation_matrix_mode,
        )


class RandomCenterCrop(K.IntensityAugmentationBase2D):
    """Simulate beam collimation by zeroing out a border of random width."""

    def __init__(self, maxcrop: int, p: float = 0.5):
        super().__init__(p=p)
        self.maxcrop = maxcrop

    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, any],
        transform: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, _, H, W = input.shape
        crops = params["crop"].to(input.device).view(B, 1, 1)

        y = torch.arange(H, device=input.device).view(1, H, 1).expand(B, H, W)
        x = torch.arange(W, device=input.device).view(1, 1, W).expand(B, H, W)

        mask = (
            (y >= crops) & (y < H - crops) & (x >= crops) & (x < W - crops)
        ).unsqueeze(1)

        return torch.where(mask, input, torch.zeros_like(input))

    def generate_parameters(self, shape: tuple[int, ...]) -> dict[str, torch.Tensor]:
        B = shape[0]
        return {"crop": torch.randint(0, self.maxcrop + 1, (B,), device=self.device)}


class RandomBilateralFilter(K.IntensityAugmentationBase2D):
    """Apply bilateral filtering with randomly sampled spatial and color sigmas."""

    def __init__(self, sigma_min: float = 1.0, sigma_max: float = 15.0, p: float = 0.5):
        super().__init__(p=p)
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def apply_transform(
        self,
        input: torch.Tensor,
        params: dict[str, torch.Tensor],
        flags: dict[str, any],
        transform: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not HAS_BILATERAL_LAYER:
            return input
        B, C, H, W = input.shape
        out = []
        for i in range(B):
            sigma_s = params["sigma_spatial"][i].item()
            sigma_c = params["sigma_color"][i].item()
            layer = BilateralFilter3d(
                1.0, sigma_s, sigma_s, sigma_c,
                use_gpu=input.device.type == "cuda",
            ).to(input.device)
            filtered_channels = []
            with torch.no_grad():
                for c in range(C):
                    ch = input[i, c].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, H, W]
                    filtered_channels.append(layer(ch)[0, 0, 0])  # [H, W]
            out.append(torch.stack(filtered_channels))  # [C, H, W]
        return torch.stack(out)  # [B, C, H, W]

    def generate_parameters(self, shape: tuple[int, ...]) -> dict[str, torch.Tensor]:
        B = shape[0]
        return {
            "sigma_spatial": torch.empty(B).uniform_(self.sigma_min, self.sigma_max),
            "sigma_color": torch.empty(B).uniform_(self.sigma_min / 40, self.sigma_max / 40),  # normalized image range [0, 1]
        }


class MAPAugmentation:
    """Augmentation pipeline for DSA MAP channels.

    Training: bilateral filter with randomly sampled sigma_spatial, sigma_color ∈ [1, 15],
              followed by per-channel random affine (rotation ±10°, translation ±50%, scale 0.9–1.1).
    Validation: bilateral filter with fixed sigma_spatial = sigma_color = 7, no affine.
    """

    def __init__(
        self,
        training: bool,
        sigma_min: int = 1,
        sigma_max: int = 15,
        val_sigma: float = 7.0,
        max_angle_deg: float = 10.0,
        max_translate: float = 0.5,
        scale_min: float = 0.9,
        scale_max: float = 1.1,
        apply_affine: bool = True,
    ):
        self.training = training
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.val_sigma = val_sigma
        self.max_angle_deg = max_angle_deg
        self.max_translate = max_translate
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.apply_affine = apply_affine

    def __call__(
        self, maps: torch.Tensor, segs: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            maps: [C, H, W] float tensor.
            segs: [C, H, W] binary float tensor (optional). When provided,
                  the same geometric transform is applied and a (maps, segs)
                  tuple is returned instead of maps alone.
        Returns:
            Augmented maps [C, H, W], or (maps, segs) when segs is given.
        """
        maps = self._apply_bilateral(maps)
        maps = self._normalize(maps)
        if self.training and self.apply_affine:
            maps, segs = self._apply_affine(maps, segs)
        if segs is not None:
            return maps, segs
        return maps

    @staticmethod
    def _normalize(maps: torch.Tensor) -> torch.Tensor:
        """Per-channel min-max normalization to [0, 1]."""
        C = maps.shape[0]
        out = []
        for c in range(C):
            ch = maps[c]
            out.append((ch - ch.min()) / (ch.max() - ch.min() + 1e-8))
        return torch.stack(out)

    def _apply_bilateral(self, maps: torch.Tensor) -> torch.Tensor:
        if not HAS_BILATERAL_LAYER:
            return maps
        if self.training:
            sigma_spatial = float(torch.randint(self.sigma_min, self.sigma_max + 1, (1,)).item())
            sigma_color = float(torch.randint(self.sigma_min, self.sigma_max + 1, (1,)).item())
        else:
            sigma_spatial = sigma_color = self.val_sigma
        layer = BilateralFilter3d(
            1.0, sigma_spatial, sigma_spatial, sigma_color,
            use_gpu=maps.device.type == "cuda",
        ).to(maps.device)
        C, H, W = maps.shape
        out = []
        with torch.no_grad():
            for c in range(C):
                img_t = maps[c].unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, H, W]
                out.append(layer(img_t)[0, 0, 0])  # [H, W]
        return torch.stack(out)  # [C, H, W]

    def _apply_affine(
        self, maps: torch.Tensor, segs: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        C, H, W = maps.shape
        max_angle = self.max_angle_deg * torch.pi / 180.0
        out_maps = []
        out_segs = [] if segs is not None else None
        for c in range(C):
            angle = (torch.rand(1).item() * 2 - 1) * max_angle
            tx = (torch.rand(1).item() * 2 - 1) * self.max_translate
            ty = (torch.rand(1).item() * 2 - 1) * self.max_translate
            scale = self.scale_min + torch.rand(1).item() * (self.scale_max - self.scale_min)
            cos_a = scale * torch.cos(torch.tensor(angle))
            sin_a = scale * torch.sin(torch.tensor(angle))
            theta = torch.tensor(
                [[cos_a, -sin_a, tx],
                 [sin_a,  cos_a, ty]],
                dtype=torch.float32, device=maps.device,
            ).unsqueeze(0)  # [1, 2, 3]
            grid = F.affine_grid(theta, (1, 1, H, W), align_corners=False)
            ch = F.grid_sample(
                maps[c:c+1].unsqueeze(0), grid,
                mode="bilinear", padding_mode="zeros", align_corners=False,
            )  # [1, 1, H, W]
            out_maps.append(ch[0, 0])
            if segs is not None:
                seg_ch = F.grid_sample(
                    segs[c:c+1].unsqueeze(0), grid,
                    mode="nearest", padding_mode="zeros", align_corners=False,
                )  # [1, 1, H, W]
                out_segs.append(seg_ch[0, 0])
        segs_out = torch.stack(out_segs) if out_segs is not None else None
        return torch.stack(out_maps), segs_out  # [C, H, W], [C, H, W] | None
