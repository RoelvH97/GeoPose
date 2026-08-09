"""Image augmentation used by GeoPose training."""

import kornia.augmentation as K
import torch

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
                    ch = input[i, c].unsqueeze(0).unsqueeze(0).unsqueeze(0)
                    filtered_channels.append(layer(ch)[0, 0, 0])
            out.append(torch.stack(filtered_channels))
        return torch.stack(out)

    def generate_parameters(self, shape: tuple[int, ...]) -> dict[str, torch.Tensor]:
        B = shape[0]
        return {
            "sigma_spatial": torch.empty(B).uniform_(self.sigma_min, self.sigma_max),
            "sigma_color": torch.empty(B).uniform_(self.sigma_min / 40, self.sigma_max / 40),
        }
