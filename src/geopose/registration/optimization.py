"""GeoReg DSA input preparation and 25-step test-time optimization."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from diffdrr.drr import DRR
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
from monai.losses import GeneralizedDiceLoss
from skimage.transform import resize

from .geometry import tensor_list
from .images import largest_component, minmax
from .projections import ProjectionInput
from .views import DSA_PATH_CHANNEL_OVERRIDES, VIEW_CHANNELS

try:
    from bilateral_filter_layer import BilateralFilter3d
except ImportError as exc:
    raise ImportError(
        "Faithful GeoPose inference requires bilateralfilter_torch==1.1.0"
    ) from exc


def _dsa_paths(
    data_root: Path, patient: str, timestamp: str
) -> dict[str, dict[str, Path]]:
    channels = VIEW_CHANNELS[timestamp]
    override = DSA_PATH_CHANNEL_OVERRIDES.get((patient, timestamp), {})
    dsa_channels = override.get("dsa", channels)
    mask_channels = override.get("mask", dsa_channels)
    metadata_channels = override.get("meta", dsa_channels)
    return {
        view: {
            "dsa": data_root / "DSATr" / f"{patient}_{dsa_channels[view]}_0000.nii.gz",
            "mask": data_root / "MAP_maskTr" / f"{patient}_{mask_channels[view]}.nii.gz",
            "metadata": data_root / "DSA_arteriesTr" / f"{patient}_{metadata_channels[view]}.json",
        }
        for view in ("lat", "pa")
    }


def _sitk_array(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path)))


def read_registration_channels(
    data_root: Path,
    patient: str,
    timestamp: str,
    *,
    size: int = 256,
) -> tuple[dict, dict, dict]:
    """Read and downsample the two MAP targets and DSA cranium masks."""
    paths = _dsa_paths(data_root, patient, timestamp)
    images, cranium_masks, metadata = {}, {}, {}
    for view, view_paths in paths.items():
        for path in view_paths.values():
            if not path.is_file():
                raise FileNotFoundError(path)
        image = np.max(_sitk_array(view_paths["dsa"]), axis=0)
        valid_columns = np.sum(image, axis=0) != 0
        valid = image[:, valid_columns] if valid_columns.any() else image
        low, high = np.percentile(valid, [25, 75])
        if low == high:
            high += 40
        image = np.clip(image, low, high)
        image -= image.min()
        mask = largest_component(_sitk_array(view_paths["mask"])[0])
        if image.shape != (size, size):
            image = resize(
                image, (size, size), preserve_range=True, anti_aliasing=True
            )
            mask = resize(
                mask, (size, size), preserve_range=True, anti_aliasing=False
            )
        images[view] = np.asarray(image, dtype=np.float32)
        cranium_masks[view] = np.asarray(mask, dtype=np.float32)
        with view_paths["metadata"].open() as stream:
            metadata[view] = json.load(stream)
    return images, cranium_masks, metadata


def prepare_registration_inputs(
    data_root: Path,
    patient: str,
    timestamp: str,
    device: torch.device,
    *,
    size: int = 256,
    projections: dict[str, ProjectionInput] | None = None,
) -> tuple[dict, dict, dict]:
    """Prepare images and cranium masks from DSA data or frozen projections."""
    if projections is None:
        arrays, mask_arrays, metadata = read_registration_channels(
            data_root, patient, timestamp, size=size
        )
    else:
        arrays = {
            view: projection.registration_image
            for view, projection in projections.items()
        }
        mask_arrays = {
            view: projection.registration_mask
            for view, projection in projections.items()
        }
        metadata = {
            view: {
                "alpha": projection.alpha_degrees,
                "d_source_to_detector": projection.source_sdd,
            }
            for view, projection in projections.items()
        }

    images, cranium_masks = {}, {}
    bilateral = BilateralFilter3d(
        1.0, 11.0, 11.0, 11.0, use_gpu=device.type == "cuda"
    ).to(device)
    for view in ("lat", "pa"):
        image = torch.tensor(arrays[view], device=device)[None, None, None]
        mask = torch.tensor(mask_arrays[view], device=device)[None, None, None]
        with torch.no_grad():
            images[view] = bilateral(image * mask)[:, :, 0]
        cranium_masks[view] = mask[:, :, 0]
    return images, cranium_masks, metadata


class TestTimeOptimizer(nn.Module):
    """The matched publication NCC + cranium-Dice GeoReg objective."""

    def __init__(
        self,
        cta_subject,
        images: dict[str, torch.Tensor],
        cranium_masks: dict[str, torch.Tensor],
        metadata: dict,
        initial_poses: dict[str, tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
        *,
        size: int = 256,
        spacing: float = 1.2,
        multiplier: float = 200.0,
    ) -> None:
        super().__init__()
        self.images = images
        self.cranium_masks = cranium_masks
        self.device = device
        self.multiplier = multiplier
        self.renderers = {
            view: DRR(
                cta_subject,
                sdd=float(metadata[view]["d_source_to_detector"]),
                height=size,
                delx=spacing,
                stop_gradients_through_grid_sample=True,
            ).to(device)
            for view in ("lat", "pa")
        }
        self.rotations = nn.ParameterDict()
        self.translations_scaled = nn.ParameterDict()
        for view in ("lat", "pa"):
            rotation, translation = initial_poses[view]
            self.rotations[view] = nn.Parameter(rotation.detach().clone().float())
            self.translations_scaled[view] = nn.Parameter(
                translation.detach().clone().float() / multiplier
            )
        self.ncc = MultiscaleNormalizedCrossCorrelation2d(
            patch_sizes=[None, 13], patch_weights=[0.5, 0.5]
        ).to(device)
        self.dice = GeneralizedDiceLoss()

    def pose(self, view: str) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            self.rotations[view],
            self.translations_scaled[view] * self.multiplier,
        )

    def render(self, view: str) -> torch.Tensor:
        rotation, translation = self.pose(view)
        return self.renderers[view](
            rotation,
            translation,
            parameterization="euler_angles",
            convention="ZYX",
            mask_to_channels=True,
        )

    def losses(self) -> tuple[dict, dict, torch.Tensor]:
        ncc_losses, cranium_dice_losses = {}, {}
        total = torch.zeros((), device=self.device)
        for view in ("lat", "pa"):
            estimate = self.render(view)
            ncc_loss = -self.ncc(
                self.images[view], estimate.sum(dim=1, keepdim=True)
            )
            # The second rendered CTA channel and target mask are cranium
            # silhouettes. This is deliberately the frozen GeoReg objective:
            # no carotid or other vessel annotation is used at test time.
            cranium_dice_loss = self.dice(
                torch.sigmoid(estimate[:, 1:2]),
                (self.cranium_masks[view] > 0).float(),
            )
            ncc_losses[view] = ncc_loss
            cranium_dice_losses[view] = cranium_dice_loss
            total = total + 0.5 * ncc_loss + 0.5 * cranium_dice_loss
        return ncc_losses, cranium_dice_losses, total

    @torch.no_grad()
    def snapshot(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return {
            view: tuple(value.detach().clone() for value in self.pose(view))
            for view in ("lat", "pa")
        }

    def optimize(self, iterations: int = 25) -> tuple[dict, dict]:
        """Run the 25-step GeoReg objective, returning the best pose per view.

        Gradient steps minimize ``0.5 * NCC + 0.5 * cranium Dice`` jointly over
        both views, but the returned pose is the one with the best NCC *per
        view*, tracked independently. This matches the published GeoReg
        protocol: cranium Dice regularizes the trajectory while photometric
        agreement selects the answer, and the two views are free to peak at
        different iterations.
        """
        with torch.no_grad():
            initial_ncc, initial_dice, _ = self.losses()
        best_loss = {view: float(initial_ncc[view]) for view in ("lat", "pa")}
        best_pose = self.snapshot()
        trace = {
            "optimizer": "NAdam",
            "learning_rate": 1e-4,
            "scheduler": "OneCycleLR",
            "max_learning_rate": 1e-2,
            "pct_start": 0.3,
            "iterations": iterations,
            "objective": "0.5 * multiscale NCC + 0.5 * cranium Dice per view",
            "mask_semantics": "binary DSA/CTA cranium silhouette; no vessel segmentation",
            "steps": [
                self._trace_step(0, 1e-4, initial_ncc, initial_dice, best_loss)
            ],
        }
        if iterations == 0:
            return best_pose, trace

        optimizer = torch.optim.NAdam(self.parameters(), lr=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=1e-2, total_steps=iterations, pct_start=0.3
        )
        for step in range(1, iterations + 1):
            optimizer.zero_grad(set_to_none=True)
            _, _, total = self.losses()
            total.backward()
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                current_ncc, current_dice, _ = self.losses()
                for view in ("lat", "pa"):
                    value = float(current_ncc[view])
                    if value < best_loss[view]:
                        best_loss[view] = value
                        best_pose[view] = tuple(
                            tensor.detach().clone() for tensor in self.pose(view)
                        )
                trace["steps"].append(
                    self._trace_step(
                        step,
                        optimizer.param_groups[0]["lr"],
                        current_ncc,
                        current_dice,
                        best_loss,
                    )
                )
        return best_pose, trace

    def _trace_step(self, step, learning_rate, ncc, dice, best):
        return {
            "step": step,
            "learning_rate": learning_rate,
            "views": {
                view: {
                    "mncc": -float(ncc[view]),
                    "cranium_dice_loss": float(dice[view]),
                    "best_mncc": -best[view],
                    "rotation": tensor_list(self.pose(view)[0]),
                    "translation": tensor_list(self.pose(view)[1]),
                }
                for view in ("lat", "pa")
            },
        }


def save_final_renders(
    output_dir: Path,
    optimizer: TestTimeOptimizer,
    poses: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        for view, (rotation, translation) in poses.items():
            estimate = optimizer.renderers[view](
                rotation,
                translation,
                parameterization="euler_angles",
                convention="ZYX",
                mask_to_channels=True,
            ).sum(dim=1, keepdim=True)
            target = optimizer.images[view]
            pair = torch.cat([minmax(target), minmax(estimate)], dim=3)
            image = (pair[0, 0].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            cv2.imwrite(str(output_dir / f"{view}_dsa_render.png"), image)
