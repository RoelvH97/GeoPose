"""Published GeoPose inference, calibration, refinement, and 25-step TTO."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
from diffdrr.pose import RigidTransform, convert
from diffdrr.utils import resample as resample_intrinsics
from monai.losses import GeneralizedDiceLoss
from omegaconf import DictConfig, OmegaConf
from pytorch3d.transforms import matrix_to_euler_angles
from scipy.ndimage import label
from skimage.transform import resize

from .models.pose_utils import delta_to_pose
from .models.refine_fusion_pose import build_refine_pose_net, refiner_view_index
from .models.resnet_pose import ResNetPose

try:
    from bilateral_filter_layer import BilateralFilter3d
except ImportError as exc:
    raise ImportError(
        "Faithful GeoPose inference requires bilateralfilter_torch==1.1.0"
    ) from exc


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPOSITORY_ROOT / "configs"
VIEW_CHANNELS = {
    "pre": {"lat": "a", "pa": "b"},
    "post": {"lat": "c", "pa": "d"},
}
CHANNEL_PAIR = {"a": "c", "c": "a", "b": "d", "d": "b"}
MAP_META_CHANNEL_OVERRIDES = {
    ("sub-stroke0016", "a"): "b",
    ("sub-stroke0016", "b"): "a",
    ("sub-stroke0020", "a"): "c",
    ("sub-stroke0020", "d"): "b",
}
DSA_PATH_CHANNEL_OVERRIDES = {
    ("sub-stroke0016", "pre"): {
        "dsa": {"lat": "b", "pa": "a"},
        "mask": {"lat": "a", "pa": "b"},
    },
    ("sub-stroke0020", "pre"): {"meta": {"lat": "c", "pa": "b"}},
    ("sub-stroke0020", "post"): {"meta": {"lat": "c", "pa": "b"}},
}
VIEW_THRESHOLD_DEGREES = 45.0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(path: Path, cfg: DictConfig, skip: bool) -> None:
    if skip:
        return
    expected_hash = str(cfg.release_contract.checkpoint_sha256)
    expected_size = int(cfg.release_contract.checkpoint_bytes)
    actual_size = path.stat().st_size
    if actual_size != expected_size:
        raise RuntimeError(
            f"Checkpoint size mismatch for {path}: {actual_size} != {expected_size}"
        )
    actual_hash = file_sha256(path)
    if actual_hash != expected_hash:
        raise RuntimeError(
            f"Checkpoint SHA-256 mismatch for {path}: {actual_hash} != {expected_hash}"
        )


def _checkpoint_state(path: Path) -> dict[str, torch.Tensor]:
    blob = torch.load(path, map_location="cpu", weights_only=False)
    state = blob.get("state_dict", blob)
    if not isinstance(state, dict):
        raise RuntimeError(f"Checkpoint has no state dictionary: {path}")
    return state


def load_init_model(path: Path, device: torch.device, skip_hash: bool = False) -> ResNetPose:
    contract = OmegaConf.load(CONFIG_DIR / "init.yaml")
    verify_checkpoint(path, contract, skip_hash)
    model_cfg = OmegaConf.create(OmegaConf.to_container(contract.model, resolve=True))
    # Avoid network access or random ImageNet initialization before strict loading.
    model_cfg.pretrained = False
    model_cfg.init_net_ckpt = None
    model_cfg.refine.enabled = False
    model = ResNetPose(model_cfg)
    model.load_state_dict(_checkpoint_state(path), strict=True)
    return model.to(device).eval()


def load_refine_model(path: Path, device: torch.device, skip_hash: bool = False) -> nn.Module:
    contract = OmegaConf.load(CONFIG_DIR / "refine.yaml")
    verify_checkpoint(path, contract, skip_hash)
    model_cfg = OmegaConf.create(OmegaConf.to_container(contract.model, resolve=True))
    model_cfg.pretrained = False
    model_cfg.init_encoder_ckpt = None
    model = build_refine_pose_net(model_cfg)
    state = {
        key[len("net."):]: value
        for key, value in _checkpoint_state(path).items()
        if key.startswith("net.")
    }
    if not state:
        raise RuntimeError(f"No net.* refiner tensors found in {path}")
    model.load_state_dict(state, strict=True)
    return model.to(device).eval()


def view_label_from_alpha(alpha_degrees: float) -> int:
    signed_rotation = -float(alpha_degrees)
    if signed_rotation > VIEW_THRESHOLD_DEGREES:
        return 2
    if signed_rotation < -VIEW_THRESHOLD_DEGREES:
        return 0
    return 1


def _resolve_metadata(metadata_dir: Path, patient: str, channel: str) -> Path:
    primary = metadata_dir / f"{patient}_{channel}.json"
    if primary.is_file():
        return primary
    alternate = metadata_dir / f"{patient}_{CHANNEL_PAIR[channel]}.json"
    if alternate.is_file():
        return alternate
    raise FileNotFoundError(f"No acquisition metadata for {patient}_{channel}")


def _map_metadata(data_root: Path, patient: str, channel: str) -> Path:
    channel = MAP_META_CHANNEL_OVERRIDES.get((patient, channel), channel)
    return _resolve_metadata(data_root / "DSA_arteriesTr", patient, channel)


def _largest_component(mask: np.ndarray) -> np.ndarray:
    components, count = label(mask > 0)
    if count == 0:
        return np.zeros_like(mask, dtype=np.float32)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return (components == sizes.argmax()).astype(np.float32)


def read_map_channel(
    data_root: Path,
    patient: str,
    channel: str,
    *,
    target_sdd: float = 1020.0,
    target_delx: float = 1.2,
    size: int = 256,
) -> tuple[np.ndarray, float, float]:
    """Apply the exact training/deployment MAP geometric preprocessing."""
    image_path = data_root / "MAPTr" / f"{patient}_{channel}_0000.nii.gz"
    mask_path = data_root / "MAP_maskTr" / f"{patient}_{channel}.nii.gz"
    for required in (image_path, mask_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    image = nib.load(str(image_path)).get_fdata(dtype=np.float32).squeeze()
    image = np.fliplr(np.rot90(image, 3))
    mask = nib.load(str(mask_path)).get_fdata(dtype=np.float32).squeeze()
    mask = _largest_component(np.fliplr(np.rot90(mask, 3)))

    valid_columns = np.sum(image, axis=0) != 0
    valid = image[:, valid_columns] if valid_columns.any() else image
    low, high = np.percentile(valid, [25, 75])
    if low == high:
        high += 40
    image = np.clip(image, low, high)
    image -= image.min()

    with _map_metadata(data_root, patient, channel).open() as stream:
        metadata = json.load(stream)
    source_sdd = float(metadata["d_source_to_detector"])
    alpha = float(metadata["alpha"])

    if source_sdd != target_sdd:
        stack = torch.from_numpy(
            np.ascontiguousarray(np.stack([image, mask]))
        ).float()[:, None]
        stack = resample_intrinsics(
            stack,
            source_sdd,
            target_delx,
            0,
            0,
            target_sdd,
            target_delx,
            0,
            0,
        )
        image = stack[0, 0].numpy()
        mask = (stack[1, 0].numpy() > 0.5).astype(np.float32)

    if image.shape != (size, size):
        image = resize(
            image, (size, size), preserve_range=True, anti_aliasing=True
        )
        mask = resize(
            mask, (size, size), preserve_range=True, anti_aliasing=False
        )
    return (image * mask).astype(np.float32), source_sdd, alpha


def bilateral_minmax(image: torch.Tensor, sigma: float = 11.0) -> torch.Tensor:
    layer = BilateralFilter3d(
        1.0,
        sigma,
        sigma,
        sigma,
        use_gpu=image.device.type == "cuda",
    ).to(image.device)
    with torch.no_grad():
        filtered = layer(image.unsqueeze(2))[:, :, 0]
    return minmax(filtered)


def minmax(image: torch.Tensor) -> torch.Tensor:
    flat = image.flatten(1)
    minimum = flat.min(1).values.view(-1, 1, 1, 1)
    maximum = flat.max(1).values.view(-1, 1, 1, 1)
    return (image - minimum) / (maximum - minimum).clamp(min=1e-8)


def pose_matrix(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    return convert(
        rotation,
        translation,
        parameterization="euler_angles",
        convention="ZYX",
    ).matrix


def matrix_to_pose(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rotation_matrix = matrix[:, :3, :3]
    rotation = matrix_to_euler_angles(rotation_matrix, convention="ZYX")
    translation = torch.einsum(
        "bij,bj->bi", rotation_matrix.transpose(1, 2), matrix[:, :3, 3]
    )
    return rotation, translation


def tensor_list(value: torch.Tensor) -> list:
    return value.detach().cpu().tolist()


def signed_forward(
    model: ResNetPose,
    image: torch.Tensor,
    metadata_label: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Run the final single-backbone signed-role deployment route."""
    full, decode_label = model._net_forward_predicted_role(
        model.net, image, metadata_label
    )
    count = int(model.cfg.num_outputs)
    rotation, translation = model._decode_pose(
        full[:, :count], view_label=decode_label
    )
    return rotation, translation, decode_label


class PoseInitializer:
    def __init__(
        self,
        init_model: ResNetPose,
        refine_model: nn.Module,
        cta_subject,
        data_root: Path,
        device: torch.device,
        *,
        size: int = 256,
        spacing: float = 1.2,
        target_sdd: float = 1020.0,
        iso_translation_y: float = 650.0,
        max_refine_updates: int = 5,
    ) -> None:
        self.init_model = init_model
        self.refine_model = refine_model
        self.data_root = data_root
        self.device = device
        self.size = size
        self.spacing = spacing
        self.target_sdd = target_sdd
        self.iso_translation_y = iso_translation_y
        self.max_refine_updates = max_refine_updates
        self.ncc = MultiscaleNormalizedCrossCorrelation2d(
            patch_sizes=[None, 13], patch_weights=[0.5, 0.5]
        ).to(device)
        self.renderer = DRR(
            cta_subject,
            sdd=target_sdd,
            height=size,
            delx=spacing,
            stop_gradients_through_grid_sample=True,
        ).to(device)
        self.iso_matrix, self.calibration_matrix = self._calibrate()

    @torch.inference_mode()
    def _calibrate(self) -> tuple[torch.Tensor, torch.Tensor]:
        rotation = torch.zeros(1, 3, device=self.device)
        translation = torch.tensor(
            [[0.0, self.iso_translation_y, 0.0]], device=self.device
        )
        iso_pose = convert(
            rotation,
            translation,
            parameterization="euler_angles",
            convention="ZYX",
        )
        image = self.renderer(iso_pose)
        if image.shape[1] > 1:
            image = image.sum(dim=1, keepdim=True)
        pa_label = torch.tensor([1], device=self.device)
        predicted_rotation, predicted_translation, decode_label = signed_forward(
            self.init_model, minmax(image), pa_label
        )
        if int(decode_label.item()) != 1:
            raise RuntimeError("Known PA calibration route did not preserve PA")
        return pose_matrix(rotation, translation), pose_matrix(
            predicted_rotation, predicted_translation
        )

    @torch.inference_mode()
    def predict(
        self, patient: str, timestamp: str
    ) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict]:
        output: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        trace = {
            "iso_matrix": tensor_list(self.iso_matrix),
            "calibration_matrix": tensor_list(self.calibration_matrix),
            "views": {},
        }
        calibration_inverse = torch.linalg.inv(self.calibration_matrix)
        for view, channel in VIEW_CHANNELS[timestamp].items():
            array, source_sdd, alpha = read_map_channel(
                self.data_root,
                patient,
                channel,
                target_sdd=self.target_sdd,
                target_delx=self.spacing,
                size=self.size,
            )
            image = torch.from_numpy(array).to(self.device)[None, None]
            image = bilateral_minmax(image)
            metadata_label = torch.tensor(
                [view_label_from_alpha(alpha)], device=self.device
            )
            rotation, translation, decode_label = signed_forward(
                self.init_model, image, metadata_label
            )
            predicted_matrix = pose_matrix(rotation, translation)
            current = self.iso_matrix @ calibration_inverse @ predicted_matrix
            view_trace = {
                "channel": channel,
                "source_sdd": source_sdd,
                "alpha_degrees": alpha,
                "metadata_label": int(metadata_label.item()),
                "decode_label": int(decode_label.item()),
                "network_pose_matrix": tensor_list(predicted_matrix),
                "calibrated_pose_matrix": tensor_list(current),
                "refinement": [],
            }

            target = minmax(image)
            rendered = self._render(current)
            best_ncc = float(self.ncc(target, rendered))
            refine_label = refiner_view_index(
                self.refine_model, decode_label, target.shape[0]
            ).to(self.device)
            for attempt in range(self.max_refine_updates):
                delta_rotation, delta_translation = self.refine_model(
                    target, rendered, refine_label
                )
                candidate = RigidTransform(current).compose(
                    delta_to_pose(delta_rotation, delta_translation).inverse()
                ).matrix
                candidate_render = self._render(candidate)
                candidate_ncc = float(self.ncc(target, candidate_render))
                accepted = candidate_ncc > best_ncc
                view_trace["refinement"].append(
                    {
                        "attempt": attempt + 1,
                        "delta_rotation": tensor_list(delta_rotation),
                        "delta_translation": tensor_list(delta_translation),
                        "candidate_matrix": tensor_list(candidate),
                        "mncc": candidate_ncc,
                        "accepted": accepted,
                    }
                )
                if not accepted:
                    break
                current, rendered, best_ncc = (
                    candidate,
                    candidate_render,
                    candidate_ncc,
                )

            rotation, translation = matrix_to_pose(current)
            output[view] = (rotation.detach(), translation.detach())
            view_trace["refined_pose_matrix"] = tensor_list(current)
            view_trace["refined_mncc"] = best_ncc
            trace["views"][view] = view_trace
        return output, trace

    def _render(self, matrix: torch.Tensor) -> torch.Tensor:
        image = self.renderer(RigidTransform(matrix))
        if image.shape[1] > 1:
            image = image.sum(dim=1, keepdim=True)
        return minmax(image)


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


def prepare_registration_inputs(
    data_root: Path,
    patient: str,
    timestamp: str,
    device: torch.device,
    *,
    size: int = 256,
) -> tuple[dict, dict, dict]:
    paths = _dsa_paths(data_root, patient, timestamp)
    images, masks, metadata = {}, {}, {}
    bilateral = BilateralFilter3d(
        1.0, 11.0, 11.0, 11.0, use_gpu=device.type == "cuda"
    ).to(device)
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
        mask = _largest_component(_sitk_array(view_paths["mask"])[0])
        if image.shape != (size, size):
            image = resize(
                image, (size, size), preserve_range=True, anti_aliasing=True
            )
            mask = resize(
                mask, (size, size), preserve_range=True, anti_aliasing=False
            )
        image_tensor = torch.tensor(
            image, device=device, dtype=torch.float32
        )[None, None, None]
        mask_tensor = torch.tensor(
            mask, device=device, dtype=torch.float32
        )[None, None, None]
        with torch.no_grad():
            images[view] = bilateral(image_tensor * mask_tensor)[:, :, 0]
        masks[view] = mask_tensor[:, :, 0]
        with view_paths["metadata"].open() as stream:
            metadata[view] = json.load(stream)
    return images, masks, metadata


class TestTimeOptimizer(nn.Module):
    """The matched publication NAdam/OneCycle GeoReg objective."""

    def __init__(
        self,
        cta_subject,
        images: dict[str, torch.Tensor],
        masks: dict[str, torch.Tensor],
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
        self.masks = masks
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
        ncc_losses, dice_losses = {}, {}
        total = torch.zeros((), device=self.device)
        for view in ("lat", "pa"):
            estimate = self.render(view)
            ncc_loss = -self.ncc(
                self.images[view], estimate.sum(dim=1, keepdim=True)
            )
            dice_loss = self.dice(
                torch.sigmoid(estimate[:, 1:2]),
                (self.masks[view] > 0).float(),
            )
            ncc_losses[view] = ncc_loss
            dice_losses[view] = dice_loss
            total = total + 0.5 * ncc_loss + 0.5 * dice_loss
        return ncc_losses, dice_losses, total

    @torch.no_grad()
    def snapshot(self) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
        return {
            view: tuple(value.detach().clone() for value in self.pose(view))
            for view in ("lat", "pa")
        }

    def optimize(self, iterations: int = 25) -> tuple[dict, dict]:
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
                    "dice_loss": float(dice[view]),
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


def run_inference(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("The publication inference pipeline requires CUDA")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda")
    data_root = args.data_root.resolve()
    patient = args.patient

    cta_path = data_root / "CTATr" / f"{patient}_0000.nii.gz"
    cta_mask_path = data_root / "CTA_skullTr" / f"{patient}.nii.gz"
    for required in (cta_path, cta_mask_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    cta_subject = read(str(cta_path), str(cta_mask_path), labels=[0, 1])

    init_model = load_init_model(
        args.init_checkpoint.resolve(), device, args.skip_hash_check
    )
    refine_model = load_refine_model(
        args.refine_checkpoint.resolve(), device, args.skip_hash_check
    )
    initializer = PoseInitializer(
        init_model,
        refine_model,
        cta_subject,
        data_root,
        device,
        max_refine_updates=args.max_refine_updates,
    )
    initial_poses, initialization_trace = initializer.predict(
        patient, args.timestamp
    )

    images, masks, metadata = prepare_registration_inputs(
        data_root, patient, args.timestamp, device
    )
    optimizer = TestTimeOptimizer(
        cta_subject,
        images,
        masks,
        metadata,
        initial_poses,
        device,
    ).to(device)
    final_poses, optimization_trace = optimizer.optimize(args.iterations)

    result = {
        "schema_version": 1,
        "contract": "geopose-inference-v1",
        "patient": patient,
        "timestamp": args.timestamp,
        "checkpoints": {
            "init": {
                "path": str(args.init_checkpoint.resolve()),
                "sha256": file_sha256(args.init_checkpoint.resolve()),
            },
            "refine": {
                "path": str(args.refine_checkpoint.resolve()),
                "sha256": file_sha256(args.refine_checkpoint.resolve()),
            },
        },
        "initialization": initialization_trace,
        "optimization": optimization_trace,
        "final_pose": {
            view: {
                "rotation_zyx_radians": tensor_list(rotation),
                "translation_mm": tensor_list(translation),
                "matrix": tensor_list(pose_matrix(rotation, translation)),
            }
            for view, (rotation, translation) in final_poses.items()
        },
        "example_note": (
            "sub-stroke9999 is the alignment template and a functional public "
            "example, not an independent held-out evaluation case."
            if patient == "sub-stroke9999"
            else None
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    save_final_renders(args.output_dir, optimizer, final_poses)
    return result


def _existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    return path


def _existing_directory(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Directory does not exist: {path}")
    return path


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run calibrated GeoPose + greedy Refine + 25-step GeoReg TTO."
    )
    parser.add_argument("--data-root", required=True, type=_existing_directory)
    parser.add_argument("--patient", default="sub-stroke9999")
    parser.add_argument("--timestamp", choices=("pre", "post"), default="pre")
    parser.add_argument("--init-checkpoint", required=True, type=_existing_file)
    parser.add_argument("--refine-checkpoint", required=True, type=_existing_file)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--max-refine-updates", type=int, default=5)
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Allow non-publication checkpoints, such as newly retrained weights.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.iterations < 0:
        raise ValueError("--iterations must be nonnegative")
    if args.max_refine_updates < 0:
        raise ValueError("--max-refine-updates must be nonnegative")
    result = run_inference(args)
    print(json.dumps(result["final_pose"], indent=2))


if __name__ == "__main__":
    main()
