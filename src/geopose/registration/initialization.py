"""GeoPose-Init calibration followed by greedy GeoPose-Refine."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn as nn
from diffdrr.drr import DRR
from diffdrr.metrics import MultiscaleNormalizedCrossCorrelation2d
from diffdrr.pose import RigidTransform, convert
from diffdrr.utils import resample as resample_intrinsics
from omegaconf import DictConfig, OmegaConf
from skimage.transform import resize

from ..init.model import ResNetPose
from ..refine.network import build_refine_pose_net, refiner_view_index
from ..shared.pose import delta_to_pose
from .geometry import matrix_to_pose, pose_matrix, tensor_list
from .images import largest_component, minmax
from .projections import ProjectionInput
from .views import (
    CHANNEL_PAIR,
    MAP_META_CHANNEL_OVERRIDES,
    VIEW_CHANNELS,
    view_label_from_alpha,
)

try:
    from bilateral_filter_layer import BilateralFilter3d
except ImportError as exc:
    raise ImportError(
        "Faithful GeoPose inference requires bilateralfilter_torch==1.1.0"
    ) from exc


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PACKAGE_ROOT / "configs"


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

    # Avoid downloading torchvision weights before strict checkpoint loading.
    model_cfg.pretrained = False
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


def read_map_channel(
    data_root: Path,
    patient: str,
    channel: str,
    *,
    target_sdd: float = 1020.0,
    target_delx: float = 1.2,
    size: int = 256,
) -> tuple[np.ndarray, float, float | None]:
    """Apply the exact training/deployment MAP geometric preprocessing."""
    image_path = data_root / "MAPTr" / f"{patient}_{channel}_0000.nii.gz"
    mask_path = data_root / "MAP_maskTr" / f"{patient}_{channel}.nii.gz"
    for required in (image_path, mask_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    image = nib.load(str(image_path)).get_fdata(dtype=np.float32).squeeze()
    image = np.fliplr(np.rot90(image, 3))
    mask = nib.load(str(mask_path)).get_fdata(dtype=np.float32).squeeze()
    mask = largest_component(np.fliplr(np.rot90(mask, 3)))

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
    alpha_value = metadata.get("alpha")
    alpha = None if alpha_value is None else float(alpha_value)

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


def resolve_view_label(alpha: float | None, view: str) -> tuple[int, str]:
    """Resolve view role from acquisition angle or channel convention."""
    if alpha is not None:
        return view_label_from_alpha(alpha), "alpha"
    if view == "lat":
        return 0, "channel_role"
    if view == "pa":
        return 1, "channel_role"
    raise ValueError(f"Unknown projection view: {view!r}")


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
        self,
        patient: str,
        timestamp: str,
        projections: dict[str, ProjectionInput] | None = None,
    ) -> tuple[dict[str, tuple[torch.Tensor, torch.Tensor]], dict]:
        output: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
        trace = {
            "iso_matrix": tensor_list(self.iso_matrix),
            "calibration_matrix": tensor_list(self.calibration_matrix),
            "views": {},
        }
        calibration_inverse = torch.linalg.inv(self.calibration_matrix)
        for view, channel in VIEW_CHANNELS[timestamp].items():
            if projections is None:
                array, source_sdd, alpha = read_map_channel(
                    self.data_root,
                    patient,
                    channel,
                    target_sdd=self.target_sdd,
                    target_delx=self.spacing,
                    size=self.size,
                )
            else:
                projection = projections[view]
                channel = projection.channel
                array = projection.init_image
                source_sdd = projection.source_sdd
                alpha = projection.alpha_degrees
            image = torch.from_numpy(array).to(self.device)[None, None]
            image = bilateral_minmax(image)
            label, label_source = resolve_view_label(alpha, view)
            metadata_label = torch.tensor([label], device=self.device)
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
                "metadata_label_source": label_source,
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
