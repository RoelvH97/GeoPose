"""Validated model-ready projection inputs for privacy-preserving examples."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np

from .views import VIEW_CHANNELS, VIEW_THRESHOLD_DEGREES


PROJECTION_CONTRACT = "geopose-projections-v1"
VIEWS = ("lat", "pa")
VIEW_FIELDS = (
    "channel",
    "alpha_degrees",
    "source_sdd",
    "init_image",
    "registration_image",
    "registration_mask",
)
REQUIRED_FIELDS = {"contract", "patient", "timestamp"} | {
    f"{view}_{field}" for view in VIEWS for field in VIEW_FIELDS
}


@dataclass(frozen=True)
class ProjectionInput:
    """One view at the deterministic Init and GeoReg preprocessing boundaries."""

    channel: str
    alpha_degrees: float
    source_sdd: float
    init_image: np.ndarray
    registration_image: np.ndarray
    registration_mask: np.ndarray


def _scalar(bundle, key: str):
    value = bundle[key]
    if value.shape != ():
        raise ValueError(f"Projection field {key!r} must be scalar")
    return value.item()


def _image(bundle, key: str, size: int) -> np.ndarray:
    image = np.asarray(bundle[key])
    if image.shape != (size, size):
        raise ValueError(
            f"Projection field {key!r} has shape {image.shape}, expected {(size, size)}"
        )
    if image.dtype != np.float32:
        raise ValueError(f"Projection field {key!r} must be float32")
    if not np.isfinite(image).all():
        raise ValueError(f"Projection field {key!r} contains non-finite values")
    return np.ascontiguousarray(image)


def _finite_float(bundle, key: str) -> float:
    value = float(_scalar(bundle, key))
    if not math.isfinite(value):
        raise ValueError(f"Projection field {key!r} must be finite")
    return value


def load_projection_file(
    path: Path,
    patient: str,
    timestamp: str,
    *,
    size: int = 256,
) -> dict[str, ProjectionInput]:
    """Load a versioned pair of downsampled projection inputs without pickle."""

    if timestamp not in VIEW_CHANNELS:
        raise ValueError(f"Unsupported projection timestamp {timestamp!r}")

    with np.load(path, allow_pickle=False) as bundle:
        fields = set(bundle.files)
        if fields != REQUIRED_FIELDS:
            difference = sorted(fields ^ REQUIRED_FIELDS)
            raise ValueError(f"Projection fields do not match the contract: {difference}")
        if _scalar(bundle, "contract") != PROJECTION_CONTRACT:
            raise ValueError(f"Unsupported projection contract in {path}")
        if _scalar(bundle, "patient") != patient:
            raise ValueError(f"Projection patient does not match {patient}")
        if _scalar(bundle, "timestamp") != timestamp:
            raise ValueError(f"Projection timestamp does not match {timestamp}")

        projections = {}
        for view in VIEWS:
            mask = _image(bundle, f"{view}_registration_mask", size)
            if float(mask.min()) < 0.0 or float(mask.max()) > 1.0:
                raise ValueError(f"Projection mask {view!r} must lie in [0, 1]")
            if not np.any(mask > 0):
                raise ValueError(f"Projection cranium mask {view!r} must not be empty")

            channel = str(_scalar(bundle, f"{view}_channel"))
            expected_channel = VIEW_CHANNELS[timestamp][view]
            if channel != expected_channel:
                raise ValueError(
                    f"Projection channel for {view!r} must be {expected_channel!r} "
                    f"at timestamp {timestamp!r}, got {channel!r}"
                )

            alpha = _finite_float(bundle, f"{view}_alpha_degrees")
            is_lateral = abs(alpha) > VIEW_THRESHOLD_DEGREES
            if (view == "lat") != is_lateral:
                raise ValueError(
                    f"Projection angle {alpha} degrees is inconsistent with view {view!r}"
                )

            source_sdd = _finite_float(bundle, f"{view}_source_sdd")
            if source_sdd <= 0.0:
                raise ValueError(
                    f"Projection source-to-detector distance for {view!r} must be positive"
                )
            projections[view] = ProjectionInput(
                channel=channel,
                alpha_degrees=alpha,
                source_sdd=source_sdd,
                init_image=_image(bundle, f"{view}_init_image", size),
                registration_image=_image(
                    bundle, f"{view}_registration_image", size
                ),
                registration_mask=mask,
            )
    return projections


def save_projection_file(
    path: Path,
    patient: str,
    timestamp: str,
    projections: dict[str, ProjectionInput],
) -> None:
    """Save deterministic 256-pixel projection boundaries as one compressed file."""

    fields: dict[str, np.ndarray] = {
        "contract": np.asarray(PROJECTION_CONTRACT),
        "patient": np.asarray(patient),
        "timestamp": np.asarray(timestamp),
    }
    for view in VIEWS:
        projection = projections[view]
        fields.update(
            {
                f"{view}_channel": np.asarray(projection.channel),
                f"{view}_alpha_degrees": np.asarray(
                    projection.alpha_degrees, dtype=np.float64
                ),
                f"{view}_source_sdd": np.asarray(
                    projection.source_sdd, dtype=np.float64
                ),
                f"{view}_init_image": np.asarray(
                    projection.init_image, dtype=np.float32
                ),
                f"{view}_registration_image": np.asarray(
                    projection.registration_image, dtype=np.float32
                ),
                f"{view}_registration_mask": np.asarray(
                    projection.registration_mask, dtype=np.float32
                ),
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **fields)
