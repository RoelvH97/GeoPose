"""Contracts for privacy-preserving, model-ready projection files."""

from pathlib import Path

import numpy as np
import pytest

from geopose.registration.projections import (
    ProjectionInput,
    load_projection_file,
    save_projection_file,
)


def _projection(channel: str, alpha: float) -> ProjectionInput:
    image = np.arange(64, dtype=np.float32).reshape(8, 8)
    return ProjectionInput(
        channel=channel,
        alpha_degrees=alpha,
        source_sdd=1100.0,
        init_image=image,
        registration_image=image + 1,
        registration_mask=np.ones((8, 8), dtype=np.float32),
    )


def test_projection_file_round_trip(tmp_path: Path):
    path = tmp_path / "sub-stroke0011_pre.npz"
    save_projection_file(
        path,
        "sub-stroke0011",
        "pre",
        {"lat": _projection("a", -89.7), "pa": _projection("b", 0.1)},
    )

    loaded = load_projection_file(path, "sub-stroke0011", "pre", size=8)

    assert loaded["lat"].channel == "a"
    assert loaded["lat"].alpha_degrees == pytest.approx(-89.7)
    assert loaded["pa"].source_sdd == 1100.0
    np.testing.assert_array_equal(
        loaded["pa"].registration_image,
        np.arange(64, dtype=np.float32).reshape(8, 8) + 1,
    )


def test_projection_file_rejects_extra_arrays(tmp_path: Path):
    path = tmp_path / "example.npz"
    save_projection_file(
        path,
        "sub-stroke0011",
        "pre",
        {"lat": _projection("a", -89.7), "pa": _projection("b", 0.1)},
    )
    with np.load(path, allow_pickle=False) as bundle:
        fields = {key: bundle[key] for key in bundle.files}
    fields["raw_dsa"] = np.zeros((8, 8, 2), dtype=np.float32)
    np.savez_compressed(path, **fields)

    with pytest.raises(ValueError, match="fields"):
        load_projection_file(path, "sub-stroke0011", "pre", size=8)


def test_projection_file_rejects_identity_mismatch(tmp_path: Path):
    path = tmp_path / "example.npz"
    save_projection_file(
        path,
        "sub-stroke0011",
        "pre",
        {"lat": _projection("a", -89.7), "pa": _projection("b", 0.1)},
    )

    with pytest.raises(ValueError, match="patient"):
        load_projection_file(path, "sub-stroke0022", "pre", size=8)
