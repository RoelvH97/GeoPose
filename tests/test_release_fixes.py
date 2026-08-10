"""Regressions for publication-readiness and reproducibility fixes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

from geopose.cli import test as test_cli
from geopose.cli import train as train_cli
from geopose.registration.pipeline import configure_reproducibility, run_inference
from geopose.registration.projections import (
    ProjectionInput,
    load_projection_file,
    save_projection_file,
)
from geopose.shared.contracts import (
    EXAMPLE_MANIFEST,
    PACKAGED_EXAMPLE,
    sha256,
)
from geopose.shared.visualization import euler_zyx_from_matrix


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


def _bundle(tmp_path: Path) -> Path:
    path = tmp_path / "projection.npz"
    save_projection_file(
        path,
        "sub-stroke0011",
        "pre",
        {"lat": _projection("a", -89.7), "pa": _projection("b", 0.1)},
    )
    return path


def _replace_array(path: Path, key: str, value) -> None:
    with np.load(path, allow_pickle=False) as bundle:
        fields = {name: bundle[name] for name in bundle.files}
    fields[key] = np.asarray(value)
    np.savez_compressed(path, **fields)


def test_packaged_example_matches_the_release_manifest():
    manifest = json.loads(EXAMPLE_MANIFEST.read_text())
    assert PACKAGED_EXAMPLE.is_file()
    assert PACKAGED_EXAMPLE.stat().st_size == manifest["projection"]["bytes"]
    assert sha256(PACKAGED_EXAMPLE) == manifest["projection"]["sha256"]


def test_cli_falls_back_to_the_packaged_example(tmp_path):
    args = argparse.Namespace(
        projection_file=None,
        data_root=tmp_path,
        patient="sub-stroke0011",
        timestamp="pre",
    )
    assert test_cli._resolve_projection_file(args) == PACKAGED_EXAMPLE.resolve()


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("lat_source_sdd", -1.0, "must be positive"),
        ("lat_source_sdd", np.nan, "must be finite"),
        ("lat_alpha_degrees", 0.0, "inconsistent with view"),
        ("lat_channel", "b", "must be 'a'"),
        ("lat_registration_mask", np.zeros((8, 8), np.float32), "must not be empty"),
    ],
)
def test_projection_contract_rejects_invalid_geometry(tmp_path, key, value, message):
    path = _bundle(tmp_path)
    _replace_array(path, key, value)
    with pytest.raises(ValueError, match=message):
        load_projection_file(path, "sub-stroke0011", "pre", size=8)


def test_training_seed_controls_configured_samplers(tmp_path):
    args = train_cli._parser().parse_args(
        [
            "init",
            "--data-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--seed", "123",
        ]
    )
    cfg = train_cli._configure(args)
    assert cfg.seed == 123
    assert cfg.data.seed == 123


def test_cli_output_directories_expand_the_user_home(monkeypatch, tmp_path):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert train_cli._output_directory("~/train") == (tmp_path / "train").resolve()
    assert test_cli._output_directory("~/test") == (tmp_path / "test").resolve()


def test_exact_gimbal_lock_euler_conversion_preserves_scalar_shape():
    rotation = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    angles = euler_zyx_from_matrix(rotation)
    assert angles.shape == (3,)
    assert angles.dtype == np.float64
    np.testing.assert_allclose(angles, [0.0, np.pi / 2, 0.0], atol=1e-7)


def test_reproducibility_policy_is_explicit_and_restores_cleanly():
    old_enabled = torch.are_deterministic_algorithms_enabled()
    old_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    old_benchmark = torch.backends.cudnn.benchmark
    old_cudnn_deterministic = torch.backends.cudnn.deterministic
    try:
        record = configure_reproducibility("warn")
        assert record == {
            "seed": 0,
            "determinism": "warn",
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
        }
        assert torch.are_deterministic_algorithms_enabled()
        assert torch.is_deterministic_algorithms_warn_only_enabled()
        with pytest.raises(ValueError, match="determinism mode"):
            configure_reproducibility("sometimes")
    finally:
        torch.use_deterministic_algorithms(old_enabled, warn_only=old_warn_only)
        torch.backends.cudnn.benchmark = old_benchmark
        torch.backends.cudnn.deterministic = old_cudnn_deterministic


@pytest.mark.integration
def test_packaged_example_inference_is_repeatable(tmp_path):
    init_checkpoint = os.environ.get("GEOPOSE_INIT_CHECKPOINT")
    refine_checkpoint = os.environ.get("GEOPOSE_REFINE_CHECKPOINT")
    data_root = os.environ.get("GEOPOSE_EXAMPLE_DATA_ROOT")
    if not init_checkpoint or not refine_checkpoint or not data_root:
        pytest.skip(
            "set GEOPOSE_INIT_CHECKPOINT, GEOPOSE_REFINE_CHECKPOINT, and "
            "GEOPOSE_EXAMPLE_DATA_ROOT"
        )
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required for publication inference")

    def _run(output: Path):
        args = argparse.Namespace(
            data_root=Path(data_root),
            patient="sub-stroke0011",
            timestamp="pre",
            projection_file=PACKAGED_EXAMPLE,
            init_checkpoint=Path(init_checkpoint),
            refine_checkpoint=Path(refine_checkpoint),
            output_dir=output,
            iterations=0,
            max_refine_updates=5,
            skip_hash_check=False,
            determinism="error",
        )
        return run_inference(args)

    first = _run(tmp_path / "first")
    second = _run(tmp_path / "second")
    for view in ("lat", "pa"):
        np.testing.assert_allclose(
            first["final_pose"][view]["matrix"],
            second["final_pose"][view]["matrix"],
            rtol=0.0,
            atol=1e-6,
        )


@pytest.mark.integration
def test_packaged_projection_matches_private_preprocessing_boundaries():
    private_root = os.environ.get("GEOPOSE_PRIVATE_DATA_ROOT")
    if not private_root:
        pytest.skip("set GEOPOSE_PRIVATE_DATA_ROOT for maintainer route equivalence")

    from geopose.registration.initialization import read_map_channel
    from geopose.registration.optimization import read_registration_channels
    from geopose.registration.views import VIEW_CHANNELS

    root = Path(private_root)
    projections = load_projection_file(
        PACKAGED_EXAMPLE, "sub-stroke0011", "pre"
    )
    registration, cranium_masks, metadata = read_registration_channels(
        root, "sub-stroke0011", "pre"
    )
    for view, channel in VIEW_CHANNELS["pre"].items():
        init_image, source_sdd, alpha = read_map_channel(
            root, "sub-stroke0011", channel
        )
        projection = projections[view]
        np.testing.assert_array_equal(projection.init_image, init_image)
        np.testing.assert_array_equal(
            projection.registration_image, registration[view]
        )
        np.testing.assert_array_equal(
            projection.registration_mask, cranium_masks[view]
        )
        assert projection.source_sdd == source_sdd
        assert projection.alpha_degrees == alpha
        assert metadata[view]["d_source_to_detector"] == source_sdd
