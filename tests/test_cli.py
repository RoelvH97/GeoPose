"""Runtime regressions for the three command-line entry points."""

import json

import pytest

from geopose.cli import preregister as preregister_cli
from geopose.cli import test as test_cli
from geopose.cli import train as train_cli


def test_preregister_prepare_runs_end_to_end(tmp_path, monkeypatch):
    """Regression: `prepare` used json without importing it and died on startup."""
    calls = {}

    def _fake_prepare(isles_root, carotid_root, source_root, **kwargs):
        calls["prepare"] = (isles_root, carotid_root, source_root, kwargs)
        return {}

    monkeypatch.setattr(preregister_cli, "prepare_public_isles", _fake_prepare)

    cohort = tmp_path / "cohort.json"
    cohort.write_text(
        json.dumps({"train": ["sub-stroke0001"], "val": ["sub-stroke0002"], "test": []})
    )
    for directory in ("isles", "carotids", "source"):
        (tmp_path / directory).mkdir()

    preregister_cli.main(
        [
            "prepare",
            "--isles-root", str(tmp_path / "isles"),
            "--carotid-root", str(tmp_path / "carotids"),
            "--source-root", str(tmp_path / "source"),
            "--cohort-file", str(cohort),
            "--skip-hd-bet",
        ]
    )

    assert calls["prepare"][3]["selected_subjects"] == {
        "sub-stroke0001",
        "sub-stroke0002",
    }
    assert calls["prepare"][3]["skip_hd_bet"] is True


def test_preregister_all_reaches_both_stages(tmp_path, monkeypatch):
    calls = {}
    monkeypatch.setattr(
        preregister_cli,
        "prepare_public_isles",
        lambda *args, **kwargs: calls.setdefault("prepare", True) or {},
    )
    monkeypatch.setattr(
        preregister_cli,
        "register_cohort",
        lambda *args, **kwargs: calls.setdefault("align", True),
    )

    cohort = tmp_path / "cohort.json"
    cohort.write_text(json.dumps({"train": ["sub-stroke0001"], "val": [], "test": []}))
    for directory in ("isles", "carotids", "source", "out"):
        (tmp_path / directory).mkdir()

    preregister_cli.main(
        [
            "all",
            "--isles-root", str(tmp_path / "isles"),
            "--carotid-root", str(tmp_path / "carotids"),
            "--source-root", str(tmp_path / "source"),
            "--output-root", str(tmp_path / "out"),
            "--cohort-file", str(cohort),
            "--skip-hd-bet",
        ]
    )

    assert calls == {"prepare": True, "align": True}


def test_preregister_defaults_to_the_frozen_cohort_file():
    assert preregister_cli.FROZEN_SPLIT.is_file()
    split = json.loads(preregister_cli.FROZEN_SPLIT.read_text())
    assert len(split["train"] + split["val"] + split["test"]) == 99


def test_train_cpu_accelerator_also_moves_dataset_rendering(tmp_path):
    args = train_cli._parser().parse_args(
        [
            "init",
            "--data-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--accelerator", "cpu",
        ]
    )
    cfg = train_cli._configure(args)
    assert cfg.trainer.accelerator == "cpu"
    assert cfg.data.device == "cpu"
    assert cfg.data.split_file == str(train_cli.SPLIT_FILE)


def test_train_rejects_init_checkpoint_outside_the_refine_stage(tmp_path):
    checkpoint = tmp_path / "init.ckpt"
    checkpoint.write_bytes(b"")
    args = train_cli._parser().parse_args(
        [
            "init",
            "--data-root", str(tmp_path),
            "--output-dir", str(tmp_path / "out"),
            "--init-checkpoint", str(checkpoint),
        ]
    )
    with pytest.raises(ValueError, match="only valid for stage=refine"):
        train_cli._configure(args)


def test_train_refine_stage_requires_an_init_checkpoint(tmp_path):
    args = train_cli._parser().parse_args(
        ["refine", "--data-root", str(tmp_path), "--output-dir", str(tmp_path / "out")]
    )
    with pytest.raises(ValueError, match="--init-checkpoint is required"):
        train_cli._configure(args)


def test_test_cli_rejects_negative_iteration_counts(tmp_path):
    checkpoint = tmp_path / "ckpt"
    checkpoint.write_bytes(b"")
    argv = [
        "--data-root", str(tmp_path),
        "--init-checkpoint", str(checkpoint),
        "--refine-checkpoint", str(checkpoint),
        "--output-dir", str(tmp_path / "out"),
        "--iterations", "-1",
    ]
    with pytest.raises(ValueError, match="must be nonnegative"):
        test_cli.main(argv)
