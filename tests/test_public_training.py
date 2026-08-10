"""Public CTA-only training contract tests."""

import inspect
from pathlib import Path

import torch

from geopose.cli.train import _parser, load_training_contract
from geopose.init.data import CTAPoseDataset
from geopose.init.loss import GeoPoseCriterion
from geopose.init.model import ResNetPose
from geopose.refine.data import SyntheticRefineDataset
from geopose.refine.model import RefinePoseModule


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_BATCH_KEYS = {
    "maps",
    "segs",
    "map_view_labels",
    "maps_weak",
    "maps_strong",
    "map_view_labels_clean",
}


class _PoseToken:
    def __init__(self, batch_size: int):
        self.batch_size = batch_size

    def compose(self, _other):
        return self


class _FakeDRR:
    mask = torch.tensor([0, 1, 2, 3, 4])

    def to(self, _device):
        return self

    def cpu(self):
        return self

    def __call__(self, pose, mask_to_channels=False):
        channels = 5 if mask_to_channels else 1
        return torch.ones(pose.batch_size, channels, 8, 8)


def test_training_cli_and_contracts_expose_no_dsa_data_path():
    assert "--dsa-root" not in _parser().format_help()
    for stage in ("init", "refine"):
        cfg = load_training_contract(stage)
        assert "dsa_root" not in cfg.data
        assert not any(key.startswith("map_") for key in cfg.data.augmentation)

    init = load_training_contract("init")
    forbidden_model_keys = {
        "lambda_da",
        "lambda_map_ncc",
        "lambda_map_r0",
        "lambda_view_cls_map",
    }
    assert forbidden_model_keys.isdisjoint(init.model)


def test_init_dataset_batch_contains_only_synthetic_cta_fields():
    dataset = object.__new__(CTAPoseDataset)
    dataset.device = torch.device("cpu")
    dataset.batch_size = 2
    dataset.art_end = 3
    dataset.training = False
    dataset.drrs = [_FakeDRR()]
    dataset.fiducials = [None]
    dataset.get_random_pose_batch = lambda n: (
        _PoseToken(n),
        torch.zeros(n, 3),
        torch.zeros(n, 3),
        torch.zeros(n, dtype=torch.bool),
        torch.ones(n, dtype=torch.long),
    )
    dataset._get_isopose = lambda: _PoseToken(1)

    batch = dataset[0]

    assert FORBIDDEN_BATCH_KEYS.isdisjoint(batch)
    assert {"images", "art_gt", "poses", "view_label"}.issubset(batch)


def test_refine_dataset_names_its_synthetic_target_explicitly():
    dataset = object.__new__(SyntheticRefineDataset)
    dataset.device = torch.device("cpu")
    dataset.batch_size = 2
    dataset.training = False
    dataset.deterministic_eval = False
    dataset.eval_seed = 0
    dataset.art_end = 3
    dataset.drrs = [_FakeDRR()]
    dataset.fiducials = [None]
    dataset._std_R = {"lat": [0.1] * 3, "pa": [0.1] * 3}
    dataset._std_t = {"lat": [1.0] * 3, "pa": [1.0] * 3}
    dataset._sample_gt_pose_batch = lambda n, generator=None: (
        _PoseToken(n),
        torch.zeros(n, dtype=torch.bool),
        torch.ones(n, dtype=torch.long),
    )

    batch = dataset[0]

    assert "target_drr" in batch
    assert "map" not in batch


def test_training_objective_and_logging_have_no_real_dsa_branches():
    criterion_parameters = inspect.signature(GeoPoseCriterion.forward).parameters
    assert not any("map" in name or "domain" in name for name in criterion_parameters)
    assert not hasattr(ResNetPose, "_log_map_panel")

    constants = {
        value
        for value in RefinePoseModule.validation_step.__code__.co_consts
        if isinstance(value, str)
    }
    assert {
        "val/ncc_noisy_synth",
        "val/ncc_synth",
        "val/delta_ncc_synth",
    }.issubset(constants)
    assert not any("dsa" in value.lower() or "map" in value.lower() for value in constants)


def test_training_sources_do_not_reference_private_dsa_files():
    paths = [
        ROOT / "src/geopose/cli/train.py",
        ROOT / "src/geopose/init/data.py",
        ROOT / "src/geopose/init/loss.py",
        ROOT / "src/geopose/refine/data.py",
        ROOT / "src/geopose/configs/init.yaml",
        ROOT / "src/geopose/configs/refine.yaml",
    ]
    forbidden = ("MAPTr", "MAP_maskTr", "DSA_arteriesTr", "MIP_arteriesTr")
    for path in paths:
        source = path.read_text()
        assert not any(token in source for token in forbidden), path
