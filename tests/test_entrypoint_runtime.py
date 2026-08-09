"""Focused regressions for root entry-point runtime failures."""

import json
from pathlib import Path

import nibabel as nib
import numpy as np
from omegaconf import OmegaConf

from geopose.init import data as init_data
from geopose.refine import data as refine_data
from geopose.registration.initialization import read_map_channel
from geopose.registration.preregistration import HU_SHIFT


def _save_nifti(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(array, np.eye(4)), path)


def test_preregistration_uses_the_frozen_hu_shift():
    assert HU_SHIFT == 1000


def test_map_preprocessing_resolves_the_largest_component_helper(tmp_path):
    patient = "sub-stroke9999"
    image = np.arange(64, dtype=np.float32).reshape(8, 8, 1)
    mask = np.zeros((8, 8, 1), dtype=np.uint8)
    mask[1:5, 1:5] = 1
    mask[7, 7] = 1

    _save_nifti(tmp_path / "MAPTr" / f"{patient}_a_0000.nii.gz", image)
    _save_nifti(tmp_path / "MAP_maskTr" / f"{patient}_a.nii.gz", mask)
    metadata = tmp_path / "DSA_arteriesTr" / f"{patient}_a.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text(
        json.dumps({"d_source_to_detector": 1020.0, "alpha": -80.0})
    )

    processed, source_sdd, alpha = read_map_channel(
        tmp_path, patient, "a", size=8
    )

    assert processed.shape == (8, 8)
    assert np.count_nonzero(processed) <= 16
    assert source_sdd == 1020.0
    assert alpha == -80.0


class _DatasetStub:
    def __init__(self, _cfg, indices, training=None):
        self.indices = list(indices)
        self.training = training

    def __len__(self):
        return len(self.indices)


def test_init_smoke_limit_bounds_every_split(monkeypatch):
    monkeypatch.setattr(init_data, "_list_image_paths", lambda *args: list(range(9)))
    monkeypatch.setattr(
        init_data,
        "split_indices",
        lambda paths, cfg: ([0, 1, 2], [3, 4, 5], [6, 7, 8]),
    )
    monkeypatch.setattr(init_data, "CTAPoseDataset", _DatasetStub)
    cfg = OmegaConf.create(
        {
            "data_root": "/unused",
            "dataset": "cta",
            "align_suffix": "alignedv2",
            "max_subjects": 2,
        }
    )

    module = init_data.CTAPoseDataModule(cfg)
    module.setup()

    assert module.train_dataset.indices == [0, 1]
    assert module.val_dataset.dataset.indices == [3, 4, 6, 7]
    assert list(module.val_dataset.indices) == [0, 1]
    assert list(module.test_dataset.indices) == [2, 3]


def test_refine_smoke_limit_bounds_every_split(monkeypatch):
    monkeypatch.setattr(refine_data, "_list_image_paths", lambda *args: list(range(9)))
    monkeypatch.setattr(
        refine_data,
        "split_indices",
        lambda paths, cfg: ([0, 1, 2], [3, 4, 5], [6, 7, 8]),
    )
    monkeypatch.setattr(refine_data, "SyntheticRefineDataset", _DatasetStub)
    cfg = OmegaConf.create(
        {
            "data": {
                "data_root": "/unused",
                "dataset": "cta",
                "align_suffix": "alignedv2",
                "max_subjects": 2,
            }
        }
    )

    module = refine_data.SyntheticRefineDataModule(cfg)
    module.setup()

    assert module.train_dataset.indices == [0, 1]
    assert module.val_dataset.indices == [3, 4]
    assert module.test_dataset.indices == [6, 7]
