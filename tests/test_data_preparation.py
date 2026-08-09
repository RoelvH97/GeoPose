from pathlib import Path

import nibabel as nib
import numpy as np
import pytest

from geopose.data.preparation import stage_public_isles


def _save(path: Path, data: np.ndarray, affine=None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(
        nib.Nifti1Image(data, np.eye(4) if affine is None else affine),
        str(path),
    )


def _public_pair(
    cta_root: Path, carotid_root: Path, affine=None
) -> tuple[Path, Path, np.ndarray, np.ndarray]:
    cta = (
        cta_root
        / "raw_data/sub-strokecase0001/ses-0001"
        / "sub-strokecase0001_ses-0001_cta.nii.gz"
    )
    mask = (
        carotid_root
        / "sub-strokecase0001/ses-0001"
        / "sub-stroke0086-max_msk.nii.gz"
    )
    image = np.arange(24, dtype=np.int16).reshape(2, 3, 4)
    labels = np.zeros((2, 3, 4), dtype=np.uint8)
    labels[0, 0, 0] = 1
    labels[1, 2, 3] = 2
    labels[0, 1, 1] = 72
    _save(cta, image)
    _save(mask, labels, affine)
    return cta, mask, image, labels


def test_public_adapter_preserves_native_cta_and_zenodo_labelmap(tmp_path):
    cta_root = tmp_path / "isles"
    carotid_root = tmp_path / "geopose_carotids"
    _, _, image, labels = _public_pair(cta_root, carotid_root)
    output = tmp_path / "standardized"
    manifest = stage_public_isles(cta_root, carotid_root, output)

    assert [record["subject"] for record in manifest["subjects"]] == ["sub-stroke0086"]
    staged_image = np.asanyarray(
        nib.load(output / "CTATr/sub-stroke0086_0000.nii.gz").dataobj
    )
    staged_labels = np.asanyarray(
        nib.load(output / "CTA_carotisTr/sub-stroke0086.nii.gz").dataobj
    )
    assert np.array_equal(staged_image, image)
    assert np.array_equal(staged_labels, labels)
    assert manifest["subjects"][0]["source_labels"] == [0, 1, 2, 72]
    assert (output / "public_isles_manifest.json").is_file()


def test_public_adapter_accepts_flat_original_names_with_pair_manifest(tmp_path):
    cta_root = tmp_path / "isles"
    tree_root = tmp_path / "tree"
    _, _, _, labels = _public_pair(cta_root, tree_root)
    carotid_root = tmp_path / "flat_carotids"
    flat_mask = carotid_root / "sub-stroke0086-max_msk.nii.gz"
    _save(flat_mask, labels)
    (carotid_root / "carotid_pairs.json").write_text(
        """{
  "schema_version": 1,
  "subjects": [
    {
      "subject": "sub-stroke0086",
      "isles_case": "sub-strokecase0001",
      "path": "sub-stroke0086-max_msk.nii.gz"
    }
  ]
}
"""
    )
    output = tmp_path / "standardized"
    manifest = stage_public_isles(cta_root, carotid_root, output)
    assert manifest["subjects"][0]["subject"] == "sub-stroke0086"
    assert np.array_equal(
        np.asanyarray(nib.load(output / "CTA_carotisTr/sub-stroke0086.nii.gz").dataobj),
        labels,
    )


def test_public_adapter_frozen_subject_filter(tmp_path):
    _public_pair(tmp_path / "isles", tmp_path / "carotids")
    with pytest.raises(FileNotFoundError, match="cohort subjects absent"):
        stage_public_isles(
            tmp_path / "isles",
            tmp_path / "carotids",
            tmp_path / "out",
            selected_subjects={"sub-stroke0001"},
        )


def test_public_adapter_rejects_grid_mismatch(tmp_path):
    affine = np.eye(4)
    affine[0, 3] = 2
    _public_pair(tmp_path / "isles", tmp_path / "carotids", affine)
    with pytest.raises(RuntimeError, match="grid mismatch"):
        stage_public_isles(
            tmp_path / "isles", tmp_path / "carotids", tmp_path / "out"
        )
