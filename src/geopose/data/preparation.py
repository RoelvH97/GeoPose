"""Adapt the public ISLES'24 archive to the frozen GeoPose input layout."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import nibabel as nib
import numpy as np


ISLES_ZENODO_RECORD = "https://zenodo.org/records/17652035"
REQUIRED_ICA_LABELS = {1, 2}


def _clinical_id(path: Path) -> str:
    match = re.search(r"(sub-stroke\d+)-max_msk\.nii\.gz$", path.name)
    if not match:
        raise ValueError(f"Unexpected derivative-carotis filename: {path}")
    return match.group(1)


def _case_id(path: Path) -> str:
    match = re.search(r"(sub-strokecase\d+)", str(path))
    if not match:
        raise ValueError(f"No sub-strokecase#### parent in {path}")
    return match.group(1)


def _carotid_records(carotid_root: Path) -> list[tuple[str, str, Path]]:
    """Return (clinical subject, public BIDS case, mask path) records."""
    manifest_path = carotid_root / "carotid_pairs.json"
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text())
        records = []
        for record in payload.get("subjects", []):
            subject = str(record["subject"])
            case = str(record["isles_case"])
            mask = (carotid_root / record["path"]).resolve()
            if not mask.is_file():
                raise FileNotFoundError(mask)
            records.append((subject, case, mask))
        if not records:
            raise ValueError(f"No subjects in {manifest_path}")
        return records

    masks = sorted(carotid_root.rglob("sub-stroke*-max_msk.nii.gz"))
    if masks:
        return [(_clinical_id(mask), _case_id(mask), mask) for mask in masks]

    raise FileNotFoundError(
        "GeoPose carotids need either preserved sub-strokecase parent "
        "directories with *-max_msk.nii.gz files or carotid_pairs.json."
    )


def discover_public_isles(
    isles_root: Path, carotid_root: Path
) -> dict[str, tuple[Path, Path]]:
    """Pair public native CTA with the separately published GeoPose mask."""
    ctas: dict[str, Path] = {}
    for cta in sorted(isles_root.rglob("*_cta.nii.gz")):
        if "raw_data" not in cta.parts:
            continue
        case = _case_id(cta)
        if case in ctas:
            raise RuntimeError(f"Multiple raw CTA files found for {case}")
        ctas[case] = cta

    pairs: dict[str, tuple[Path, Path]] = {}
    for subject, case, mask in _carotid_records(carotid_root):
        cta = ctas.get(case)
        if cta is None:
            raise FileNotFoundError(f"No raw CTA for {case}, required by {mask}")
        if subject in pairs:
            raise RuntimeError(f"Duplicate GeoPose carotid mask for {subject}")
        pairs[subject] = (cta, mask)

    if not pairs:
        raise FileNotFoundError(
            "No native raw CTA + GeoPose carotid pairs found. Expected public "
            "ISLES raw_data/sub-strokecase*/ses-0001/*_cta.nii.gz and the "
            "GeoPose Zenodo carotid archive. "
            f"CTA root: {isles_root}; carotid root: {carotid_root}."
        )
    return pairs


def _same_grid(
    image: nib.spatialimages.SpatialImage,
    mask: nib.spatialimages.SpatialImage,
) -> bool:
    return image.shape[:3] == mask.shape[:3] and np.allclose(
        image.affine, mask.affine, rtol=0.0, atol=1e-3
    )


def stage_public_isles(
    isles_root: Path,
    carotid_root: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
    selected_subjects: set[str] | None = None,
) -> dict:
    """Create CTATr/CTA_carotisTr without resampling or relabeling public data."""
    pairs = discover_public_isles(isles_root, carotid_root)
    if selected_subjects is not None:
        missing = selected_subjects - set(pairs)
        if missing:
            raise FileNotFoundError(
                "Frozen GeoPose cohort subjects absent from ISLES derivatives: "
                f"{sorted(missing)}"
            )
        pairs = {subject: pairs[subject] for subject in sorted(selected_subjects)}

    image_dir = output_root / "CTATr"
    carotid_dir = output_root / "CTA_carotisTr"
    image_dir.mkdir(parents=True, exist_ok=True)
    carotid_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for subject, (cta_path, carotid_path) in sorted(pairs.items()):
        image_out = image_dir / f"{subject}_0000.nii.gz"
        carotid_out = carotid_dir / f"{subject}.nii.gz"
        cta = nib.load(str(cta_path))
        carotid = nib.load(str(carotid_path))
        if not _same_grid(cta, carotid):
            raise RuntimeError(f"CTA/derivative-carotis grid mismatch for {subject}")
        labels = {int(value) for value in np.unique(np.asanyarray(carotid.dataobj))}
        missing_labels = REQUIRED_ICA_LABELS - labels
        if missing_labels:
            raise RuntimeError(
                f"{subject} derivative-carotis mask is missing GeoPose ICA labels "
                f"{sorted(missing_labels)}; present labels are {sorted(labels)}"
            )
        if overwrite or not image_out.is_file():
            shutil.copy2(cta_path, image_out)
        if overwrite or not carotid_out.is_file():
            shutil.copy2(carotid_path, carotid_out)
        records.append(
            {
                "subject": subject,
                "source_cta": str(cta_path.resolve()),
                "source_carotid": str(carotid_path.resolve()),
                "cta": str(image_out.resolve()),
                "carotid": str(carotid_out.resolve()),
                "source_labels": sorted(labels),
                "ica_labels_used_by_geopose": [1, 2],
            }
        )

    manifest = {
        "schema_version": 1,
        "contract": "geopose-public-isles-adapter-v1",
        "sources": {
            "cta": ISLES_ZENODO_RECORD,
            "carotids": "GeoPose Zenodo deposit (DOI pending)",
        },
        "input_layout": "public native raw CTA paired with GeoPose native-grid carotid mask",
        "operations": ["validated grid", "validated labels 1 and 2", "byte-for-byte copy"],
        "subjects": records,
    }
    (output_root / "public_isles_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return manifest


def run_hd_bet(
    standardized_root: Path,
    *,
    executable: str = "hd-bet",
) -> None:
    """Run the exact HD-BET 2.0.1 invocation used for alignedv2."""
    image_dir = standardized_root / "CTATr"
    if not image_dir.is_dir():
        raise FileNotFoundError(image_dir)
    output_dir = standardized_root / "brainmasks_Tr"
    command = [
        executable,
        "-i",
        str(image_dir),
        "-o",
        str(output_dir),
        "--save_bet_mask",
        "--no_bet_image",
        "--verbose",
    ]
    subprocess.run(command, check=True)


def prepare_public_isles(
    isles_root: Path,
    carotid_root: Path,
    standardized_root: Path,
    *,
    hd_bet_executable: str = "hd-bet",
    skip_hd_bet: bool = False,
    overwrite: bool = False,
    selected_subjects: set[str] | None = None,
) -> dict:
    manifest = stage_public_isles(
        isles_root,
        carotid_root,
        standardized_root,
        overwrite=overwrite,
        selected_subjects=selected_subjects,
    )
    if not skip_hd_bet:
        run_hd_bet(standardized_root, executable=hd_bet_executable)
    return manifest
