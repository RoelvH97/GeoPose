"""FireANTs preregistration that creates the alignedv2 cohort."""

from __future__ import annotations

import json
import re
from pathlib import Path

import torch
from fireants.io import BatchedImages, Image
from fireants.registration.affine import AffineRegistration
from fireants.registration.moments import MomentsRegistration
from tqdm import tqdm


HU_SHIFT = 1000


def closest_rotation(linear: torch.Tensor) -> torch.Tensor:
    """Return the closest proper rotation by SVD polar decomposition."""
    left, _, right_t = torch.linalg.svd(linear)
    determinant = torch.linalg.det(left @ right_t)
    signs = torch.ones_like(linear[..., 0])
    signs[..., -1] = determinant
    return left @ torch.diag_embed(signs) @ right_t


def rigidify_affine(registration: AffineRegistration) -> None:
    """Drop scale and shear while preserving translation."""
    with torch.no_grad():
        linear = registration.affine.data[:, :, :3]
        registration.affine.data[:, :, :3] = closest_rotation(linear)


def patient_id(path: Path) -> str:
    return re.sub(r"_0000\.nii\.gz$", "", path.name)


def discover_subjects(source_root: Path) -> dict[str, dict[str, Path]]:
    images = source_root / "CTATr"
    carotids = source_root / "CTA_carotisTr"
    masks = source_root / "brainmasks_Tr"
    subjects: dict[str, dict[str, Path]] = {}
    for image in sorted(images.glob("sub-stroke*_0000.nii.gz")):
        pid = patient_id(image)
        carotid = carotids / f"{pid}.nii.gz"
        mask = masks / f"{pid}_0000_bet.nii.gz"
        if carotid.is_file() and mask.is_file():
            subjects[pid] = {"image": image, "carotid": carotid, "mask": mask}
    return subjects


def _outputs(output_root: Path, pid: str) -> dict[str, Path]:
    return {
        "image": output_root / "images_alignedv2" / f"{pid}_0000.nii.gz",
        "carotid": output_root / "carotis_alignedv2" / f"{pid}.nii.gz",
        "mask": output_root / "masks_alignedv2" / f"{pid}.nii.gz",
        "transform": output_root / "transforms_alignedv2" / f"{pid}.pt",
        "metadata": output_root / "transforms_alignedv2" / f"{pid}.json",
    }


def register_cohort(
    source_root: Path,
    output_root: Path,
    *,
    fixed_subject: str = "sub-stroke0001",
    limit: int = 0,
    selected_subjects: set[str] | None = None,
    overwrite: bool = False,
) -> None:
    """Run the exact alignedv2 COM-initialized, rigidified FireANTs method."""
    if not torch.cuda.is_available():
        raise RuntimeError("alignedv2 preregistration requires a CUDA-capable GPU")

    subjects = discover_subjects(source_root)
    if selected_subjects is not None:
        missing = selected_subjects - set(subjects)
        if missing:
            raise FileNotFoundError(f"Missing requested input subjects: {sorted(missing)}")
        subjects = {pid: subjects[pid] for pid in sorted(selected_subjects)}
    if limit:
        subjects = dict(list(subjects.items())[:limit])
    if not subjects:
        raise FileNotFoundError(
            "No complete CTATr/CTA_carotisTr/brainmasks_Tr subject triples found"
        )

    fixed_image_path = source_root / "CTATr" / f"{fixed_subject}_0000.nii.gz"
    fixed_mask_path = source_root / "brainmasks_Tr" / f"{fixed_subject}_0000_bet.nii.gz"
    for required in (fixed_image_path, fixed_mask_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    for subdirectory in (
        "images_alignedv2",
        "carotis_alignedv2",
        "masks_alignedv2",
        "transforms_alignedv2",
    ):
        (output_root / subdirectory).mkdir(parents=True, exist_ok=True)

    fixed_image = Image.load_file(str(fixed_image_path))
    fixed_mask = Image.load_file(str(fixed_mask_path))
    fixed_image_batch = BatchedImages([fixed_image])
    fixed_mask_batch = BatchedImages([fixed_mask])

    for pid, paths in tqdm(subjects.items(), desc="alignedv2 preregistration"):
        outputs = _outputs(output_root, pid)
        required_outputs = ("image", "carotid", "mask", "transform", "metadata")
        if not overwrite and all(outputs[key].is_file() for key in required_outputs):
            continue

        moving_image = Image.load_file(str(paths["image"]))
        moving_carotid = Image.load_file(str(paths["carotid"]))
        moving_carotid.interpolation_mode = "nearest"
        moving_mask = Image.load_file(str(paths["mask"]))
        moving_image.array = moving_image.array + HU_SHIFT

        moving_image_batch = BatchedImages([moving_image])
        moving_carotid_batch = BatchedImages([moving_carotid])
        moving_mask_batch = BatchedImages([moving_mask])

        moments = MomentsRegistration(
            scale=4,
            fixed_images=fixed_mask_batch,
            moving_images=moving_mask_batch,
            moments=1,
            transl_mode="com",
        )
        moments.optimize()

        registration = AffineRegistration(
            scales=[8, 4, 2, 1],
            iterations=[200, 200, 100, 50],
            fixed_images=fixed_mask_batch,
            moving_images=moving_mask_batch,
            optimizer="Adam",
            optimizer_lr=3e-3,
            loss_type="mse",
            init_rigid=moments.get_affine_init(),
        )
        registration.optimize()
        rigidify_affine(registration)

        moved_image = registration.evaluate(fixed_image_batch, moving_image_batch) - HU_SHIFT
        moved_carotid = registration.evaluate(fixed_image_batch, moving_carotid_batch)
        moving_mask.interpolation_mode = "nearest"
        nearest_mask_batch = BatchedImages([moving_mask])
        moved_mask = registration.evaluate(fixed_image_batch, nearest_mask_batch)

        registration.save_moved_images(moved_image, str(outputs["image"]))
        registration.save_moved_images(moved_carotid, str(outputs["carotid"]))
        registration.save_moved_images(moved_mask, str(outputs["mask"]))
        affine = registration.affine.detach().cpu()
        torch.save(affine, outputs["transform"])
        outputs["metadata"].write_text(
            json.dumps(
                {
                    "contract": "alignedv2-v1",
                    "subject": pid,
                    "fixed_subject": fixed_subject,
                    "moments": 1,
                    "translation_mode": "com",
                    "scales": [8, 4, 2, 1],
                    "iterations": [200, 200, 100, 50],
                    "optimizer": "Adam",
                    "optimizer_lr": 0.003,
                    "loss": "mse",
                    "rigidification": "SVD polar decomposition",
                    "image_hu_shift": HU_SHIFT,
                    "label_interpolation": "nearest",
                },
                indent=2,
            )
            + "\n"
        )


