"""FireANTs preregistration used to create the published alignedv2 cohort."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from fireants.io import BatchedImages, Image
from fireants.registration.affine import AffineRegistration
from fireants.registration.moments import MomentsRegistration
from tqdm import tqdm

from .data_preparation import prepare_public_isles


HU_SHIFT = 1000.0
FROZEN_SPLIT = Path(__file__).resolve().parents[2] / "assets" / "isles_split_v1.json"


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
        moved_mask = registration.evaluate(fixed_image_batch, moving_mask_batch)

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


def _common_prepare_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--isles-root", required=True, type=Path)
    parser.add_argument(
        "--carotid-root",
        required=True,
        type=Path,
        help="Extracted GeoPose Zenodo carotid-segmentation directory.",
    )
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--hd-bet-executable", default="hd-bet")
    parser.add_argument(
        "--cohort-file",
        type=Path,
        default=FROZEN_SPLIT,
        help="Frozen 99-subject publication split; pass another JSON deliberately.",
    )
    parser.add_argument("--skip-hd-bet", action="store_true")
    parser.add_argument("--overwrite", action="store_true")


def _common_align_arguments(parser: argparse.ArgumentParser, source_required: bool) -> None:
    if source_required:
        parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--fixed-subject", default="sub-stroke0001")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--subjects", nargs="+")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare public ISLES'24 CTA/ICA data and create the frozen "
            "images/carotis/masks_alignedv2 cohort."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser(
        "prepare", help="Pair public CTA with GeoPose carotids and run HD-BET 2.0.1."
    )
    _common_prepare_arguments(prepare)

    align = commands.add_parser(
        "align", help="Run the exact FireANTs alignedv2 preregistration."
    )
    _common_align_arguments(align, source_required=True)
    align.add_argument("--overwrite", action="store_true")

    complete = commands.add_parser(
        "all", help="Prepare public ISLES data, run HD-BET, then preregister."
    )
    _common_prepare_arguments(complete)
    _common_align_arguments(complete, source_required=False)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    source_root = args.source_root.expanduser().resolve()

    if args.command in ("prepare", "all"):
        split = json.loads(args.cohort_file.expanduser().resolve().read_text())
        cohort = set(split["train"]) | set(split["val"]) | set(split["test"])
        prepare_public_isles(
            args.isles_root.expanduser().resolve(),
            args.carotid_root.expanduser().resolve(),
            source_root,
            hd_bet_executable=args.hd_bet_executable,
            skip_hd_bet=args.skip_hd_bet,
            overwrite=args.overwrite,
            selected_subjects=cohort,
        )

    if args.command in ("align", "all"):
        register_cohort(
            source_root,
            args.output_root.expanduser().resolve(),
            fixed_subject=args.fixed_subject,
            limit=args.limit,
            selected_subjects=set(args.subjects) if args.subjects else None,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()





