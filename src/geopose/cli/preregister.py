"""CLI for ISLES staging and alignedv2 preregistration."""

from __future__ import annotations

import argparse
from pathlib import Path

from ..data.preparation import prepare_public_isles
from ..registration.preregistration import register_cohort


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
FROZEN_SPLIT = REPOSITORY_ROOT / "assets/isles_split_v1.json"


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
