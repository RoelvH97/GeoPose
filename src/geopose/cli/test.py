"""CLI for calibrated GeoPose + Refine + 25-step GeoReg inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..registration.pipeline import run_inference
from ..shared.contracts import PACKAGED_EXAMPLE, verify_example_bundle


def _existing_file(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"File does not exist: {path}")
    return path


def _existing_directory(value: str) -> Path:
    path = Path(value).expanduser().resolve()
    if not path.is_dir():
        raise argparse.ArgumentTypeError(f"Directory does not exist: {path}")
    return path


def _output_directory(value: str) -> Path:
    return Path(value).expanduser().resolve()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run calibrated GeoPose + greedy Refine + 25-step GeoReg TTO."
    )
    parser.add_argument("--data-root", required=True, type=_existing_directory)
    parser.add_argument("--patient", default="sub-stroke0011")
    parser.add_argument("--timestamp", choices=("pre", "post"), default="pre")
    parser.add_argument(
        "--projection-file",
        type=_existing_file,
        help="Optional model-ready 256x256 projections; otherwise read full DSA data.",
    )
    parser.add_argument("--init-checkpoint", required=True, type=_existing_file)
    parser.add_argument("--refine-checkpoint", required=True, type=_existing_file)
    parser.add_argument("--output-dir", required=True, type=_output_directory)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--max-refine-updates", type=int, default=5)
    parser.add_argument(
        "--determinism",
        choices=("off", "warn", "error"),
        default="warn",
        help=(
            "PyTorch deterministic-algorithm policy. 'warn' is the portable "
            "default for third-party CUDA kernels; 'error' rejects unsupported ops."
        ),
    )
    parser.add_argument(
        "--skip-hash-check",
        action="store_true",
        help="Allow non-publication checkpoints, such as newly retrained weights.",
    )
    return parser


def _resolve_projection_file(args) -> Path | None:
    if args.projection_file is not None:
        return args.projection_file
    candidate = (
        args.data_root / "ProjectionTr" / f"{args.patient}_{args.timestamp}.npz"
    )
    if candidate.is_file():
        return candidate.resolve()
    if (args.patient, args.timestamp) == ("sub-stroke0011", "pre"):
        return PACKAGED_EXAMPLE.resolve()
    return None


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.iterations < 0:
        raise ValueError("--iterations must be nonnegative")
    if args.max_refine_updates < 0:
        raise ValueError("--max-refine-updates must be nonnegative")
    args.projection_file = _resolve_projection_file(args)
    verify_example_bundle(args.projection_file, args.patient, args.timestamp)
    result = run_inference(args)
    print(json.dumps(result["final_pose"], indent=2))


if __name__ == "__main__":
    main()
