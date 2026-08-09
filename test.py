#!/usr/bin/env python3
"""Run calibrated GeoPose, greedy refinement, and 25-step test-time optimization."""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from geopose.contracts import verify_example_bundle
from geopose.inference import main


def _verify_published_example(argv: list[str]) -> None:
    if "-h" in argv or "--help" in argv:
        return
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--patient", default="sub-stroke9999")
    parser.add_argument("--timestamp", default="pre")
    args, _ = parser.parse_known_args(argv)
    if args.data_root is not None:
        verify_example_bundle(args.data_root.resolve(), args.patient, args.timestamp)


if __name__ == "__main__":
    _verify_published_example(sys.argv[1:])
    main()
