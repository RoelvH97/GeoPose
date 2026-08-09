#!/usr/bin/env python3
"""Run calibrated GeoPose, greedy refinement, and 25-step test-time optimization."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from geopose.cli.test import main


if __name__ == "__main__":
    main()
