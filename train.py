#!/usr/bin/env python3
"""Train the frozen GeoPose-Init or GeoPose-Refine publication recipe."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from geopose.cli.train import main


if __name__ == "__main__":
    main()
