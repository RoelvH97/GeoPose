#!/usr/bin/env python3
"""Create the alignedv2 GeoPose training cohort."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from geopose.preregistration import main


if __name__ == "__main__":
    main()
