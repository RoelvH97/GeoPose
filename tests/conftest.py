"""Test fixtures and an import guard for the publication package."""

from pathlib import Path

import geopose


ROOT = Path(__file__).resolve().parents[1]


def pytest_configure():
    """Fail loudly if another `geopose` distribution shadows this checkout.

    The research repository installs a package of the same name, so a stray
    entry on sys.path can silently make the whole suite test the wrong code.
    `pythonpath = ["src"]` in pyproject.toml puts this checkout first; this
    check confirms it actually won.
    """
    imported = Path(geopose.__file__).resolve()
    expected = ROOT / "src/geopose/__init__.py"
    if imported != expected:
        raise RuntimeError(
            f"Tests imported geopose from {imported}, expected {expected}. "
            "Another geopose installation is shadowing this checkout."
        )
