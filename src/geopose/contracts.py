"""Small, dependency-free integrity checks for published release artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_MANIFEST = REPOSITORY_ROOT / "artifacts/example_sub-stroke9999.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_files(root: Path, files: dict[str, dict]) -> None:
    """Verify relative files against byte-size and SHA-256 records."""
    for relative_path, record in files.items():
        path = root / relative_path
        if not path.is_file():
            raise FileNotFoundError(path)
        actual_size = path.stat().st_size
        expected_size = int(record["bytes"])
        if actual_size != expected_size:
            raise RuntimeError(
                f"File size mismatch for {path}: {actual_size} != {expected_size}"
            )
        actual_hash = sha256(path)
        expected_hash = str(record["sha256"])
        if actual_hash != expected_hash:
            raise RuntimeError(
                f"SHA-256 mismatch for {path}: {actual_hash} != {expected_hash}"
            )


def verify_example_bundle(
    data_root: Path,
    patient: str,
    timestamp: str,
    *,
    manifest_path: Path = EXAMPLE_MANIFEST,
) -> None:
    """Verify the exact Zenodo example when the published example is requested."""
    if (patient, timestamp) != ("sub-stroke9999", "pre"):
        return
    manifest = json.loads(manifest_path.read_text())
    if manifest["patient"] != patient or manifest["timestamp"] != timestamp:
        raise RuntimeError(f"Example manifest identity mismatch: {manifest_path}")
    verify_files(data_root, manifest["files"])
