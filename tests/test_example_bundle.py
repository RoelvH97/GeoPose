import hashlib
import json
from pathlib import Path

import pytest

from geopose.contracts import verify_example_bundle, verify_files


ROOT = Path(__file__).resolve().parents[1]


def _record(content: bytes) -> dict:
    return {
        "bytes": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def test_verify_files_accepts_exact_content_and_rejects_tampering(tmp_path):
    content = b"published example"
    path = tmp_path / "DSATr/example.nii.gz"
    path.parent.mkdir()
    path.write_bytes(content)
    files = {"DSATr/example.nii.gz": _record(content)}
    verify_files(tmp_path, files)

    path.write_bytes(content + b"!")
    with pytest.raises(RuntimeError, match="size mismatch"):
        verify_files(tmp_path, files)


def test_example_verification_is_scoped_to_published_example(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "patient": "sub-stroke9999",
                "timestamp": "pre",
                "files": {"missing.nii.gz": _record(b"x")},
            }
        )
    )
    verify_example_bundle(
        tmp_path,
        "sub-stroke0001",
        "pre",
        manifest_path=manifest,
    )
    with pytest.raises(FileNotFoundError):
        verify_example_bundle(
            tmp_path,
            "sub-stroke9999",
            "pre",
            manifest_path=manifest,
        )


def test_published_example_manifest_is_complete():
    manifest = json.loads(
        (ROOT / "artifacts/example_sub-stroke9999.json").read_text()
    )
    assert manifest["schema_version"] == 2
    assert manifest["zenodo_doi"] is None
    assert len(manifest["files"]) == 10
    assert sum(record["bytes"] for record in manifest["files"].values()) == (
        manifest["bundle_bytes"]
    )
    assert all(
        len(record["sha256"]) == 64 for record in manifest["files"].values()
    )
