import hashlib
import json
from pathlib import Path

import pytest

from geopose.shared.contracts import verify_example_bundle, verify_files


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
    projection = tmp_path / "sub-stroke0011_pre.npz"
    projection.write_bytes(b"projection")
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "patient": "sub-stroke0011",
                "timestamp": "pre",
                "projection": _record(b"projection"),
            }
        )
    )
    verify_example_bundle(
        projection,
        "sub-stroke0001",
        "pre",
        manifest_path=manifest,
    )
    verify_example_bundle(
        projection,
        "sub-stroke0011",
        "pre",
        manifest_path=manifest,
    )
    projection.write_bytes(b"tampered")
    with pytest.raises(RuntimeError):
        verify_example_bundle(
            projection,
            "sub-stroke0011",
            "pre",
            manifest_path=manifest,
        )


def test_published_example_manifest_is_complete():
    manifest = json.loads(
        (ROOT / "src/geopose/artifacts/example_sub-stroke0011.json").read_text()
    )
    assert manifest["schema_version"] == 3
    # zenodo_doi is null until the deposit is minted; once set it must stay a DOI.
    doi = manifest["zenodo_doi"]
    assert doi is None or doi.startswith("10.")
    assert manifest["projection"]["bytes"] == manifest["bundle_bytes"]
    assert len(manifest["projection"]["sha256"]) == 64
    assert manifest["contains_raw_dsa"] is False
