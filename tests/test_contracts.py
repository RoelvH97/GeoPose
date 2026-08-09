import hashlib
import json
from pathlib import Path

from geopose.training import load_training_contract


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_training_contracts():
    init = load_training_contract("init")
    refine = load_training_contract("refine")

    assert init.model.backbone == "resnet34"
    assert init.model.view_role_emb_classes == 3
    assert init.model.dsa_view_anchor_mode == "known_role_predicted_side"
    assert init.model.lambda_dice == 0.1
    assert init.model.lambda_proj == 0.1
    assert init.trainer.max_epochs == 400

    assert refine.model.architecture == "pooled_late_fusion"
    assert list(refine.model.delta_R_scale) == [0.13, 0.1, 0.14]
    assert list(refine.model.delta_t_scale) == [4.0, 33.0, 5.0]
    assert refine.refine.lambda_dice == 0.1
    assert refine.refine.lambda_proj == 0.1
    assert refine.seed == 0


def test_checkpoint_and_split_manifests():
    artifacts = json.loads((ROOT / "artifacts/checkpoints.json").read_text())
    assert artifacts["files"]["geopose_init.ckpt"]["sha256"] == (
        "ba25e34b48bb75124ecd0a5bb402efb3373af7189b8364346dacf073c04abfa0"
    )
    assert artifacts["files"]["geopose_refine.ckpt"]["sha256"] == (
        "52d8aa8cb89c9e0ec185f65cd35fc7ea25079f130d8f749f95935cb52da92da2"
    )

    split = json.loads((ROOT / "assets/isles_split_v1.json").read_text())
    assert len(split["train"]) == 69
    assert len(split["val"]) == 10
    assert len(split["test"]) == 20
    all_ids = split["train"] + split["val"] + split["test"]
    assert len(all_ids) == len(set(all_ids)) == 99
    assert "sub-stroke9999" not in all_ids


def test_release_source_hashes_match_provenance():
    manifest = json.loads((ROOT / "artifacts/source_provenance.json").read_text())
    for record in manifest["files"]:
        source = ROOT / record["path"]
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        assert digest == record["release_sha256"], source
