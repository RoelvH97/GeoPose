"""Single source of truth for the ISLES2024 CTA patient split."""

from __future__ import annotations

import json
import os

SPLITS = ("train", "val", "test")


def patient_id(image_path: str) -> str:
    """``.../images_alignedv2/sub-stroke0001_0000.nii.gz`` -> ``sub-stroke0001``."""
    return os.path.basename(image_path).replace("_0000.nii.gz", "")


def load_patient_split(path: str) -> dict[str, list[str]]:
    """Read + validate a split file: three non-empty, pairwise-disjoint id lists."""
    with open(path) as fh:
        data = json.load(fh)
    splits = {}
    for key in SPLITS:
        ids = data.get(key)
        if not isinstance(ids, list) or not ids:
            raise ValueError(f"{path} is missing a non-empty '{key}' list")
        splits[key] = list(ids)
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            overlap = sorted(set(splits[a]) & set(splits[b]))
            if overlap:
                raise ValueError(f"{path}: {a}/{b} overlap on {overlap}")
    return splits


def split_file_of(cfg) -> str | None:
    """``cfg.split_file`` if set to a real path, else None (fractional fallback)."""
    path = cfg.get("split_file", None)
    return str(path) if path else None


def train_patient_ids(cfg) -> set[str] | None:
    """Train-split ids for cfg, or None when cfg has no split file."""
    path = split_file_of(cfg)
    return set(load_patient_split(path)["train"]) if path else None


def split_indices(image_paths: list[str], cfg) -> tuple[list[int], list[int], list[int]]:
    """(train, val, test) indices into `image_paths`."""
    path = split_file_of(cfg)
    if path is None:
        n = len(image_paths)
        n_train = int(n * cfg.train_fraction)
        n_val = int(n * cfg.val_fraction)
        idx = list(range(n))
        return idx[:n_train], idx[n_train:n_train + n_val], idx[n_train + n_val:]

    splits = load_patient_split(path)
    index = {patient_id(p): i for i, p in enumerate(image_paths)}
    assigned = set().union(*(set(v) for v in splits.values()))
    unassigned = sorted(set(index) - assigned)
    absent = sorted(assigned - set(index))
    if unassigned or absent:
        raise ValueError(
            f"{path} does not describe this cohort: "
            f"{len(unassigned)} image(s) with no split assignment ({unassigned[:5]}), "
            f"{len(absent)} split id(s) with no image ({absent[:5]})"
        )
    return tuple(sorted(index[pid] for pid in splits[key]) for key in SPLITS)
