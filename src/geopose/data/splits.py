"""Single source of truth for the ISLES2024 CTA patient split.

Every datamodule used to re-derive its own split with ``int(n * train_fraction)`` over
the sorted image list. That is fine for TopBrain/TopCoW, whose subjects have no real
DSA, but on the CTA cohort it produced a split that disagreed with the refine manifests'
independent shuffle -- 11 of 12 refine-val patients were in the init network's train
set. ``split_file`` pins the assignment to an explicit per-patient list
(``assets/splits/isles_split_v1.json``, built by scripts/data_prep/build_isles_split.py)
so the whole cascade shares one partition; without it the historical fractional
behaviour is preserved unchanged.
"""

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
    """(train, val, test) indices into `image_paths`.

    With ``cfg.split_file`` the assignment comes from that file and the cohort must
    match it exactly, so a data-root change or a stale split file fails loudly instead
    of quietly reassigning patients. Otherwise the indices are the historical
    contiguous ``train_fraction``/``val_fraction`` slices.
    """
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
