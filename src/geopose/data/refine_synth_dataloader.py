"""Synthetic data path for the standalone refiner — direction (B).

Instead of a manifest of real DSA + GeoReg poses (see
:mod:`geopose.data.refine_dataloader`), this generates training pairs *fully
synthetically* from the CTA volumes, so net2 gets dense, clean, direct
δ-supervision and cannot collapse to identity:

  * sample a GT ("optimal") pose like the GeoPose pipeline (view-conditional
    anchor, ``cfg.data.pose`` noise),
  * render it and **intensity-augment** → channel 1 (the "input"/target view;
    the LXPose-style augs stand in for the real-DSA appearance),
  * sample ``K`` pose deviations δ (view-conditional std, Phase-9 calibrated),
  * render the deviated poses **clean** → channel 2,
  * the refine net regresses δ — supervised via corrected-vs-GT geodesic.

Returns the exact batch dict :class:`RefinePoseModule` consumes, so the model,
criterion, and logging are reused unchanged. One item per patient;
``RandomSampler(num_samples=epoch_len)`` mixes patients across an epoch and each
item is a batch of ``batch_size`` independent, mixed-view poses (the BatchNorm
mini-batch — see docs/refinement.html#bn-batch-diversity).
"""

from __future__ import annotations

import os

import pytorch_lightning as pl
import torch
from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.pose import convert
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, RandomSampler

from geopose.models.pose_utils import delta_to_pose

from .augmentations import DRRAugmentations
from .dataloader import _list_image_paths
from .splits import split_indices
from .skeleton_fiducials import carotid_skeleton_world


def _passthrough_collate(batch):
    """Unwrap an item that already contains a complete pose mini-batch."""
    if len(batch) != 1:
        raise ValueError(f"Expected DataLoader batch_size=1, got {len(batch)}")
    return batch[0]


class SyntheticRefineDataset(Dataset):
    """One item per patient; each ``__getitem__`` samples a fresh GT pose and K δ."""

    def __init__(self, cfg: DictConfig, indices: list[int], training: bool):
        self.cfg = cfg
        self.training = training
        d = cfg.data
        self.device = torch.device(d.device)
        self.batch_size = int(cfg.refine.batch_size)
        self.pose = d.pose
        self.deterministic_eval = bool(cfg.refine.get("deterministic_eval", True))
        self.eval_seed = int(cfg.refine.get("eval_seed", cfg.get("seed", 0)))

        dataset = getattr(d, "dataset", "cta")
        align_suffix = getattr(d, "align_suffix", "alignedTr")
        paths = _list_image_paths(d.data_root, dataset, align_suffix)
        paths = [paths[i] for i in indices]
        if not paths:
            raise ValueError("SyntheticRefineDataset got an empty index list")
        # CTA uses carotid masks (carotis_alignedTr) and supervises the carotid
        # channels 1:3; TopCoW-style datasets (topbrain/topcow) use the full
        # vasculature labelmap (labels_alignedTr) and supervise the union of all
        # vessels. art_end (Dice slice end) is 3 for CTA, else per-subject =
        # mask.max()+1 (resolved in __getitem__ from the DRR's channel count).
        mask_subdir = f"carotis_{align_suffix}" if dataset == "cta" else "labels_alignedTr"
        mask_dir = os.path.join(d.data_root, mask_subdir)
        self.art_end = 3 if dataset == "cta" else None

        # Carotid-skeleton fiducials for the LXPose projection (mPD) loss — same
        # block CTAPoseDataset reads (data.fiducials), so the two paths share the
        # .npy cache. Off by default ⇒ no "fiducials" in the batch ⇒ the criterion
        # skips the term ⇒ existing runs are byte-identical. CTA-only.
        fid_cfg = d.get("fiducials", {})
        fiducials_enabled = bool(fid_cfg.get("enabled", False)) and dataset == "cta"

        self.drrs: list[DRR] = []
        self.fiducials: list[torch.Tensor | None] = []
        for p in paths:
            pid = os.path.basename(p).replace("_0000.nii.gz", "")
            mask_path = os.path.join(mask_dir, f"{pid}.nii.gz")
            # Skeletonise BEFORE read() so read's canonicalize reorients the points
            # into the DRR's centered frame (drr.perspective_projection's frame).
            fid_world = (
                carotid_skeleton_world(
                    mask_path, labels=tuple(fid_cfg.get("labels", (1, 2))),
                    max_points=fid_cfg.get("max_points", 128),
                    cache_dir=fid_cfg.get("cache_dir", None),
                )
                if fiducials_enabled else None
            )
            subject = read(p, mask_path, labels=list(d.drr.labels), fiducials=fid_world)
            self.drrs.append(DRR(
                subject, sdd=d.drr.sdd, height=d.drr.height, delx=d.drr.delx,
                stop_gradients_through_grid_sample=d.drr.stop_gradients,
            ))
            self.fiducials.append(getattr(subject, "fiducials", None))

        rc = cfg.refine
        self._std_R = {"lat": list(rc.delta_std_R_lat), "pa": list(rc.delta_std_R_pa)}
        self._std_t = {"lat": list(rc.delta_std_t_lat), "pa": list(rc.delta_std_t_pa)}

        aug = d.augmentation
        self.drr_aug = DRRAugmentations(
            p=aug.p, max_crop=aug.max_crop,
            clahe_clip=tuple(aug.drr_clahe_clip), gamma_range=tuple(aug.drr_gamma_range),
            noise_std=aug.drr_noise_std,
            use_plasma=bool(getattr(aug, "use_plasma", False)),
            plasma_roughness=tuple(getattr(aug, "plasma_roughness", (0.1, 0.5))),
            plasma_brightness_intensity=tuple(getattr(aug, "plasma_brightness_intensity", (-1.0, 1.0))),
            plasma_p=float(getattr(aug, "plasma_p", 1.0)),
        ).to(self.device)

    def __len__(self) -> int:
        return len(self.drrs)

    def _sample_gt_pose_batch(self, n: int, generator=None):
        """Sample ``n`` independent view-conditional GT poses in one shot —
        identical scheme to :meth:`CTAPoseDataset.get_random_pose_batch` (50% PA,
        25% LAT-, 25% LAT+). Returns ``(pose [n,4,4], lateral_mask [n] bool,
        view_label [n] long)`` where ``view_label`` is the GeoPose 3-way convention
        (0=LAT-, 1=PA, 2=LAT+) — the left/right lateral distinction the refiner's
        3-way view embedding conditions on.

        Sampling a *batch* of independent, mixed-view poses (rather than K
        copies of one view) is what gives each batch element its own distinct
        render — so the backbone's BatchNorm sees a diverse mini-batch and its
        running stats generalise to validation, exactly as the primary GeoPose
        net does. See docs/refinement.html#bn-batch-diversity.
        """
        p = self.pose
        t_y_std = float(getattr(p, "t_y_std", p.t_std))
        t_scale = torch.tensor(
            [float(p.t_std), t_y_std, float(p.t_std)], device=self.device
        )
        t = torch.randn(
            n, 3, device=self.device, generator=generator
        ) * t_scale
        t[:, 1] += float(p.t_y_offset)
        R = torch.randn(n, 3, device=self.device, generator=generator) * float(p.r_std)
        # view_label 0/1/2 (LAT-/PA/LAT+) + signed ±π/2 anchor — mirrors get_random_pose_batch.
        is_lat = torch.rand(n, device=self.device, generator=generator) < float(p.lateral_prob)
        side_draw = torch.rand(n, device=self.device, generator=generator)
        lat_side = torch.where(side_draw < 0.5, -1, 1)                              # ±1
        view_label = torch.where(is_lat, lat_side + 1,
                                 torch.ones(n, dtype=torch.long, device=self.device))  # {0,1,2}
        sign = torch.tensor([-1.0, 0.0, 1.0], device=self.device)[view_label]
        R[:, 0] += sign * (torch.pi / 2)
        pose = convert(R, t, parameterization=p.parameterization, convention=p.convention)
        lateral_mask = view_label != 1
        return pose, lateral_mask, view_label

    def __getitem__(self, idx):
        drr = self.drrs[idx].to(self.device)
        K = self.batch_size                                          # batch / BN dimension
        generator = None
        if not self.training and self.deterministic_eval:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.eval_seed + int(idx))
        # Dice channel slice end: carotid 1:3 for CTA, else all vessels present in
        # this subject (mask.max()+1 = its mask_to_channels channel count).
        art_end = self.art_end if self.art_end is not None else int(drr.mask.max().item()) + 1

        # K independent, mixed-view GT poses (one distinct render each) — the
        # GeoPose batching, so BatchNorm sees a diverse batch (not K copies).
        optimal_pose, lateral_mask, view_label = self._sample_gt_pose_batch(
            K, generator=generator
        )  # [K,4,4], [K], [K]

        # ── Channel 1: render @ each GT pose, intensity-augmented (the "input" view) ──
        ch1 = drr(optimal_pose)
        if ch1.shape[1] > 1:
            ch1 = ch1.sum(dim=1, keepdim=True)
        ch1 = self._normalize(ch1)
        if self.training:
            ch1 = self.drr_aug(ch1)                                  # [K,1,H,W] independent aug

        # ── Channel 2: per-element δ (view-conditional std), rendered clean ──
        std_R = torch.where(lateral_mask[:, None],
                            torch.tensor(self._std_R["lat"], device=self.device),
                            torch.tensor(self._std_R["pa"], device=self.device))   # [K,3]
        std_t = torch.where(lateral_mask[:, None],
                            torch.tensor(self._std_t["lat"], device=self.device),
                            torch.tensor(self._std_t["pa"], device=self.device))   # [K,3]
        dR = torch.randn(K, 3, device=self.device, generator=generator) * std_R
        dt = torch.randn(K, 3, device=self.device, generator=generator) * std_t
        noisy_pose = optimal_pose.compose(delta_to_pose(dR, dt))     # noisy = GT ∘ δ
        ch2 = drr(noisy_pose)
        if ch2.shape[1] > 1:
            ch2 = ch2.sum(dim=1, keepdim=True)
        ch2 = self._normalize(ch2)
        drr.cpu()

        item = {
            "drr": drr,                    # patient's DRR (CPU); module re-renders corrected
            "map": ch1,                    # [K, 1, H, W]  channel 1 (aug render @ GT, distinct per element)
            "drr_noisy": ch2,              # [K, 1, H, W]  channel 2 (clean render @ noisy)
            "noisy_pose": noisy_pose,      # RigidTransform [K, 4, 4]
            "optimal_pose": optimal_pose,  # RigidTransform [K, 4, 4]  (the sampled GTs)
            "true_dR": dR,                 # [K, 3] axis-angle (rad)
            "true_dt": dt,                 # [K, 3] mm
            "lateral_mask": lateral_mask,  # [K]  binary PA/LAT (per-view logging split)
            "view_label": view_label,      # [K]  3-way 0=LAT-,1=PA,2=LAT+ (view embedding)
            "art_end": art_end,            # int  Dice channel slice end (cta=3; topbrain/topcow=N)
            "patient_id": f"synth{idx}", "view": "mixed", "timestamp": "-", "channel": "-",
        }
        # Skeleton fiducials ([1, N, 3] world coords in the DRR's centered frame);
        # present only when data.fiducials.enabled. The module projects them under
        # the corrected/optimal poses for the mPD term.
        if self.fiducials[idx] is not None:
            item["fiducials"] = self.fiducials[idx]
        return item

    @staticmethod
    def _normalize(images: torch.Tensor) -> torch.Tensor:
        N, C, H, W = images.shape
        flat = images.reshape(N * C, -1)
        vmin = flat.min(dim=1).values.view(N, C, 1, 1)
        vmax = flat.max(dim=1).values.view(N, C, 1, 1)
        return (images - vmin) / (vmax - vmin).clamp(min=1e-8)


class SyntheticRefineDataModule(pl.LightningDataModule):
    """Patient-split synthetic refiner data (mirrors CTAPoseDataModule's split)."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):
        d = self.cfg.data
        dataset = getattr(d, "dataset", "cta")
        align_suffix = getattr(d, "align_suffix", "alignedTr")
        val_override = d.get("val", None)

        if val_override is None:
            # single-dataset split (e.g. the CTA refiner)
            paths = _list_image_paths(d.data_root, dataset, align_suffix)
            train_idx, val_idx, test_idx = split_indices(paths, d)
            if d.max_subjects is not None:
                train_idx = train_idx[: d.max_subjects]
            self.train_dataset = SyntheticRefineDataset(self.cfg, train_idx, training=True)
            self.val_dataset = SyntheticRefineDataset(self.cfg, val_idx, training=False)
            self.test_dataset = SyntheticRefineDataset(self.cfg, test_idx, training=False)
            return

        # Cross-dataset: train the refiner on ALL of the main dataset (topbrain/
        # topcow), validate/test on the CTA split — same held-out CTA subjects as
        # a native CTA refiner run, so the new runs are directly comparable.
        train_idx = list(range(len(_list_image_paths(d.data_root, dataset, align_suffix))))
        if d.max_subjects is not None:
            train_idx = train_idx[: d.max_subjects]
        val_cfg = OmegaConf.merge(self.cfg, {"data": val_override})
        vd = val_cfg.data
        vt_paths = _list_image_paths(vd.data_root, vd.dataset, getattr(vd, "align_suffix", "alignedTr"))
        _, val_idx, test_idx = split_indices(vt_paths, vd)
        self.train_dataset = SyntheticRefineDataset(self.cfg, train_idx, training=True)
        self.val_dataset = SyntheticRefineDataset(val_cfg, val_idx, training=False)
        self.test_dataset = SyntheticRefineDataset(val_cfg, test_idx, training=False)

    def _make_loader(self, dataset, *, shuffle: bool, sampler=None):
        return DataLoader(
            dataset, batch_size=1, sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            num_workers=self.cfg.data.num_workers, collate_fn=_passthrough_collate,
        )

    def train_dataloader(self):
        generator = torch.Generator()
        generator.manual_seed(int(self.cfg.get("seed", 0)))
        sampler = RandomSampler(
            self.train_dataset, replacement=True,
            num_samples=self.cfg.data.epoch_len, generator=generator,
        )
        return self._make_loader(self.train_dataset, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._make_loader(self.test_dataset, shuffle=False)


