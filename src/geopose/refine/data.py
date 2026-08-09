"""Synthetic training data for GeoPose-Refine."""

from __future__ import annotations

import os

import pytorch_lightning as pl
import torch
from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.pose import convert
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, RandomSampler

from geopose.shared.pose import delta_to_pose

from ..data.augmentations import DRRAugmentations
from ..init.data import _list_image_paths
from ..data.splits import split_indices
from ..data.fiducials import carotid_skeleton_world


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

        mask_subdir = f"carotis_{align_suffix}" if dataset == "cta" else "labels_alignedTr"
        mask_dir = os.path.join(d.data_root, mask_subdir)
        self.art_end = 3 if dataset == "cta" else None

        fid_cfg = d.get("fiducials", {})
        fiducials_enabled = bool(fid_cfg.get("enabled", False)) and dataset == "cta"

        self.drrs: list[DRR] = []
        self.fiducials: list[torch.Tensor | None] = []
        for p in paths:
            pid = os.path.basename(p).replace("_0000.nii.gz", "")
            mask_path = os.path.join(mask_dir, f"{pid}.nii.gz")

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
        """Sample independent view-conditioned ground-truth poses."""
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

        is_lat = torch.rand(n, device=self.device, generator=generator) < float(p.lateral_prob)
        side_draw = torch.rand(n, device=self.device, generator=generator)
        lat_side = torch.where(side_draw < 0.5, -1, 1)
        view_label = torch.where(is_lat, lat_side + 1,
                                 torch.ones(n, dtype=torch.long, device=self.device))
        sign = torch.tensor([-1.0, 0.0, 1.0], device=self.device)[view_label]
        R[:, 0] += sign * (torch.pi / 2)
        pose = convert(R, t, parameterization=p.parameterization, convention=p.convention)
        lateral_mask = view_label != 1
        return pose, lateral_mask, view_label

    def __getitem__(self, idx):
        drr = self.drrs[idx].to(self.device)
        K = self.batch_size
        generator = None
        if not self.training and self.deterministic_eval:
            generator = torch.Generator(device=self.device)
            generator.manual_seed(self.eval_seed + int(idx))

        art_end = self.art_end if self.art_end is not None else int(drr.mask.max().item()) + 1

        optimal_pose, lateral_mask, view_label = self._sample_gt_pose_batch(
            K, generator=generator
        )

        ch1 = drr(optimal_pose)
        if ch1.shape[1] > 1:
            ch1 = ch1.sum(dim=1, keepdim=True)
        ch1 = self._normalize(ch1)
        if self.training:
            ch1 = self.drr_aug(ch1)

        std_R = torch.where(lateral_mask[:, None],
                            torch.tensor(self._std_R["lat"], device=self.device),
                            torch.tensor(self._std_R["pa"], device=self.device))
        std_t = torch.where(lateral_mask[:, None],
                            torch.tensor(self._std_t["lat"], device=self.device),
                            torch.tensor(self._std_t["pa"], device=self.device))
        dR = torch.randn(K, 3, device=self.device, generator=generator) * std_R
        dt = torch.randn(K, 3, device=self.device, generator=generator) * std_t
        noisy_pose = optimal_pose.compose(delta_to_pose(dR, dt))
        ch2 = drr(noisy_pose)
        if ch2.shape[1] > 1:
            ch2 = ch2.sum(dim=1, keepdim=True)
        ch2 = self._normalize(ch2)
        drr.cpu()

        item = {
            "drr": drr,
            "map": ch1,
            "drr_noisy": ch2,
            "noisy_pose": noisy_pose,
            "optimal_pose": optimal_pose,
            "true_dR": dR,
            "true_dt": dt,
            "lateral_mask": lateral_mask,
            "view_label": view_label,
            "art_end": art_end,
            "patient_id": f"synth{idx}", "view": "mixed", "timestamp": "-", "channel": "-",
        }

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

            paths = _list_image_paths(d.data_root, dataset, align_suffix)
            train_idx, val_idx, test_idx = split_indices(paths, d)
            if d.max_subjects is not None:
                train_idx = train_idx[: d.max_subjects]
            self.train_dataset = SyntheticRefineDataset(self.cfg, train_idx, training=True)
            self.val_dataset = SyntheticRefineDataset(self.cfg, val_idx, training=False)
            self.test_dataset = SyntheticRefineDataset(self.cfg, test_idx, training=False)
            return

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


