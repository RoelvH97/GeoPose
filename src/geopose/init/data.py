"""Synthetic CTA training data for GeoPose-Init."""

import glob
import os

import pytorch_lightning as pl
import torch
from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform, convert
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset

from ..data.augmentations import DRRAugmentations
from ..data.fiducials import carotid_skeleton_world
from ..data.splits import split_file_of, split_indices


TEMPLATE_ID = "sub-stroke9999"


def _list_image_paths(data_root: str, dataset: str = "cta", align_suffix: str = "alignedTr") -> list[str]:
    """Sorted list of training image paths."""
    paths = sorted(glob.glob(os.path.join(data_root, f"images_{align_suffix}", "*_0000.nii.gz")))
    if dataset != "cta":
        return paths
    return [p for p in paths if os.path.basename(p).replace("_0000.nii.gz", "") != TEMPLATE_ID]


class CTAPoseDataset(Dataset):
    """Dataset for CTAPose: one item per subject."""

    def __init__(self, cfg: DictConfig, indices: list | None = None):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.pose_cfg = cfg.pose
        self.batch_size = cfg.batch_size
        self.dataset = getattr(cfg, "dataset", "cta")

        self.align_suffix = getattr(cfg, "align_suffix", "alignedTr")

        # CTA labels 1:3 are carotids; other cohorts use every vessel label.
        self.art_end = 3 if self.dataset == "cta" else None

        mask_dir = os.path.join(cfg.data_root, f"carotis_{self.align_suffix}") if self.dataset == "cta" else None

        all_image_paths = _list_image_paths(cfg.data_root, self.dataset, self.align_suffix)
        if not all_image_paths:
            raise FileNotFoundError(f"No *_0000.nii.gz files found in {cfg.data_root}/images_{self.align_suffix}")
        image_paths = [all_image_paths[i] for i in indices] if indices is not None else all_image_paths

        self.training = True
        aug = cfg.augmentation
        self.disable_drr_aug = bool(getattr(aug, "disable_drr_aug", False))
        self.drr_aug = DRRAugmentations(
            p=aug.p,
            max_crop=aug.max_crop,
            clahe_clip=tuple(aug.drr_clahe_clip),
            gamma_range=tuple(aug.drr_gamma_range),
            noise_std=aug.drr_noise_std,
            use_plasma=bool(getattr(aug, "use_plasma", False)),
            plasma_roughness=tuple(getattr(aug, "plasma_roughness", (0.1, 0.5))),
            plasma_brightness_intensity=tuple(getattr(aug, "plasma_brightness_intensity", (-1.0, 1.0))),
            plasma_p=float(getattr(aug, "plasma_p", 1.0)),
        ).to(self.device)

        fid_cfg = cfg.get("fiducials", {})
        self.fiducials_enabled = bool(fid_cfg.get("enabled", False))
        self._fid_labels = tuple(fid_cfg.get("labels", (1, 2)))
        self._fid_max_points = fid_cfg.get("max_points", 128)
        self._fid_cache_dir = fid_cfg.get("cache_dir", None)

        self.drrs = []
        self.fiducials = []
        for img_path in image_paths:
            patient_id = os.path.basename(img_path).replace("_0000.nii.gz", "")

            if self.dataset == "cta":
                mask_path = os.path.join(mask_dir, f"{patient_id}.nii.gz")
                if not os.path.exists(mask_path):
                    raise FileNotFoundError(f"Mask not found for {img_path}: expected {mask_path}")
                subject = read(img_path, mask_path, labels=list(cfg.drr.labels),
                               fiducials=self._skeleton_fiducials(mask_path))
            else:

                mask_path = os.path.join(cfg.data_root, "labels_alignedTr", f"{patient_id}.nii.gz")
                if not os.path.exists(mask_path):
                    raise FileNotFoundError(f"TopBrain label not found for {img_path}: expected {mask_path}")
                subject = read(img_path, mask_path, labels=list(cfg.drr.labels),
                               fiducials=self._skeleton_fiducials(mask_path))

            drr = DRR(subject, sdd=cfg.drr.sdd, height=cfg.drr.height, delx=cfg.drr.delx,
                      stop_gradients_through_grid_sample=cfg.drr.stop_gradients)
            self.drrs.append(drr)

            self.fiducials.append(getattr(subject, "fiducials", None))


    def _skeleton_fiducials(self, mask_path: str):
        """Skeleton fiducials for the mPD loss, or None when disabled."""
        if not self.fiducials_enabled:
            return None
        return carotid_skeleton_world(
            mask_path, labels=self._fid_labels,
            max_points=self._fid_max_points, cache_dir=self._fid_cache_dir,
        )


    def __len__(self):
        return len(self.drrs)

    def __getitem__(self, idx):
        drr = self.drrs[idx].to(self.device)

        poses, R, t, lateral_mask, view_label = self.get_random_pose_batch(self.batch_size)
        images_mc    = drr(poses, mask_to_channels=True)
        images_clean = self._normalize(images_mc.sum(dim=1, keepdim=True))

        art_end   = self.art_end if self.art_end is not None else images_mc.shape[1]
        art_gt    = images_mc[:, 1:art_end].sum(dim=1, keepdim=True)

        pose_iso = self._get_isopose()
        image_iso = drr(pose_iso)
        image_iso = self._normalize(image_iso)

        drr.cpu()

        images = self._augment_drr(images_clean)

        item = {
            "drr":             drr,
            "images":          images,
            "images_clean":    images_clean,
            "art_gt":          art_gt,
            "art_end":         art_end,
            "poses":           poses,
            "R":               R,
            "t":               t,
            "lateral_mask":    lateral_mask,
            "view_label":      view_label,
            "image_iso":       image_iso,
            "pose_iso":        pose_iso,
        }

        if self.fiducials[idx] is not None:
            item["fiducials"] = self.fiducials[idx]
        return item

    @staticmethod
    def _normalize(images: torch.Tensor) -> torch.Tensor:
        """Min-max normalise each image in a [N, C, H, W] tensor to [0, 1]."""
        N, C, H, W = images.shape
        flat = images.reshape(N * C, -1)
        vmin = flat.min(dim=1).values.view(N, C, 1, 1)
        vmax = flat.max(dim=1).values.view(N, C, 1, 1)
        return (images - vmin) / (vmax - vmin).clamp(min=1e-8)

    def _augment_drr(self, images: torch.Tensor) -> torch.Tensor:
        if self.training and not self.disable_drr_aug:
            return self.drr_aug(images)
        return images


    def _get_isopose(self) -> RigidTransform:
        p = self.pose_cfg
        R = torch.zeros(1, 3, device=self.device)
        t = torch.tensor([[0.0, p.t_y_offset, 0.0]], device=self.device)
        return convert(R, t, parameterization=p.parameterization, convention=p.convention)

    def get_random_pose_batch(self, n: int):
        """Sample n random poses; returns (pose [n,4,4], R [n,3], t [n,3], lateral_mask [n], view_label [n])."""
        p = self.pose_cfg
        t_y_std = getattr(p, 't_y_std', p.t_std)

        t = torch.stack([
            torch.distributions.Normal(0, p.t_std).sample((n,)),
            torch.distributions.Normal(0, t_y_std).sample((n,)),
            torch.distributions.Normal(0, p.t_std).sample((n,)),
        ], dim=1)
        R = torch.distributions.Normal(0, p.r_std).sample((n, 3))

        t[:, 1] += p.t_y_offset
        is_lat = torch.rand(n) < p.lateral_prob
        lat_side = torch.where(torch.rand(n) < 0.5, -1, 1)
        view_label = torch.where(is_lat, lat_side + 1, torch.ones(n, dtype=torch.long))
        sign_lookup = torch.tensor([-1.0, 0.0, 1.0])
        R[:, 0] += sign_lookup[view_label] * (torch.pi / 2)
        lateral_mask = view_label != 1

        t = t.to(self.device)
        R = R.to(self.device)
        lateral_mask = lateral_mask.to(self.device)
        view_label = view_label.to(self.device)

        pose = convert(R, t, parameterization=p.parameterization, convention=p.convention)
        return pose, R, t, lateral_mask, view_label

    def get_random_pose(self):
        pose, R, t, _, _ = self.get_random_pose_batch(1)
        return pose, R[0], t[0]


def _collate_fn(batch):
    """Passthrough collate: DataLoader batch_size must be 1."""
    assert len(batch) == 1, (
        "CTAPoseDataset collate expects DataLoader batch_size=1; "
        f"got {len(batch)} items."
    )
    return batch[0]


class CTAPoseDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

    def setup(self, stage=None):

        val_override = self.cfg.get("val", None)
        if val_override is not None:
            self._setup_cross_dataset(val_override)
            return

        dataset = getattr(self.cfg, "dataset", "cta")
        paths = _list_image_paths(self.cfg.data_root, dataset,
                                  getattr(self.cfg, "align_suffix", "alignedTr"))
        if dataset != "cta" and split_file_of(self.cfg) is not None:
            raise ValueError(
                f"data.split_file names ISLES CTA patients but data.dataset is "
                f"'{dataset}'. Non-CTA datasets hold out CTA subjects through their "
                f"`val:` override block, not through their own split."
            )
        train_indices, val_indices, test_indices = split_indices(paths, self.cfg)

        if self.cfg.max_subjects is not None:
            train_indices = train_indices[: self.cfg.max_subjects]

        train_dataset = CTAPoseDataset(self.cfg, indices=train_indices)
        train_dataset.training = True
        val_dataset = CTAPoseDataset(self.cfg, indices=val_indices + test_indices)
        val_dataset.training = False

        n_val_loaded = len(val_indices)
        self.train_dataset = train_dataset
        self.val_dataset   = Subset(val_dataset, list(range(n_val_loaded)))
        self.test_dataset  = Subset(val_dataset, list(range(n_val_loaded, len(val_dataset))))

    def _setup_cross_dataset(self, val_override):
        """Train on the main dataset (all subjects), val/test on cfg.val's dataset."""

        n_train_total = len(_list_image_paths(self.cfg.data_root, self.cfg.dataset,
                                              getattr(self.cfg, "align_suffix", "alignedTr")))
        train_indices = list(range(n_train_total))
        if self.cfg.max_subjects is not None:
            train_indices = train_indices[: self.cfg.max_subjects]

        val_cfg = OmegaConf.merge(self.cfg, val_override)
        vt_paths = _list_image_paths(val_cfg.data_root, val_cfg.dataset,
                                     getattr(val_cfg, "align_suffix", "alignedTr"))
        _, val_indices, test_indices = split_indices(vt_paths, val_cfg)

        train_dataset = CTAPoseDataset(self.cfg, indices=train_indices)
        train_dataset.training = True
        val_dataset = CTAPoseDataset(val_cfg, indices=val_indices + test_indices)
        val_dataset.training = False

        n_val_loaded = len(val_indices)
        self.train_dataset = train_dataset
        self.val_dataset   = Subset(val_dataset, list(range(n_val_loaded)))
        self.test_dataset  = Subset(val_dataset, list(range(n_val_loaded, len(val_dataset))))

    def _make_loader(self, dataset, *, shuffle: bool, sampler=None):
        return DataLoader(
            dataset,
            batch_size=1,
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            num_workers=self.cfg.num_workers,
            collate_fn=_collate_fn,
        )

    def train_dataloader(self):
        sampler = RandomSampler(self.train_dataset, replacement=True, num_samples=self.cfg.epoch_len)
        return self._make_loader(self.train_dataset, shuffle=False, sampler=sampler)

    def val_dataloader(self):
        return self._make_loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._make_loader(self.test_dataset, shuffle=False)
