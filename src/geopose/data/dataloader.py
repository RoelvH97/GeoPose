import glob
import json
import os

import nibabel as nib
import numpy as np
from scipy.ndimage import label
import pytorch_lightning as pl
import torch
from diffdrr.data import read
from diffdrr.drr import DRR
from diffdrr.pose import RigidTransform, convert
from diffdrr.utils import resample as resample_intrinsics
from omegaconf import DictConfig, OmegaConf
from skimage.transform import resize
from torch.utils.data import DataLoader, Dataset, RandomSampler, Subset

from .augmentations import DRRAugmentations, MAPAugmentation
from .skeleton_fiducials import carotid_skeleton_world
from .splits import split_file_of, split_indices, train_patient_ids


# The sub-stroke9999 volume is a template, not a patient.
TEMPLATE_ID = "sub-stroke9999"

# DSA acquisition pairs that share comparable intrinsics: a↔c (frontal), b↔d (lateral).
# Used as a JSON-metadata fallback when a channel's own .json is missing.
_CHANNEL_PAIR = {"a": "c", "c": "a", "b": "d", "d": "b"}


def _resolve_meta(meta_dir: str, patient_id: str, ch: str) -> str | None:
    """Return path to the channel's intrinsics JSON, falling back to its paired channel."""
    primary = os.path.join(meta_dir, f"{patient_id}_{ch}.json")
    if os.path.exists(primary):
        return primary
    alt = os.path.join(meta_dir, f"{patient_id}_{_CHANNEL_PAIR[ch]}.json")
    return alt if os.path.exists(alt) else None


# View-label classification from the C-arm primary angle `alpha` (degrees).
# IMPORTANT: `alpha` has the OPPOSITE sign to the registered pose's R[:,0]
# (the DRR rotation about Z). Verified against the JSONs' `rot_final`, which
# is in our Euler-ZYX convention (e.g. sub-stroke0001_a: alpha=-79.1° but
# rot_final[0][0]=+1.52 rad ≈ +π/2). So a NEGATIVE alpha is a LAT+ (+π/2)
# acquisition and a POSITIVE alpha is LAT- (-π/2). The mapping below negates
# alpha so labels line up with get_random_pose_batch:
#   0 → LAT(-π/2)   (alpha > +threshold)
#   1 → PA          (|alpha| ≤ threshold)
#   2 → LAT(+π/2)   (alpha < -threshold; the ISLES-common lateral)
# Threshold of 45° cleanly separates the data clusters (|alpha| ≈ 0 vs ~80°).
_VIEW_LABEL_ALPHA_THRESHOLD_DEG = 45.0


def _view_label_from_alpha(alpha_deg: float) -> int:
    # Negate: R[:,0] ≈ -alpha, and the label follows R[:,0]'s sign.
    r0_sign_deg = -alpha_deg
    if r0_sign_deg > _VIEW_LABEL_ALPHA_THRESHOLD_DEG:
        return 2
    if r0_sign_deg < -_VIEW_LABEL_ALPHA_THRESHOLD_DEG:
        return 0
    return 1


def _list_image_paths(data_root: str, dataset: str = "cta", align_suffix: str = "alignedTr") -> list[str]:
    """Sorted list of training image paths.

    For the `cta` dataset the sub-stroke9999 template is dropped, leaving all 99 real
    subjects; the old alignedTr-era corrupted-subject filter is gone because alignedv2
    is validated per subject by scripts/data_prep/qc_alignedv2_subjects.py and the split
    file decides who may be scored. TopBrain has no template, so all
    images_<align_suffix> files are returned. `align_suffix` selects the co-registration
    variant (alignedTr = v1 ANTs, alignedv2 = v2 register_topbrain+COM); it names both
    the images_* and carotis_* subdirs.
    """
    paths = sorted(glob.glob(os.path.join(data_root, f"images_{align_suffix}", "*_0000.nii.gz")))
    if dataset != "cta":
        return paths
    return [p for p in paths if os.path.basename(p).replace("_0000.nii.gz", "") != TEMPLATE_ID]


class CTAPoseDataset(Dataset):
    """Dataset for CTAPose: one item per subject.

    __getitem__ renders `cfg.batch_size` random poses in a single batched DRR
    call, exploiting GPU parallelism.  The DataLoader should use batch_size=1
    together with the passthrough collate_fn defined below.
    """

    def __init__(self, cfg: DictConfig, indices: list | None = None):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        self.pose_cfg = cfg.pose
        self.batch_size = cfg.batch_size
        self.dataset = getattr(cfg, "dataset", "cta")
        # Co-registration variant: alignedTr (v1, ANTs) | alignedv2 (v2, register_topbrain+COM).
        self.align_suffix = getattr(cfg, "align_suffix", "alignedTr")

        # End of the artery channel slice for the Dice/occupancy target. CTA
        # supervises the two carotid arteries (mask_to_channels gives label i in
        # channel i; carotid = channels 1:3). TopBrain has no carotid sub-label
        # but a full vasculature labelmap (max label 40 ⇒ 41 channels), so the
        # target is the union of ALL vessels (channels 1:N). None = "to the end",
        # resolved per render in __getitem__ from images_mc.shape[1].
        self.art_end = 3 if self.dataset == "cta" else None

        mask_dir = os.path.join(cfg.data_root, f"carotis_{self.align_suffix}") if self.dataset == "cta" else None

        all_image_paths = _list_image_paths(cfg.data_root, self.dataset, self.align_suffix)
        if not all_image_paths:
            raise FileNotFoundError(f"No *_0000.nii.gz files found in {cfg.data_root}/images_{self.align_suffix}")
        image_paths = [all_image_paths[i] for i in indices] if indices is not None else all_image_paths

        self.training = True
        aug = cfg.augmentation
        self.map_aug_train = MAPAugmentation(
            training=True,
            sigma_min=aug.map_bilateral_sigma_min,
            sigma_max=aug.map_bilateral_sigma_max,
            val_sigma=aug.map_bilateral_val_sigma,
            max_angle_deg=aug.map_affine_max_angle_deg,
            max_translate=aug.map_affine_max_translate,
            scale_min=aug.map_affine_scale_min,
            scale_max=aug.map_affine_scale_max,
        )
        self.map_aug_val = MAPAugmentation(
            training=False,
            val_sigma=aug.map_bilateral_val_sigma,
        )

        # Per-channel L-R flip for lateral MAP scans during training. Breaks
        # the channel-index → side correlation so the view classifier (and the
        # MAP-NCC / pose-anchor heads that consume view labels) must read the
        # side from the image content. When a channel is flipped, its view
        # label is also swapped 0 ↔ 2. PA channels (label 1) are never flipped.
        self.lateral_flip_p = float(getattr(aug, "lateral_flip_p", 0.0))

        self.disable_drr_aug = bool(getattr(aug, "disable_drr_aug", False))
        # Emit a weak/strong MAP view pair for the semi-supervised EMA mean-teacher
        # on the unlabeled real DSA pool (pairs with model.ema.dsa.enabled).
        self.map_consistency = bool(getattr(aug, "map_consistency", False))
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

        # Vessel-skeleton fiducials for the LXPose-style projection (mPD) loss.
        # Off by default (batch carries no "fiducials" ⇒ criterion skips the term
        # ⇒ existing runs are byte-identical). Dataset-agnostic: the mPD term is a
        # self-consistency reprojection error (fiducials projected under the GT vs
        # predicted pose), so any stable point set in the training volume works —
        # CTA skeletonises the carotid labels, TopBrain/TopCoW their vasculature
        # labelmap. `labels` is therefore per-data-config and has no default that
        # is meaningful across datasets.
        fid_cfg = cfg.get("fiducials", {})
        self.fiducials_enabled = bool(fid_cfg.get("enabled", False))
        self._fid_labels = tuple(fid_cfg.get("labels", (1, 2)))
        self._fid_max_points = fid_cfg.get("max_points", 128)
        self._fid_cache_dir = fid_cfg.get("cache_dir", None)

        self.drrs = []
        self.fiducials = []   # per-patient [1, N, 3] world coords (centered frame) or None
        self.maps = []
        self.segs = []
        self.map_view_labels = []
        for img_path in image_paths:
            patient_id = os.path.basename(img_path).replace("_0000.nii.gz", "")

            if self.dataset == "cta":
                mask_path = os.path.join(mask_dir, f"{patient_id}.nii.gz")
                if not os.path.exists(mask_path):
                    raise FileNotFoundError(f"Mask not found for {img_path}: expected {mask_path}")
                subject = read(img_path, mask_path, labels=list(cfg.drr.labels),
                               fiducials=self._skeleton_fiducials(mask_path))
            else:
                # TopBrain: render with the aligned vasculature labelmap so the
                # DRR carries per-vessel channels (Dice/occupancy target = union
                # of all vessels). labels includes 0, so the union is the whole
                # head ⇒ the summed projection stays a full intensity DRR.
                mask_path = os.path.join(cfg.data_root, "labels_alignedTr", f"{patient_id}.nii.gz")
                if not os.path.exists(mask_path):
                    raise FileNotFoundError(f"TopBrain label not found for {img_path}: expected {mask_path}")
                subject = read(img_path, mask_path, labels=list(cfg.drr.labels),
                               fiducials=self._skeleton_fiducials(mask_path))

            drr = DRR(subject, sdd=cfg.drr.sdd, height=cfg.drr.height, delx=cfg.drr.delx,
                      stop_gradients_through_grid_sample=cfg.drr.stop_gradients)
            self.drrs.append(drr)
            # subject.fiducials is the reoriented [1, N, 3] (centered frame), or
            # None when fiducials are disabled / the labels are empty.
            self.fiducials.append(getattr(subject, "fiducials", None))

            if self.dataset == "cta":
                map_paths = [os.path.join(cfg.dsa_root, "MAPTr", f"{patient_id}_{ch}_0000.nii.gz") for ch in "abcd"]
                meta_dir = os.path.join(cfg.dsa_root, "DSA_arteriesTr")
                meta_paths = [_resolve_meta(meta_dir, patient_id, ch) for ch in "abcd"]
                if all(os.path.exists(p) for p in map_paths) and None not in meta_paths:
                    maps, segs, view_labels = self.load_map(patient_id)
                    self.maps.append(maps)
                    self.segs.append(segs)
                    self.map_view_labels.append(view_labels)
                    continue

            self.maps.append(torch.zeros(4, cfg.drr.height, cfg.drr.height))
            self.segs.append(torch.zeros(4, cfg.drr.height, cfg.drr.height))
            # Dummy labels for patients with no MAP data — never consumed by
            # the model because the corresponding map channels are all-zero
            # and get masked out by `valid_mask`. Default to the dominant
            # pattern (a/c → LAT-, b/d → PA) so any accidental use is benign.
            self.map_view_labels.append(torch.tensor([0, 1, 0, 1], dtype=torch.long))

        # Decoupled DSA MAP pool for non-CTA datasets: MAPs are independent of the
        # CT volume, so coupled losses (NCC against the rendered DRR) must be off,
        # but the MAPs are still usable for domain-adaptation / distribution losses.
        self.map_pool: list[torch.Tensor] | None = None
        self.seg_pool: list[torch.Tensor] | None = None
        self.map_view_label_pool: list[torch.Tensor] | None = None
        if self.dataset != "cta":
            self.map_pool, self.seg_pool, self.map_view_label_pool = self._load_map_pool()

    def _skeleton_fiducials(self, mask_path: str):
        """Skeleton fiducials for the mPD loss, or None when disabled.

        Called BEFORE read() so read's canonicalize reorients the points into the
        same isocentre-centred frame the DRR renders in.
        """
        if not self.fiducials_enabled:
            return None
        return carotid_skeleton_world(
            mask_path, labels=self._fid_labels,
            max_points=self._fid_max_points, cache_dir=self._fid_cache_dir,
        )

    def _load_map_pool(self) -> tuple[list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """Load the train-split ISLES DSA MAPs as a CTA-independent pool.

        The pool feeds domain-adaptation / distribution losses, so restricting it to the
        split's train patients keeps held-out DSA out of training. Without a split file
        (e.g. a bare TopBrain run) it falls back to every usable MAP subject.
        """
        map_dir = os.path.join(self.cfg.dsa_root, "MAPTr")
        meta_dir = os.path.join(self.cfg.dsa_root, "DSA_arteriesTr")
        allowed = train_patient_ids(self.cfg)
        dsa_ids = sorted(
            os.path.basename(p).replace("_a_0000.nii.gz", "")
            for p in glob.glob(os.path.join(map_dir, "*_a_0000.nii.gz"))
        )
        maps_pool: list[torch.Tensor] = []
        segs_pool: list[torch.Tensor] = []
        view_labels_pool: list[torch.Tensor] = []
        for pid in dsa_ids:
            if allowed is not None and pid not in allowed:
                continue
            map_paths = [os.path.join(map_dir, f"{pid}_{ch}_0000.nii.gz") for ch in "abcd"]
            meta_paths = [_resolve_meta(meta_dir, pid, ch) for ch in "abcd"]
            if all(os.path.exists(p) for p in map_paths) and None not in meta_paths:
                m, s, v = self.load_map(pid)
                maps_pool.append(m)
                segs_pool.append(s)
                view_labels_pool.append(v)
        return maps_pool, segs_pool, view_labels_pool

    def load_map(self, patient_id: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Load 4 DSA MAP channels (a/b/c/d) + per-channel view labels."""
        return load_dsa_map_4ch(
            patient_id,
            dsa_root=self.cfg.dsa_root,
            target_sdd=float(self.cfg.drr.sdd),
            target_size=int(self.cfg.drr.height),
            delx=float(self.cfg.drr.delx),
        )

    def __len__(self):
        return len(self.drrs)

    def __getitem__(self, idx):
        drr = self.drrs[idx].to(self.device)

        # Generate batch_size random poses and render them all in one GPU call.
        # Render per-label (mask_to_channels) so we can both sum to the full
        # projection (network input / NCC) and keep the carotid channels (1:3)
        # as the GT-pose Dice target — this avoids a second DRR render in the
        # training step. Summing the channels reproduces the plain projection.
        poses, R, t, lateral_mask, view_label = self.get_random_pose_batch(self.batch_size)
        images_mc    = drr(poses, mask_to_channels=True)                   # [B, n_labels, H, W]
        images_clean = self._normalize(images_mc.sum(dim=1, keepdim=True)) # [B, 1, H, W] pre-augmentation
        # Raw (un-normalised) carotid projection at the GT pose. Kept raw so it
        # shares the path-length scale of the in-step pred render — the squared
        # soft Dice is only scale-invariant under a *common* scale.
        art_end   = self.art_end if self.art_end is not None else images_mc.shape[1]
        art_gt    = images_mc[:, 1:art_end].sum(dim=1, keepdim=True)    # [B, 1, H, W]

        # Single iso reference image
        pose_iso = self._get_isopose()
        image_iso = drr(pose_iso)  # [1, C, H, W]
        image_iso = self._normalize(image_iso)

        drr.cpu()  # free GPU memory; moved back per-sample in training step

        # Apply xvr-style intensity augmentations to synthetic DRRs. The student
        # sees the augmented image; images_clean (pre-aug) feeds the EMA teacher.
        # In val/test (and when DRR aug is disabled) _augment_drr is identity, so
        # images and images_clean coincide.
        images = self._augment_drr(images_clean)

        # All 4 DSA MAP channels for this patient, augmented (segs get same geometric transform).
        # In non-CTA mode, the MAPs are independent of the CT — random in train, deterministic in val/test.
        if self.map_pool:
            pool_idx = (
                int(torch.randint(len(self.map_pool), (1,)).item())
                if self.training
                else idx % len(self.map_pool)
            )
            maps_src = self.map_pool[pool_idx]
            segs_src = self.seg_pool[pool_idx]
            map_view_labels_src = self.map_view_label_pool[pool_idx]
        else:
            maps_src = self.maps[idx]
            segs_src = self.segs[idx]
            map_view_labels_src = self.map_view_labels[idx]
        maps_src_dev        = maps_src.to(self.device)
        map_view_labels_dev = map_view_labels_src.to(self.device)
        maps, segs = self._augment_maps(maps_src_dev, segs_src.to(self.device))  # [4, H, W] each
        maps, segs, map_view_labels = self._maybe_flip_lateral_per_channel(
            maps, segs, map_view_labels_dev
        )

        item = {
            "drr":             drr,                                    # single DRR object (CPU)
            "images":          images,                                 # [B, C, H, W] — augmented (student input)
            "images_clean":    images_clean,                           # [B, C, H, W] — un-augmented (EMA-teacher input)
            "art_gt":          art_gt,                                  # [B, 1, H, W] raw artery projection at GT pose (Dice target)
            "art_end":         art_end,                                 # int — end of the artery channel slice (cta=3 carotid; topbrain=N all vessels)
            "poses":           poses,                                  # RigidTransform [B, 4, 4]
            "R":               R,                                      # [B, 3] raw rotation params
            "t":               t,                                      # [B, 3] raw translations (mm)
            "lateral_mask":    lateral_mask,                           # [B] bool — True iff view is lateral
            "view_label":      view_label,                             # [B] long ∈ {0,1,2}: random-pose labels
            "image_iso":       image_iso,                              # [1, C, H, W]
            "pose_iso":        pose_iso,                               # RigidTransform [1, 4, 4]
            "maps":            maps,                                   # [4, H, W] DSA MAP channels
            "segs":            segs,                                   # [4, H, W] artery segmentations
            "map_view_labels": map_view_labels,                        # [4] long: per-channel label from alpha (post lateral-flip aug)
        }

        # Carotid-skeleton fiducials for the LXPose projection loss ([1, N, 3] world
        # coords in the DRR's centered frame). Present only when data.fiducials.enabled;
        # absent key ⇒ the training step + criterion skip the projection term.
        if self.fiducials[idx] is not None:
            item["fiducials"] = self.fiducials[idx]

        # Weak/strong MAP view pair for the semi-supervised EMA mean-teacher on the
        # unlabeled real DSA pool (model.ema.dsa). Both views share the RAW MAP
        # geometry (no affine, no L-R flip), so a single pose is the correct
        # consistency target; they differ only photometrically — the teacher's
        # val-style light bilateral (weak) vs the student's strong xvr-style
        # intensity corruption (strong). map_view_labels_clean are the pre-flip
        # labels that match the un-flipped pair. Train-only ⇒ val/test unchanged.
        if self.map_consistency and self.training:
            base = MAPAugmentation._normalize(maps_src_dev)             # geometry-identical [0, 1] base
            item["maps_weak"]             = self.map_aug_val(maps_src_dev.clone())   # [4, H, W] light bilateral + norm
            item["maps_strong"]           = self.drr_aug(base.unsqueeze(1)).squeeze(1)  # [4, H, W] strong intensity aug
            item["map_view_labels_clean"] = map_view_labels_dev         # [4] long: pre-flip labels for the pair
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

    def _maybe_flip_lateral_per_channel(
        self,
        maps: torch.Tensor,
        segs: torch.Tensor,
        view_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Independently L-R-flip each lateral MAP channel with prob
        ``lateral_flip_p``. Flipping also toggles the view label 0↔2.
        PA channels (label 1) are never flipped. Identity in val/test or
        when ``lateral_flip_p == 0``.
        """
        if not self.training or self.lateral_flip_p <= 0.0:
            return maps, segs, view_labels
        view_labels = view_labels.clone()
        for c in range(maps.shape[0]):
            if int(view_labels[c]) != 1 and torch.rand(1, device=view_labels.device).item() < self.lateral_flip_p:
                maps[c] = torch.flip(maps[c], dims=[-1])
                segs[c] = torch.flip(segs[c], dims=[-1])
                view_labels[c] = 2 - view_labels[c]   # 0 ↔ 2
        return maps, segs, view_labels

    def _augment_maps(
        self, maps: torch.Tensor, segs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        aug = self.map_aug_train if self.training else self.map_aug_val
        return aug(maps, segs)

    def _get_isopose(self) -> RigidTransform:
        p = self.pose_cfg
        R = torch.zeros(1, 3, device=self.device)
        t = torch.tensor([[0.0, p.t_y_offset, 0.0]], device=self.device)
        return convert(R, t, parameterization=p.parameterization, convention=p.convention)

    def get_random_pose_batch(self, n: int):
        """Sample n random poses; returns (pose [n,4,4], R [n,3], t [n,3], lateral_mask [n], view_label [n]).

        View distribution reflects clinical bi-plane acquisition (one PA + one
        lateral per case), with the lateral side chosen at random:
            * 50% PA            — R[:,0] residual added to 0
            * 25% LAT, left     — R[:,0] residual added to -π/2  (view_label=0)
            * 25% LAT, right    — R[:,0] residual added to +π/2  (view_label=2)
        PA carries view_label=1. ``lateral_mask = view_label != 1`` is kept for
        backward-compatible binary-view consumers (e.g. refinement embeddings).

        ``lateral_prob`` from the pose config still controls the *any-lateral*
        marginal (defaults to 0.5); the two lateral sides are equiprobable
        conditional on being lateral.
        """
        p = self.pose_cfg
        t_y_std = getattr(p, 't_y_std', p.t_std)

        # Vectorised sampling
        t = torch.stack([
            torch.distributions.Normal(0, p.t_std).sample((n,)),
            torch.distributions.Normal(0, t_y_std).sample((n,)),
            torch.distributions.Normal(0, p.t_std).sample((n,)),
        ], dim=1)  # [n, 3]
        R = torch.distributions.Normal(0, p.r_std).sample((n, 3))  # [n, 3]

        t[:, 1] += p.t_y_offset
        is_lat = torch.rand(n) < p.lateral_prob
        lat_side = torch.where(torch.rand(n) < 0.5, -1, 1)            # ±1
        view_label = torch.where(is_lat, lat_side + 1, torch.ones(n, dtype=torch.long))  # 0/1/2
        sign_lookup = torch.tensor([-1.0, 0.0, 1.0])
        R[:, 0] += sign_lookup[view_label] * (torch.pi / 2)
        lateral_mask = view_label != 1

        t = t.to(self.device)
        R = R.to(self.device)
        lateral_mask = lateral_mask.to(self.device)
        view_label = view_label.to(self.device)

        pose = convert(R, t, parameterization=p.parameterization, convention=p.convention)
        return pose, R, t, lateral_mask, view_label

    # Keep single-pose variant for backwards compatibility / external callers
    def get_random_pose(self):
        pose, R, t, _, _ = self.get_random_pose_batch(1)
        return pose, R[0], t[0]


def load_dsa_maps(
    patient_id: str,
    dsa_root: str,
    target_sdd: float,
    target_size: int,
    delx: float,
    channels: tuple[str, ...] = ("a", "b", "c", "d"),
) -> dict[str, tuple[torch.Tensor, torch.Tensor, int]]:
    """Per-channel DSA MAP loader.

    Returns ``{channel: (img [H, W], seg [H, W], view_label)}`` for each
    requested channel. ``view_label`` ∈ {0=LAT-, 1=PA, 2=LAT+} is derived
    from the channel's primary angle ``alpha`` (falling back to its paired
    channel's JSON if missing). Channels are SDD-resampled to ``target_sdd``
    and resized to ``target_size``.
    """
    size = target_size
    out: dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    img_dir = os.path.join(dsa_root, "MAPTr")
    msk_dir = os.path.join(dsa_root, "MAP_maskTr")
    art_dir = os.path.join(dsa_root, "MIP_arteriesTr")
    meta_dir = os.path.join(dsa_root, "DSA_arteriesTr")

    for ch in channels:
        path = os.path.join(img_dir, f"{patient_id}_{ch}_0000.nii.gz")
        arr = nib.load(path).get_fdata(dtype=np.float32).squeeze()
        arr = np.fliplr(np.rot90(arr, 3))

        mask_path = os.path.join(msk_dir, f"{patient_id}_{ch}.nii.gz")
        msk = nib.load(mask_path).get_fdata(dtype=np.float32).squeeze()
        msk = np.fliplr(np.rot90(msk, 3))

        labeled, num_mask = label(msk > 0)
        if num_mask > 1:
            sizes = np.bincount(labeled.ravel())
            sizes[0] = 0  # ignore background
            msk = (labeled == sizes.argmax()).astype(np.float32)

        art_path = os.path.join(art_dir, f"{patient_id}_{ch}.nii.gz")
        art = nib.load(art_path).get_fdata(dtype=np.float32)
        art = np.max(art == 2, axis=2).squeeze()
        art = np.fliplr(np.rot90(art, 3))

        labeled_art, num_art = label(art)
        if num_art > 1:
            sizes_art = np.bincount(labeled_art.ravel())
            sizes_art[0] = 0
            art = (labeled_art == sizes_art.argmax()).astype(np.float32)

        # Normalise over valid (non-zero) region
        img_nonzero = np.sum(arr, axis=0) != 0
        arr_valid = arr[:, img_nonzero] if img_nonzero.any() else arr
        vmin, vmax = np.percentile(arr_valid, [25, 75])
        if vmin == vmax:
            vmax += 40
        arr = np.clip(arr, vmin, vmax)
        arr -= arr.min()

        # Resample to network's SDD so the projection geometry matches training.
        # Follows xvr/diffdrr inference: rescale at native resolution, then resize.
        # Falls back to the paired channel (a↔c, b↔d) if this channel has no JSON.
        with open(_resolve_meta(meta_dir, patient_id, ch)) as fp:
            meta = json.load(fp)
        src_sdd = float(meta["d_source_to_detector"])
        view_label = _view_label_from_alpha(float(meta["alpha"]))
        if src_sdd != target_sdd:
            stack = torch.from_numpy(np.stack([arr, msk, art])).float()[:, None]  # [3,1,H,W]
            stack = resample_intrinsics(stack, src_sdd, delx, 0, 0, target_sdd, delx, 0, 0)
            arr = stack[0, 0].numpy()
            msk = (stack[1, 0].numpy() > 0.5).astype(np.float32)
            art = (stack[2, 0].numpy() > 0.5).astype(np.float32)

        # Resize to detector size
        if arr.shape != (size, size):
            arr = resize(arr, (size, size), preserve_range=True, anti_aliasing=True)
            msk = resize(msk, (size, size), preserve_range=True, anti_aliasing=False)
            art = resize(art, (size, size), preserve_range=True, anti_aliasing=False)

        img_out = torch.tensor(arr, dtype=torch.float32) * torch.tensor(msk, dtype=torch.float32)
        seg_out = torch.tensor(art, dtype=torch.float32) * torch.tensor(msk, dtype=torch.float32)

        out[ch] = (img_out, seg_out, view_label)  # ([H, W] image, [H, W] seg, int label)

    return out

 
def load_dsa_map_4ch(
    patient_id: str,
    dsa_root: str,
    target_sdd: float,
    target_size: int,
    delx: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """4-channel convenience wrapper around :func:`load_dsa_maps`.

    Returns ``(maps, segs, view_labels)`` where maps/segs have shape
    ``[4, target_size, target_size]`` and view_labels has shape ``[4]``
    (long, with values in {0=LAT-, 1=PA, 2=LAT+}). All stacked in fixed
    ``a/b/c/d`` order.
    """
    out = load_dsa_maps(patient_id, dsa_root, target_sdd, target_size, delx)
    imgs        = torch.stack([out[ch][0] for ch in "abcd"])
    segs        = torch.stack([out[ch][1] for ch in "abcd"])
    view_labels = torch.tensor([out[ch][2] for ch in "abcd"], dtype=torch.long)
    return imgs, segs, view_labels


def _collate_fn(batch):
    """Passthrough collate: DataLoader batch_size must be 1.

    Each item already contains a full batch of rendered poses for one patient,
    so we simply unwrap the outer list dimension added by DataLoader.
    """
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
        # Cross-dataset mode: when cfg.val is set, train on the main dataset and
        # validate/test on a *different* dataset (e.g. train TopBrain, val on the
        # CTA split) so metrics stay comparable to native CTA runs.
        val_override = self.cfg.get("val", None)
        if val_override is not None:
            self._setup_cross_dataset(val_override)
            return

        # Determine split indices before loading any data
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

        # Only load subjects that are actually needed for each split
        train_dataset = CTAPoseDataset(self.cfg, indices=train_indices)
        train_dataset.training = True
        val_dataset = CTAPoseDataset(self.cfg, indices=val_indices + test_indices)
        val_dataset.training = False

        # val/test are contiguous in val_dataset; remap indices accordingly
        n_val_loaded = len(val_indices)
        self.train_dataset = train_dataset
        self.val_dataset   = Subset(val_dataset, list(range(n_val_loaded)))
        self.test_dataset  = Subset(val_dataset, list(range(n_val_loaded, len(val_dataset))))

    def _setup_cross_dataset(self, val_override):
        """Train on the main dataset (all subjects), val/test on cfg.val's dataset.

        The val/test split is the val dataset's own split (its split_file, or its
        train_fraction/val_fraction slices when it has none), so the held-out subjects
        match a native run on that dataset — only the training source changes.
        """
        # Train: every subject of the main dataset (val comes from elsewhere).
        n_train_total = len(_list_image_paths(self.cfg.data_root, self.cfg.dataset,
                                              getattr(self.cfg, "align_suffix", "alignedTr")))
        train_indices = list(range(n_train_total))
        if self.cfg.max_subjects is not None:
            train_indices = train_indices[: self.cfg.max_subjects]

        # Val/test: the same held-out subjects a native val-dataset run would use.
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
            batch_size=1,           # each item IS a full batch of poses
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
