# GeoPose: calibrated 6-DOF C-arm pose estimation

GeoPose estimates the pose of biplane cerebral angiography views relative to a
preoperative CTA. It turns a single 256×256 projection into a useful 6-DOF
initialization, transfers that prediction into the coordinate frame of a
previously unseen CTA through **isopose calibration**, and optionally improves
it with learned and differentiable-rendering refinement.

This repository is the reference implementation accompanying the to-be-published
GeoPose manuscript. The code is the complete executable specification when a
minor wording or notation difference exists between the manuscript and the
release.

<p align="center">
  <img src="docs/assets/geopose_showcase.gif" alt="GeoPose registration progression across held-out test cases" width="100%">
</p>
<p align="center"><sub>
Native pose → calibrated GeoPose-Init → GeoPose-Refine → 25-step GeoReg. Magenta/cyan contours show the DSA/CTA cranium silhouettes used by the test-time objective.
</sub></p>

## Why GeoPose

- **Calibrated across CTA coordinate frames.** A network trained in one canonical
  anatomical frame can be used with a native, unseen CTA. GeoPose renders that
  CTA at a known isopose, measures the network's frame bias, and transfers the
  angiography prediction into the new CTA frame.
- **No patient-specific network fitting.** GeoPose-Init predicts a pose in one
  forward pass and GeoPose-Refine predicts an SE(3) correction from the target
  projection and a render at the current pose.
- **Sub-second learned inference.** For one biplanar PA/LAT pair on an NVIDIA
  H100, calibrated GeoPose-Init takes 21.8 ms, one Refine update takes the total
  to 52.1 ms, and complete greedy refinement takes 147.0 ms (mean warmed
  online-stage wall clock).
- **View-aware 6-DOF prediction.** A shared ResNet-34 uses signed view roles
  `LAT−`, `PA`, and `LAT+`, including their ±π/2 rotation anchors.
- **No vessel segmentation for new angiography.** Test-time GeoReg refinement
  uses image similarity and **cranium-silhouette Dice**. New DSA images do not
  require carotid or intracranial-vessel annotations.
- **Physics remains in the loop.** DiffDRR renders the new CTA under the
  predicted pose, enabling greedy learned correction and short NAdam refinement.

GeoPose complements [GeoReg](https://github.com/RoelvH97/GeoReg): GeoReg provides
direct differentiable DSA-to-CTA registration, while GeoPose supplies a fast,
calibrated initialization and learned correction that reduce dependence on a
long optimization from a generic starting pose.

## Inference speed

GeoPose provides a useful registration pose before iterative optimization in
well under a second:

| Online output for one biplanar pair | Runtime on H100, mean ± SD |
|---|---:|
| Calibrated GeoPose-Init | 21.8 ± 2.3 ms |
| GeoPose-Init + Refine (×1) | 52.1 ± 6.9 ms |
| GeoPose-Init + greedy Refine (×K) | 147.0 ± 39.0 ms |

These are warmed, CUDA-synchronized wall-clock measurements for one 256×256
PA/LAT pair over ten held-out patients. They cover calibration rendering and
prediction, both angiography-view predictions, pose composition, and the
configured refinement renders, forwards, and acceptance checks. Checkpoint
loading, data I/O, preprocessing, and one-time renderer construction are
excluded. The optional 25-step GeoReg stage is not included in the three rows;
with that stage, the complete online registration takes approximately 2 s per
biplanar pair on the same GPU class.

## Method

```text
                         unseen native CTA
                                │
                    render known PA isopose
                                │
                                ▼
DSA MAP ──► GeoPose-Init ──► frame-transfer calibration ──► initial pose
                                                               │
                    DSA MAP + CTA render at current pose       │
                                │                              ▼
                                └────► GeoPose-Refine ──► refined pose
                                                               │
                    optional 25-step NCC + cranium-Dice GeoReg ┘
```

### 1. GeoPose-Init

`ResNetPose` regresses Euler-ZYX rotation and translation residuals about a
view-specific isopose anchor. It is trained entirely on synthetic DRRs with
known poses. The signed view role distinguishes the two lateral acquisition
directions while retaining a single shared backbone.

### 2. Isopose calibration

Canonical-frame training makes pose labels comparable across patients, but a
new CTA can arrive in a different voxel/anatomical frame. GeoPose therefore
renders the new CTA at the known isopose `P_iso`, passes that render through the
same network, and obtains `P_cal`. If the angiography prediction is `P_pred`, the
pose transferred into the new CTA frame is

```text
P_native = P_iso · inverse(P_cal) · P_pred.
```

This calibration is computed once per CTA, uses no angiography annotation, and
does not update network weights.

### 3. GeoPose-Refine

The warm-started siamese refiner receives `(target projection, current CTA
render)` and predicts an axis-angle/translation correction. Corrections are
right-composed on SE(3); at inference, an update is accepted only while
multiscale NCC improves.

### 4. Short GeoReg refinement

The optional final stage runs 25 NAdam/OneCycle steps over the pose parameters.
Its frozen objective combines multiscale NCC with Dice between DSA and rendered
CTA **cranium masks**. The best-NCC pose is retained independently per view.
Despite the historical variable name `MAP_maskTr`, these are cranium masks—not
carotid masks.

## Installation

```bash
git clone https://github.com/RoelvH97/GeoPose.git
cd GeoPose
conda env create -f environment.yml
conda activate geopose
```

Python 3.13, PyTorch 2.6, and the primary dependencies are pinned. PyTorch3D is
pinned to a Git commit. `fireants`, `bilateralfilter-torch`, and `HD-BET` do not
provide wheels for every platform, so a CUDA-capable Linux environment is the
supported path for preregistration and inference. CPU execution is supported
for the training smoke tests.

Installation exposes:

- `geopose-preregister` — stage public CTA data and build the canonical cohort;
- `geopose-train` — train GeoPose-Init or GeoPose-Refine;
- `geopose-test` — run calibration, learned refinement, and GeoReg.

The root scripts `preregister.py`, `train.py`, and `test.py` provide the same
commands from a source checkout.

## Release artifacts

The privacy-preserving `sub-stroke0011_pre.npz` projection bundle is included in
the Python package and verified by size and SHA-256 before use. It contains only
the deterministic 256×256 model-boundary arrays, acquisition scalars, and
cranium masks—no DSA sequence or full-resolution angiography.

The manuscript repository is currently in its pre-archive state:

| Artifact | Status | Integrity record |
| --- | --- | --- |
| Example projection bundle | Packaged | `artifacts/example_sub-stroke0011.json` |
| GeoPose-Init and GeoPose-Refine checkpoints | Hashes frozen; archive URL pending | `artifacts/checkpoints.json` |
| Native-grid carotid masks and CTA cranium masks | Companion archive pending | `artifacts/data_contract.json` |

Before the archival release, `zenodo_doi` and `download_url` in the manifests
must be populated. The code refuses the official checkpoint names when their
bytes do not match the frozen hashes.

## Data

### Training data

Training uses public ISLES'24 CTA and the GeoPose native-grid carotid masks:

| Source | Location |
| --- | --- |
| ISLES'24 CTA | [Zenodo record 17652035](https://zenodo.org/records/17652035) |
| GeoPose carotid masks | Companion GeoPose archive (DOI pending) |

Carotid masks supervise the synthetic training objectives. They are **not** an
input required to fit or register a new angiography case at test time.

The frozen cohort contains 99 patients split 69/10/20 by patient in
`src/geopose/assets/isles_split_v1.json`. Dataset loading rejects missing,
unassigned, or overlapping subjects.

```text
<source-root>/
  CTATr/<subject>_0000.nii.gz
  CTA_carotisTr/<subject>.nii.gz
  brainmasks_Tr/<subject>_0000_bet.nii.gz

<aligned-root>/
  images_alignedv2/<subject>_0000.nii.gz
  carotis_alignedv2/<subject>.nii.gz
  masks_alignedv2/<subject>.nii.gz
  transforms_alignedv2/<subject>.{pt,json}
```

### New-case inference data

For a new patient, inference requires a native CTA, its cranium mask, two DSA
MAP projections with cranium masks, and acquisition geometry:

```text
<data-root>/
  CTATr/<subject>_0000.nii.gz
  CTA_skullTr/<subject>.nii.gz
  DSATr/<subject>_<channel>_0000.nii.gz
  MAPTr/<subject>_<channel>_0000.nii.gz
  MAP_maskTr/<subject>_<channel>.nii.gz       # DSA cranium mask
  DSA_arteriesTr/<subject>_<channel>.json     # acquisition geometry
```

The final directory name is retained for compatibility with the research data
layout; the JSON supplies geometry and does not imply that an artery
segmentation is consumed.

Full-cohort inference cannot be reproduced publicly because the clinical DSA
series are not released. The packaged example replaces the private DSA inputs
at the deterministic model boundary.

## Quick start

### 1. Prepare and align the public CTA cohort

```bash
geopose-preregister all \
  --isles-root   /path/to/isles24 \
  --carotid-root /path/to/geopose_carotids \
  --source-root  /path/to/staged \
  --output-root  /path/to/alignedv2
```

### 2. Train the two stages

```bash
geopose-train init \
  --data-root /path/to/alignedv2 \
  --output-dir runs/init \
  --seed 0

geopose-train refine \
  --data-root /path/to/alignedv2 \
  --output-dir runs/refine \
  --init-checkpoint runs/init/checkpoints/<best>.ckpt \
  --seed 0
```

`--smoke --accelerator cpu` runs one epoch with two sampled training batches.
New checkpoints use unambiguous `epoch=NNN.ckpt` filenames; model selection
continues to monitor `val/loss`.

### 3. Run the packaged example

After downloading the public CTA, companion CTA cranium mask, and frozen
checkpoints:

```bash
geopose-test \
  --data-root         /path/to/example \
  --patient           sub-stroke0011 \
  --init-checkpoint   geopose_init.ckpt \
  --refine-checkpoint geopose_refine.ckpt \
  --output-dir        runs/example
```

For `sub-stroke0011/pre`, the CLI automatically uses the packaged projection
bundle when `<data-root>/ProjectionTr/sub-stroke0011_pre.npz` is absent. An
explicit `--projection-file` takes precedence.

`result.json` records checkpoint and projection hashes, the calibration and
optimization traces, the determinism policy, and final poses as Euler-ZYX
radians, millimetre translations, and 4×4 matrices. Target/render PNGs are
written beside it.

Use `--determinism warn` (default) for portable best-effort deterministic CUDA
execution, `error` to reject an operation PyTorch marks nondeterministic, or
`off` for benchmarking. Third-party CUDA extensions can still vary slightly
across hardware, so scientific comparisons should use one environment and
predeclared pose tolerances.

## Reproducibility scope

- Frozen checkpoint, split, example, and source manifests use SHA-256 records.
- The launcher now propagates `--seed` to Lightning, dataset samplers, and
  deterministic validation sampling.
- The release configs reproduce the documented training procedure from public
  data; they do not promise bitwise recreation of the published weights because
  the original GeoPose-Init run did not record a global seed.
- `source_provenance.json` records the research and release snapshots. Its test
  enforces release-file integrity; behavioral equivalence is covered separately
  by projection-boundary and optional end-to-end integration tests.

## Pose conventions

Poses are DiffDRR camera transforms. Rotation uses Euler-ZYX radians and
translation uses millimetres, with isocentre `t = (0, 650, 0)` mm. Signed view
roles are `0 = LAT−`, `1 = PA`, and `2 = LAT+`; acquisition angle is thresholded
at ±45°. Two source-archive acquisition errata are documented in
`registration/views.py`.

## Tests

```bash
pytest
```

The default suite covers geometry, split and release contracts, public-data
preparation, projection validation, CLI behavior, deterministic controls, and
training configuration. Optional tests are enabled with:

```bash
export GEOPOSE_INIT_CHECKPOINT=/path/to/geopose_init.ckpt
export GEOPOSE_REFINE_CHECKPOINT=/path/to/geopose_refine.ckpt
export GEOPOSE_EXAMPLE_DATA_ROOT=/path/to/example
pytest -m integration
```

Private-route equivalence additionally requires `GEOPOSE_PRIVATE_DATA_ROOT` and
is intended for the release maintainer, not public users.

## Citation

See `CITATION.cff`. The archival paper DOI will be added when assigned.

## License

GeoPose is released under the MIT License. See `LICENSE`.
