# GeoPose: Patient-agnostic DSA-to-CTA registration through projection-space calibration and transform composition

GeoPose registers biplanar digital subtraction angiography (DSA) to a patient's
preprocedural CT angiography (CTA). A DSA image is a 2D projection, and therefore has no
direct spatial correspondence with the 3D CTA. GeoPose estimates the 6-DOF
C-arm pose for each view and expresses it in the coordinate frame of the
patient's native CTA. The recovered poses can be used to project CTA anatomy
onto the acquired views or to reconstruct 3D vasculature from the biplanar
images.

The models are trained on a population of synthetic CTA projections and keep
the same weights for every test patient. At inference, GeoPose renders the new
CTA once at a known pose. This **isopose calibration** measures the difference
between the model's canonical training frame and the CTA's native frame. The
calibrated prediction can then be improved by the learned refiner and,
optionally, a short image-based GeoReg optimization.

This repository contains the reference implementation for the forthcoming
GeoPose manuscript.

<p align="center">
  <img src="docs/assets/geopose_showcase.gif" alt="GeoPose registration progression across held-out test cases" width="100%">
</p>
<p align="center"><sub>
Native pose → calibrated GeoPose-Init → GeoPose-Refine → 25-step GeoReg. Magenta/cyan contours show the DSA/CTA cranium silhouettes used by the test-time objective.
</sub></p>

## What GeoPose does

GeoPose-Init predicts a pose in one forward pass. A single calibration render
then transfers that prediction from the canonical training frame to the frame
of an unseen, unregistered CTA. GeoPose-Refine compares the target projection
with a CTA render at the current pose and predicts an SE(3) correction. Neither
network is fitted or updated for a new patient.

A shared ResNet-34 handles the signed view roles `LAT−`, `PA`, and `LAT+`, with
rotation anchors at ±π/2 for the lateral views. For one biplanar PA/LAT pair on
an NVIDIA H100, calibrated GeoPose-Init takes 21.8 ms. One Refine update brings
the total to 52.1 ms, while greedy refinement takes 147.0 ms on average after
warm-up.

The optional GeoReg stage uses image similarity and Dice overlap between the
DSA and CTA cranium silhouettes. It does not require carotid or intracranial
vessel annotations for a new DSA study. DiffDRR supplies the CTA renders used by
both the learned correction and the short NAdam optimization.

GeoPose builds on [GeoReg](https://github.com/RoelvH97/GeoReg). GeoReg performs
direct differentiable DSA-to-CTA registration. GeoPose provides a calibrated
initial pose and a learned correction, so GeoReg does not have to begin a long
optimization from a generic pose.

## Inference speed

GeoPose produces a registration pose before iterative optimization in about
0.2 seconds:

| Online output for one biplanar pair | Runtime on H100, mean ± SD |
|---|---:|
| Calibrated GeoPose-Init | 21.8 ± 2.3 ms |
| GeoPose-Init + Refine (×1) | 52.1 ± 6.9 ms |
| GeoPose-Init + greedy Refine (×K) | 147.0 ± 39.0 ms |

These measurements include the calibration render and prediction, PA and LAT
pose predictions, pose composition, and all configured refinement renders,
network evaluations, and acceptance checks. They do not include the optional
25-step GeoReg stage. With that stage, online registration takes about 2 seconds
per biplanar pair on the same GPU class.

## Method

<p align="center">
  <img src="docs/assets/geopose_method.png" alt="GeoPose method: pose estimation, projection-space calibration, learned pose refinement, and optional GeoReg optimization" width="100%">
</p>
<p align="center"><sub>
GeoPose predicts pose in a canonical frame, transfers it to the native CTA frame, and applies learned refinement before an optional GeoReg optimization.
</sub></p>

### 1. GeoPose-Init

`ResNetPose` predicts Euler-ZYX rotation and translation residuals relative to a
view-specific isopose anchor. Training uses only synthetic DRRs with known
poses. The signed view role distinguishes the two lateral acquisition
directions, while all views share one backbone.

### 2. Isopose calibration

Training in a canonical frame makes pose labels comparable across patients. A
new CTA, however, may use a different voxel or anatomical frame. GeoPose renders
the CTA at the known isopose `P_iso` and passes the result through the same
network to obtain `P_cal`. For an angiography prediction `P_pred`, the pose in
the new CTA frame is

```text
P_native = P_iso · inverse(P_cal) · P_pred.
```

GeoPose computes this calibration once per CTA. It requires no angiography
annotation and does not update the network weights.

### 3. GeoPose-Refine

The siamese refiner receives `(target projection, current CTA render)` and
predicts an axis-angle and translation correction. It right-composes each
correction on SE(3) and accepts an inference-time update only if multiscale NCC
improves.

### 4. Short GeoReg refinement

The optional final stage runs 25 NAdam/OneCycle steps over the pose parameters.
Its fixed objective combines multiscale NCC with Dice overlap between DSA and
rendered CTA **cranium masks**. GeoReg retains the pose with the best NCC for
each view independently. Despite its historical name, `MAP_maskTr` contains
cranium masks, not carotid masks.

## Installation

```bash
git clone https://github.com/RoelvH97/GeoPose.git
cd GeoPose
conda env create -f environment.yml
conda activate geopose
```

The environment pins Python 3.13, PyTorch 2.6, and the main dependencies. It
pins PyTorch3D to a specific Git commit. Because `fireants`,
`bilateralfilter-torch`, and `HD-BET` do not provide wheels for every platform,
the supported setup for preregistration and inference is a CUDA-capable Linux
environment. The training smoke tests can run on a CPU.

Installation exposes:

- `geopose-preregister` stages public CTA data and builds the canonical cohort.
- `geopose-train` trains GeoPose-Init or GeoPose-Refine.
- `geopose-test` runs calibration, learned refinement, and GeoReg.

The root scripts `preregister.py`, `train.py`, and `test.py` provide the same
commands from a source checkout.

## Release artifacts

The Python package includes the privacy-preserving
`sub-stroke0011_pre.npz` projection bundle. Before using it, the code verifies
its size and SHA-256 hash. The bundle contains deterministic 256×256 DSA MAP
arrays, acquisition scalars, and cranium masks. It contains neither a DSA
sequence nor full-resolution angiography.

The manuscript repository is currently in its pre-archive state:

| Artifact | Status | Integrity record |
| --- | --- | --- |
| Example projection bundle | Packaged | `artifacts/example_sub-stroke0011.json` |
| GeoPose-Init and GeoPose-Refine checkpoints | Hashes frozen; archive URL pending | `artifacts/checkpoints.json` |
| Native-grid carotid masks and CTA cranium masks | Companion archive pending | `artifacts/data_contract.json` |

The archival release requires values for `zenodo_doi` and `download_url` in the
manifests. The code rejects files that use the official checkpoint names but do
not match the frozen hashes.

## Data

### Training data

Training uses public ISLES'24 CTA and the GeoPose native-grid carotid masks:

| Source | Location |
| --- | --- |
| ISLES'24 CTA | [Zenodo record 17652035](https://zenodo.org/records/17652035) |
| GeoPose carotid masks | Companion GeoPose archive (DOI pending) |

Carotid masks supervise the synthetic training objectives. They are not needed
to register a new angiography case at test time.

The frozen cohort contains 99 patients, split by patient into sets of 69, 10,
and 20. The split is stored in `src/geopose/assets/isles_split_v1.json`. The
dataset loader rejects missing, unassigned, or overlapping subjects.

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

The `DSA_arteriesTr` directory name comes from the research data layout. Its
JSON files contain acquisition geometry; inference does not read an artery
segmentation from them.

The clinical DSA series are not public, so the full-cohort inference experiment
cannot be reproduced from this release. The packaged example substitutes for
the private DSA inputs at the deterministic model boundary.

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
New checkpoints use `epoch=NNN.ckpt` filenames, and model selection monitors
`val/loss`.

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

For `sub-stroke0011/pre`, the CLI uses the packaged projection bundle if
`<data-root>/ProjectionTr/sub-stroke0011_pre.npz` is absent. An explicit
`--projection-file` takes precedence.

`result.json` records checkpoint and projection hashes, the calibration and
optimization traces, the determinism policy, and final poses as Euler-ZYX
radians, millimetre translations, and 4×4 matrices. Target/render PNGs are
written beside it.

The default, `--determinism warn`, requests deterministic CUDA execution where
the platform supports it. Use `error` to reject operations that PyTorch marks
as nondeterministic, or `off` for benchmarking. Third-party CUDA extensions may
still produce slightly different results across hardware. Scientific
comparisons should therefore use one environment and pose tolerances chosen in
advance.

## Reproducibility scope

- The checkpoint, split, example, and source manifests contain SHA-256 records.
- The launcher passes `--seed` to Lightning, the dataset samplers, and the
  deterministic validation sampler.
- The release configs reproduce the documented training procedure from public
  data. They cannot reproduce the published weights bit for bit because the
  original GeoPose-Init run did not record a global seed.
- `source_provenance.json` identifies the research and release snapshots. One
  test checks release-file integrity; projection-boundary tests and optional
  end-to-end integration tests cover behavioral equivalence separately.

## Pose conventions

Poses are DiffDRR camera transforms. Rotations use Euler-ZYX radians, and
translations use millimetres with isocentre `t = (0, 650, 0)` mm. The signed
view roles are `0 = LAT−`, `1 = PA`, and `2 = LAT+`. The acquisition-angle
threshold is ±45°. `registration/views.py` documents two acquisition errata in
the source archive.

## Tests

```bash
pytest
```

The default suite tests geometry, split and release contracts, public-data
preparation, projection validation, CLI behavior, deterministic controls, and
training configuration. Run the optional tests with:

```bash
export GEOPOSE_INIT_CHECKPOINT=/path/to/geopose_init.ckpt
export GEOPOSE_REFINE_CHECKPOINT=/path/to/geopose_refine.ckpt
export GEOPOSE_EXAMPLE_DATA_ROOT=/path/to/example
pytest -m integration
```

Private-route equivalence also requires `GEOPOSE_PRIVATE_DATA_ROOT`. These tests
are for the release maintainer because they depend on data that are not public.

## Citation

See `CITATION.cff`. The archival paper DOI will be added when assigned.

## License

GeoPose is released under the MIT License. See `LICENSE`.
