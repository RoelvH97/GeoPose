"""End-to-end GeoPose publication inference orchestration."""

from __future__ import annotations

import argparse
import json

import torch
from diffdrr.data import read

from .geometry import pose_matrix, tensor_list
from .initialization import PoseInitializer, file_sha256, load_init_model, load_refine_model
from .optimization import TestTimeOptimizer, prepare_registration_inputs, save_final_renders
from .projections import load_projection_file


def configure_reproducibility(mode: str = "warn") -> dict:
    """Configure repeatable inference and report the determinism policy."""
    if mode not in {"off", "warn", "error"}:
        raise ValueError(f"Unknown determinism mode: {mode!r}")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = mode != "off"
    torch.use_deterministic_algorithms(
        mode != "off",
        warn_only=mode == "warn",
    )
    return {
        "seed": 0,
        "determinism": mode,
        "cudnn_benchmark": False,
        "cudnn_deterministic": mode != "off",
    }


def run_inference(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("The publication inference pipeline requires CUDA")
    reproducibility = configure_reproducibility(
        getattr(args, "determinism", "warn")
    )
    device = torch.device("cuda")
    data_root = args.data_root.resolve()
    patient = args.patient

    projection_path = getattr(args, "projection_file", None)
    projections = None
    if projection_path is not None:
        projection_path = projection_path.resolve()
        projections = load_projection_file(projection_path, patient, args.timestamp)

    cta_path = data_root / "CTATr" / f"{patient}_0000.nii.gz"
    cta_mask_path = data_root / "CTA_skullTr" / f"{patient}.nii.gz"
    for required in (cta_path, cta_mask_path):
        if not required.is_file():
            raise FileNotFoundError(required)
    cta_subject = read(str(cta_path), str(cta_mask_path), labels=[0, 1])

    init_model = load_init_model(
        args.init_checkpoint.resolve(), device, args.skip_hash_check
    )
    refine_model = load_refine_model(
        args.refine_checkpoint.resolve(), device, args.skip_hash_check
    )
    initializer = PoseInitializer(
        init_model,
        refine_model,
        cta_subject,
        data_root,
        device,
        max_refine_updates=args.max_refine_updates,
    )
    initial_poses, initialization_trace = initializer.predict(
        patient, args.timestamp, projections
    )

    images, cranium_masks, metadata = prepare_registration_inputs(
        data_root, patient, args.timestamp, device, projections=projections
    )
    optimizer = TestTimeOptimizer(
        cta_subject,
        images,
        cranium_masks,
        metadata,
        initial_poses,
        device,
    ).to(device)
    final_poses, optimization_trace = optimizer.optimize(args.iterations)

    result = {
        "schema_version": 2,
        "contract": "geopose-inference-v2",
        "reproducibility": reproducibility,
        "patient": patient,
        "timestamp": args.timestamp,
        "projection_input": (
            {"path": str(projection_path), "sha256": file_sha256(projection_path)}
            if projection_path is not None
            else None
        ),
        "checkpoints": {
            "init": {
                "path": str(args.init_checkpoint.resolve()),
                "sha256": file_sha256(args.init_checkpoint.resolve()),
            },
            "refine": {
                "path": str(args.refine_checkpoint.resolve()),
                "sha256": file_sha256(args.refine_checkpoint.resolve()),
            },
        },
        "initialization": initialization_trace,
        "optimization": optimization_trace,
        "final_pose": {
            view: {
                "rotation_zyx_radians": tensor_list(rotation),
                "translation_mm": tensor_list(translation),
                "matrix": tensor_list(pose_matrix(rotation, translation)),
            }
            for view, (rotation, translation) in final_poses.items()
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    save_final_renders(args.output_dir, optimizer, final_poses)
    return result
