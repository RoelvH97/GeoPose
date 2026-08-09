"""End-to-end GeoPose publication inference orchestration."""

from __future__ import annotations

import argparse
import json

import torch
from diffdrr.data import read

from .geometry import pose_matrix, tensor_list
from .initialization import PoseInitializer, file_sha256, load_init_model, load_refine_model
from .optimization import TestTimeOptimizer, prepare_registration_inputs, save_final_renders


def run_inference(args: argparse.Namespace) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("The publication inference pipeline requires CUDA")
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    device = torch.device("cuda")
    data_root = args.data_root.resolve()
    patient = args.patient

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
        patient, args.timestamp
    )

    images, masks, metadata = prepare_registration_inputs(
        data_root, patient, args.timestamp, device
    )
    optimizer = TestTimeOptimizer(
        cta_subject,
        images,
        masks,
        metadata,
        initial_poses,
        device,
    ).to(device)
    final_poses, optimization_trace = optimizer.optimize(args.iterations)

    result = {
        "schema_version": 1,
        "contract": "geopose-inference-v1",
        "patient": patient,
        "timestamp": args.timestamp,
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
        "example_note": (
            "sub-stroke9999 is the alignment template and a functional public "
            "example, not an independent held-out evaluation case."
            if patient == "sub-stroke9999"
            else None
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    save_final_renders(args.output_dir, optimizer, final_poses)
    return result
