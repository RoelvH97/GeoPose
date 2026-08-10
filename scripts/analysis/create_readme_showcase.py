#!/usr/bin/env python3
"""Create the animated two-view GeoPose README showcase.

The first four columns use CTA DRRs at the native, calibrated GeoPose-Init,
GeoPose-Refine, and evolving 25-step GeoReg poses. The fifth column is the
fixed DSA MAP. Magenta/cyan contours are the target DSA and projected CTA
cranium silhouettes used by the GeoReg objective; no vessel annotations are
loaded by this script.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from diffdrr.data import read
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import binary_dilation, binary_erosion

from geopose.registration.optimization import (
    TestTimeOptimizer,
    prepare_registration_inputs,
    read_registration_channels,
)


DEFAULT_DATA_ROOT = Path(
    os.environ.get("GEOPOSE_DATA_ROOT", "data/Dataset004_ISLES2024")
)
DEFAULT_GEOREG_ROOT = Path(
    os.environ.get("GEOREG_OUTPUT_ROOT", "../GeoReg/outputs")
)
DEFAULT_PATIENTS = ("sub-stroke0001", "sub-stroke0011", "sub-stroke0079")
DISPLAY_VIEWS = ("pa", "lat")
STAGE_VARIANTS = ("canonical", "geopose_init", "geopose_refine_greedy")

TARGET_COLOR = np.asarray([220, 20, 96], dtype=np.uint8)
PROJECTED_COLOR = np.asarray([0, 188, 212], dtype=np.uint8)
OVERLAP_COLOR = np.asarray([255, 255, 255], dtype=np.uint8)
BACKGROUND_COLOR = (250, 250, 252)
TEXT_COLOR = (28, 31, 36)
MUTED_COLOR = (128, 135, 145)

PANEL = 170
GIF_COLORS = 48
GAP = 10
LEFT = 58
RIGHT = 18
TOP = 76
BOTTOM = 62
WIDTH = LEFT + 5 * PANEL + 4 * GAP + RIGHT
HEIGHT = TOP + 2 * PANEL + GAP + BOTTOM


@dataclass
class CaseFrames:
    patient: str
    stage_images: dict[tuple[str, str], np.ndarray]
    tto_images: dict[str, list[np.ndarray]]
    dsa_images: dict[str, np.ndarray]
    trajectory: dict
    sources: dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--georeg-root", type=Path, default=DEFAULT_GEOREG_ROOT)
    parser.add_argument("--patients", nargs="+", default=list(DEFAULT_PATIENTS))
    parser.add_argument("--timestamp", choices=("pre", "post"), default="pre")
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--spacing", type=float, default=1.2)
    parser.add_argument(
        "--device",
        choices=("auto", "cpu", "cuda"),
        default="auto",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("docs/assets/geopose_showcase.gif"),
    )
    parser.add_argument(
        "--final-frame",
        type=Path,
        default=Path("docs/assets/geopose_showcase_final.png"),
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        default=Path("docs/assets/geopose_showcase.json"),
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    with path.open() as stream:
        return json.load(stream)


def find_record(root: Path, variant: str, patient: str) -> tuple[Path, dict]:
    matches = []
    for path in sorted((root / "instant_pose" / "raw" / variant).glob("*/metrics.json")):
        record = load_json(path)
        if record.get("patient_id") == patient:
            matches.append((path, record))
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected one {variant} record for {patient}, found "
            f"{[str(path) for path, _ in matches]}"
        )
    path, record = matches[0]
    if record.get("status") != "completed":
        raise ValueError(f"Incomplete evaluation record: {path}")
    return path, record


def score_pose(score: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(score["rotation_rad"], dtype=torch.float32, device=device),
        torch.tensor(score["translation_mm"], dtype=torch.float32, device=device),
    )


def clone_poses(
    poses: dict[str, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, tuple[torch.Tensor, torch.Tensor]]:
    return {
        view: tuple(value.detach().clone() for value in pose)
        for view, pose in poses.items()
    }


def optimize_with_snapshots(
    optimizer: TestTimeOptimizer,
    iterations: int,
) -> tuple[list[dict[str, tuple[torch.Tensor, torch.Tensor]]], dict]:
    """Run the frozen optimizer while retaining every true current pose."""
    with torch.no_grad():
        initial_ncc, initial_dice, _ = optimizer.losses()
    best_loss = {view: float(initial_ncc[view]) for view in DISPLAY_VIEWS}
    best_pose = clone_poses(optimizer.snapshot())
    snapshots = [clone_poses(optimizer.snapshot())]
    trace = {
        "iterations": iterations,
        "objective": "0.5 * multiscale NCC + 0.5 * cranium Dice per view",
        "steps": [
            {
                "step": 0,
                "views": {
                    view: {
                        "mncc": -float(initial_ncc[view]),
                        "cranium_dice_loss": float(initial_dice[view]),
                        "best_mncc": -best_loss[view],
                    }
                    for view in DISPLAY_VIEWS
                },
            }
        ],
    }
    if iterations == 0:
        return snapshots, trace

    nadam = torch.optim.NAdam(optimizer.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        nadam,
        max_lr=1e-2,
        total_steps=iterations,
        pct_start=0.3,
    )
    for step in range(1, iterations + 1):
        nadam.zero_grad(set_to_none=True)
        _, _, total = optimizer.losses()
        total.backward()
        nadam.step()
        scheduler.step()
        snapshots.append(clone_poses(optimizer.snapshot()))
        with torch.no_grad():
            current_ncc, current_dice, _ = optimizer.losses()
        for view in DISPLAY_VIEWS:
            value = float(current_ncc[view])
            if value < best_loss[view]:
                best_loss[view] = value
                best_pose[view] = tuple(
                    tensor.detach().clone() for tensor in optimizer.pose(view)
                )
        trace["steps"].append(
            {
                "step": step,
                "learning_rate": float(nadam.param_groups[0]["lr"]),
                "views": {
                    view: {
                        "mncc": -float(current_ncc[view]),
                        "cranium_dice_loss": float(current_dice[view]),
                        "best_mncc": -best_loss[view],
                    }
                    for view in DISPLAY_VIEWS
                },
            }
        )

    # The production method returns the best-mNCC pose independently per view.
    snapshots[-1] = clone_poses(best_pose)
    trace["final_frame_pose"] = "best per-view mNCC including iteration zero"
    return snapshots, trace


def render_pose(
    renderer,
    pose: tuple[torch.Tensor, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    rotation, translation = pose
    with torch.no_grad():
        projection = renderer(
            rotation,
            translation,
            parameterization="euler_angles",
            convention="ZYX",
            mask_to_channels=True,
        )[0]
    grayscale = projection.sum(dim=0).detach().cpu().numpy()
    cranium = projection[1].detach().cpu().numpy() > 0
    return grayscale.astype(np.float32), cranium


def robust_window(arrays: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([array[array > 0].ravel() for array in arrays if np.any(array > 0)])
    if not values.size:
        return 0.0, 1.0
    low, high = np.percentile(values, [5.0, 99.5])
    if high <= low:
        high = low + 1.0
    return float(low), float(high)


def normalize(array: np.ndarray, window: tuple[float, float] | None = None) -> np.ndarray:
    array = np.asarray(array, dtype=np.float32)
    if window is None:
        positive = array[array > 0]
        if positive.size:
            low, high = np.percentile(positive, [1.0, 99.0])
        else:
            low, high = float(array.min()), float(array.max())
    else:
        low, high = window
    if high <= low:
        high = low + 1.0
    return np.clip((array - low) / (high - low), 0.0, 1.0)


def mask_outline(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask) > 0.5
    outline = binary ^ binary_erosion(binary, iterations=1)
    return binary_dilation(outline, iterations=1)


def overlay_cranium(
    background: np.ndarray,
    target_mask: np.ndarray,
    projected_mask: np.ndarray,
) -> np.ndarray:
    gray = (normalize(background) * 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    target = mask_outline(target_mask)
    projected = mask_outline(projected_mask)
    halo = binary_dilation(target | projected, iterations=1)
    rgb[halo] = (rgb[halo].astype(np.float32) * 0.25).astype(np.uint8)
    rgb[target] = TARGET_COLOR
    rgb[projected] = PROJECTED_COLOR
    rgb[target & projected] = OVERLAP_COLOR
    return rgb


def overlay_with_window(
    background: np.ndarray,
    window: tuple[float, float],
    target_mask: np.ndarray,
    projected_mask: np.ndarray,
) -> np.ndarray:
    gray = (normalize(background, window) * 255).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    target = mask_outline(target_mask)
    projected = mask_outline(projected_mask)
    halo = binary_dilation(target | projected, iterations=1)
    rgb[halo] = (rgb[halo].astype(np.float32) * 0.25).astype(np.uint8)
    rgb[target] = TARGET_COLOR
    rgb[projected] = PROJECTED_COLOR
    rgb[target & projected] = OVERLAP_COLOR
    return rgb


def build_case(args: argparse.Namespace, patient: str, device: torch.device) -> CaseFrames:
    records = {}
    sources = {}
    for variant in STAGE_VARIANTS:
        path, record = find_record(args.georeg_root, variant, patient)
        records[variant] = record
        sources[variant] = {"path": str(path.resolve()), "sha256": sha256(path)}

    indices = {record["dataset_index"] for record in records.values()}
    if len(indices) != 1:
        raise ValueError(f"Stage records disagree on dataset index for {patient}: {indices}")

    cta_path = args.data_root / "CTATr" / f"{patient}_0000.nii.gz"
    cranium_path = args.data_root / "CTA_skullTr" / f"{patient}.nii.gz"
    for path in (cta_path, cranium_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    cta_subject = read(str(cta_path), str(cranium_path), labels=[0, 1])

    raw_dsa, raw_masks, _ = read_registration_channels(
        args.data_root,
        patient,
        args.timestamp,
        size=args.size,
    )
    images, cranium_masks, metadata = prepare_registration_inputs(
        args.data_root,
        patient,
        args.timestamp,
        device,
        size=args.size,
    )

    stage_poses = {
        variant: {
            view: score_pose(record["scores"][args.timestamp][view], device)
            for view in DISPLAY_VIEWS
        }
        for variant, record in records.items()
    }
    optimizer = TestTimeOptimizer(
        cta_subject,
        images,
        cranium_masks,
        metadata,
        stage_poses["geopose_refine_greedy"],
        device,
        size=args.size,
        spacing=args.spacing,
    ).to(device)
    snapshots, trajectory = optimize_with_snapshots(optimizer, args.iterations)

    rendered = {view: {} for view in DISPLAY_VIEWS}
    rendered_masks = {view: {} for view in DISPLAY_VIEWS}
    tto_rendered = {view: [] for view in DISPLAY_VIEWS}
    tto_masks = {view: [] for view in DISPLAY_VIEWS}
    for view in DISPLAY_VIEWS:
        renderer = optimizer.renderers[view]
        for variant in STAGE_VARIANTS:
            image, mask = render_pose(renderer, stage_poses[variant][view])
            rendered[view][variant] = image
            rendered_masks[view][variant] = mask
        for poses in snapshots:
            image, mask = render_pose(renderer, poses[view])
            tto_rendered[view].append(image)
            tto_masks[view].append(mask)

    stage_images = {}
    tto_images = {}
    dsa_images = {}
    for view in DISPLAY_VIEWS:
        all_drrs = list(rendered[view].values()) + tto_rendered[view]
        window = robust_window(all_drrs)
        target_mask = raw_masks[view]
        for variant in STAGE_VARIANTS:
            stage_images[(variant, view)] = overlay_with_window(
                rendered[view][variant],
                window,
                target_mask,
                rendered_masks[view][variant],
            )
        tto_images[view] = [
            overlay_with_window(image, window, target_mask, mask)
            for image, mask in zip(tto_rendered[view], tto_masks[view])
        ]
        dsa_images[view] = overlay_cranium(
            raw_dsa[view],
            target_mask,
            tto_masks[view][-1],
        )

    trajectory["patient"] = patient
    trajectory["dataset_index"] = next(iter(indices))
    trajectory["timestamp"] = args.timestamp
    trajectory["final_best_mncc"] = {
        view: trajectory["steps"][-1]["views"][view]["best_mncc"]
        for view in DISPLAY_VIEWS
    }
    sources["cta"] = {"path": str(cta_path.resolve()), "sha256": sha256(cta_path)}
    sources["cta_cranium"] = {
        "path": str(cranium_path.resolve()),
        "sha256": sha256(cranium_path),
    }
    return CaseFrames(
        patient=patient,
        stage_images=stage_images,
        tto_images=tto_images,
        dsa_images=dsa_images,
        trajectory=trajectory,
        sources=sources,
    )


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / name
    return ImageFont.truetype(str(path), size=size)


def paste_panel(canvas: Image.Image, array: np.ndarray, column: int, row: int) -> None:
    panel = Image.fromarray(array).resize((PANEL, PANEL), Image.Resampling.LANCZOS)
    x = LEFT + column * (PANEL + GAP)
    y = TOP + row * (PANEL + GAP)
    canvas.paste(panel, (x, y))


def draw_hidden(draw: ImageDraw.ImageDraw, column: int, row: int) -> None:
    x = LEFT + column * (PANEL + GAP)
    y = TOP + row * (PANEL + GAP)
    draw.rounded_rectangle(
        (x, y, x + PANEL, y + PANEL),
        radius=5,
        fill=(240, 242, 246),
        outline=(222, 226, 232),
        width=1,
    )


def draw_centered(
    draw: ImageDraw.ImageDraw,
    xy: tuple[float, float],
    text: str,
    selected_font,
    fill=TEXT_COLOR,
) -> None:
    draw.multiline_text(
        xy,
        text,
        font=selected_font,
        fill=fill,
        anchor="mm",
        align="center",
        spacing=1,
    )


def compose_frame(
    case: CaseFrames,
    case_number: int,
    case_count: int,
    reveal: int,
    tto_iteration: int,
) -> Image.Image:
    canvas = Image.new("RGB", (WIDTH, HEIGHT), BACKGROUND_COLOR)
    draw = ImageDraw.Draw(canvas)
    draw_centered(
        draw,
        (WIDTH / 2, 17),
        f"GeoPose registration  ·  held-out test case {case_number}/{case_count}",
        font(17, bold=True),
    )

    headings = (
        "Native pose",
        "GeoPose-Init\n+ calibration",
        "GeoPose-Refine",
        f"GeoReg\niteration {tto_iteration:02d}/25",
        "DSA MAP",
    )
    visible = (True, reveal >= 1, reveal >= 2, reveal >= 3, True)
    for column, heading in enumerate(headings):
        x = LEFT + column * (PANEL + GAP) + PANEL / 2
        draw_centered(
            draw,
            (x, 52),
            heading,
            font(12, bold=True),
            TEXT_COLOR if visible[column] else MUTED_COLOR,
        )
        for row in range(2):
            if not visible[column]:
                draw_hidden(draw, column, row)

    for row, view in enumerate(DISPLAY_VIEWS):
        draw_centered(
            draw,
            (27, TOP + row * (PANEL + GAP) + PANEL / 2),
            view.upper(),
            font(14, bold=True),
        )
        paste_panel(canvas, case.stage_images[("canonical", view)], 0, row)
        paste_panel(canvas, case.dsa_images[view], 4, row)
        if reveal >= 1:
            paste_panel(canvas, case.stage_images[("geopose_init", view)], 1, row)
        if reveal >= 2:
            paste_panel(
                canvas,
                case.stage_images[("geopose_refine_greedy", view)],
                2,
                row,
            )
        if reveal >= 3:
            paste_panel(canvas, case.tto_images[view][tto_iteration], 3, row)

    legend_y = HEIGHT - 28
    center = WIDTH / 2
    draw.line((center - 205, legend_y, center - 165, legend_y), fill=tuple(TARGET_COLOR), width=5)
    draw.text((center - 155, legend_y), "DSA cranium", font=font(11), fill=TEXT_COLOR, anchor="lm")
    draw.line((center + 5, legend_y, center + 45, legend_y), fill=tuple(PROJECTED_COLOR), width=5)
    draw.text((center + 55, legend_y), "Projected CTA cranium", font=font(11), fill=TEXT_COLOR, anchor="lm")
    return canvas


def frame_schedule(iterations: int) -> list[tuple[int, int, int]]:
    schedule = [(0, 0, 500), (1, 0, 500), (2, 0, 500), (3, 0, 300)]
    schedule.extend((3, step, 75) for step in range(1, iterations))
    schedule.append((3, iterations, 1400))
    return schedule


def save_animation(
    cases: list[CaseFrames],
    output: Path,
    final_frame: Path,
    iterations: int,
) -> tuple[int, int]:
    frames = []
    durations = []
    schedule = frame_schedule(iterations)
    for case_number, case in enumerate(cases, start=1):
        for reveal, step, duration in schedule:
            frames.append(
                compose_frame(case, case_number, len(cases), reveal, step)
            )
            durations.append(duration)

    output.parent.mkdir(parents=True, exist_ok=True)
    first = frames[0].quantize(colors=GIF_COLORS, method=Image.Quantize.MEDIANCUT)
    paletted = [first]
    paletted.extend(
        frame.quantize(palette=first, dither=Image.Dither.NONE)
        for frame in frames[1:]
    )
    paletted[0].save(
        output,
        save_all=True,
        append_images=paletted[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    final_frame.parent.mkdir(parents=True, exist_ok=True)
    frames[-1].save(final_frame, optimize=True)
    return len(frames), sum(durations)


def main() -> None:
    args = parse_args()
    if args.iterations != 25:
        raise ValueError("The README storyboard and labels require exactly 25 iterations")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)

    cases = []
    for patient in args.patients:
        print(f"Preparing {patient} on {device}", flush=True)
        cases.append(build_case(args, patient, device))
    frame_count, duration_ms = save_animation(
        cases,
        args.output,
        args.final_frame,
        args.iterations,
    )

    provenance = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "README showcase of the staged GeoPose registration workflow",
        "selection": {
            "patients": list(args.patients),
            "split": "frozen 20-patient GeoPose test split",
            "timestamp": args.timestamp,
            "policy": "three prespecified held-out cases spanning the test split; pre-intervention pair only",
        },
        "display": {
            "view_order": list(DISPLAY_VIEWS),
            "columns": [
                "native pose",
                "GeoPose-Init with isopose calibration",
                "GeoPose-Refine",
                "true 25-step GeoReg optimizer trajectory",
                "fixed DSA MAP at final retained pose",
            ],
            "contours": {
                "magenta": "target DSA cranium silhouette",
                "cyan": "projected CTA cranium silhouette",
                "white": "coincident contours",
            },
            "vessel_annotations_used": False,
        },
        "animation": {
            "frames": frame_count,
            "duration_ms": duration_ms,
            "loop": True,
            "dimensions_px": [WIDTH, HEIGHT],
        },
        "rendering": {
            "device": str(device),
            "detector_size_px": args.size,
            "detector_spacing_mm": args.spacing,
            "parameterization": "euler_angles",
            "convention": "ZYX",
            "optimizer": "NAdam + OneCycleLR",
            "objective": "0.5 * multiscale NCC + 0.5 * cranium Dice per view",
            "final_pose_selection": "best per-view mNCC including iteration zero",
        },
        "cases": [
            {
                "patient": case.patient,
                "trajectory": case.trajectory,
                "sources": case.sources,
            }
            for case in cases
        ],
        "outputs": {
            "gif": str(args.output.resolve()),
            "gif_sha256": sha256(args.output),
            "gif_bytes": args.output.stat().st_size,
            "final_frame": str(args.final_frame.resolve()),
            "final_frame_sha256": sha256(args.final_frame),
        },
    }
    args.provenance.parent.mkdir(parents=True, exist_ok=True)
    args.provenance.write_text(json.dumps(provenance, indent=2) + "\n")
    print(
        f"Saved {args.output} ({args.output.stat().st_size / 1024 / 1024:.2f} MiB, "
        f"{frame_count} frames, {duration_ms / 1000:.2f} s)",
        flush=True,
    )
    print(f"Saved {args.final_frame}", flush=True)
    print(f"Saved {args.provenance}", flush=True)


if __name__ == "__main__":
    main()
