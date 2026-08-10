import importlib.util
from pathlib import Path
import sys

import numpy as np
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analysis/create_readme_showcase.py"
ASSET = Path(__file__).resolve().parents[1] / "docs/assets/geopose_showcase.gif"
SPEC = importlib.util.spec_from_file_location("create_readme_showcase", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_frame_schedule_reveals_stages_then_runs_all_25_iterations():
    schedule = MODULE.frame_schedule(25)
    assert schedule[:4] == [
        (0, 0, 500),
        (1, 0, 500),
        (2, 0, 500),
        (3, 0, 300),
    ]
    assert [step for reveal, step, _ in schedule if reveal == 3] == list(range(26))
    assert schedule[-1] == (3, 25, 1400)


def test_cranium_overlay_uses_target_projected_and_overlap_colors():
    background = np.arange(25, dtype=np.float32).reshape(5, 5)
    target = np.zeros((5, 5), dtype=bool)
    projected = np.zeros((5, 5), dtype=bool)
    target[1:4, 1:4] = True
    projected[2:5, 2:5] = True
    image = MODULE.overlay_cranium(background, target, projected)
    assert image.shape == (5, 5, 3)
    colors = {tuple(pixel) for pixel in image.reshape(-1, 3)}
    assert tuple(MODULE.TARGET_COLOR) in colors
    assert tuple(MODULE.PROJECTED_COLOR) in colors
    assert tuple(MODULE.OVERLAP_COLOR) in colors


def test_composed_frame_has_readme_friendly_dimensions():
    assert MODULE.WIDTH == 966
    assert MODULE.HEIGHT == 488


def test_committed_gif_obeys_readme_asset_contract():
    with Image.open(ASSET) as animation:
        assert animation.size == (MODULE.WIDTH, MODULE.HEIGHT)
        assert animation.n_frames == 3 * len(MODULE.frame_schedule(25))
        assert animation.info["loop"] == 0
        durations = []
        for frame in range(animation.n_frames):
            animation.seek(frame)
            durations.append(animation.info["duration"])

    assert durations[:4] == [500, 500, 500, 300]
    assert durations[29:33] == [500, 500, 500, 300]
    assert durations[58:62] == [500, 500, 500, 300]
    assert durations[-1] == 1400
    assert ASSET.stat().st_size < 10 * 1024 * 1024
