import os
from pathlib import Path

import pytest
import torch

from geopose.inference import load_init_model, load_refine_model


INIT_ENV = "GEOPOSE_INIT_CHECKPOINT"
REFINE_ENV = "GEOPOSE_REFINE_CHECKPOINT"


@pytest.mark.integration
def test_publication_checkpoints_strict_load_from_frozen_contracts():
    init_path = os.environ.get(INIT_ENV)
    refine_path = os.environ.get(REFINE_ENV)
    if not init_path or not refine_path:
        pytest.skip(f"set {INIT_ENV} and {REFINE_ENV} to run checkpoint integration")
    init = load_init_model(Path(init_path), torch.device("cpu"))
    refine = load_refine_model(Path(refine_path), torch.device("cpu"))
    assert sum(parameter.numel() for parameter in init.parameters()) == 21_283_674
    assert sum(parameter.numel() for parameter in refine.parameters()) == 22_592_854

