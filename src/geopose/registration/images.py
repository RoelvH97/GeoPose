"""Image normalization and connected-component helpers."""

from __future__ import annotations

import numpy as np
import torch
from scipy.ndimage import label


def largest_component(mask: np.ndarray) -> np.ndarray:
    components, count = label(mask > 0)
    if count == 0:
        return np.zeros_like(mask, dtype=np.float32)
    sizes = np.bincount(components.ravel())
    sizes[0] = 0
    return (components == sizes.argmax()).astype(np.float32)


def minmax(image: torch.Tensor) -> torch.Tensor:
    flat = image.flatten(1)
    minimum = flat.min(1).values.view(-1, 1, 1, 1)
    maximum = flat.max(1).values.view(-1, 1, 1, 1)
    return (image - minimum) / (maximum - minimum).clamp(min=1e-8)
