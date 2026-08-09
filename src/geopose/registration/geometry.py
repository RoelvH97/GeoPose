"""DiffDRR pose conversion and JSON-safe tensor serialization."""

from __future__ import annotations

import torch
from diffdrr.pose import convert
from pytorch3d.transforms import matrix_to_euler_angles


def pose_matrix(rotation: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    return convert(
        rotation,
        translation,
        parameterization="euler_angles",
        convention="ZYX",
    ).matrix


def matrix_to_pose(matrix: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    rotation_matrix = matrix[:, :3, :3]
    rotation = matrix_to_euler_angles(rotation_matrix, convention="ZYX")
    translation = torch.einsum(
        "bij,bj->bi", rotation_matrix.transpose(1, 2), matrix[:, :3, 3]
    )
    return rotation, translation


def tensor_list(value: torch.Tensor) -> list:
    return value.detach().cpu().tolist()
