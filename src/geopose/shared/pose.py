"""Pose composition and delta sampling for the refinement CNN."""

from __future__ import annotations

import torch

from diffdrr.pose import convert


def _broadcast_std(std, device, dtype=torch.float32):
    if isinstance(std, (int, float)):
        std = (float(std),) * 3
    return torch.tensor(tuple(std), dtype=dtype, device=device)


def sample_delta(
    batch_size: int,
    std_R,
    std_t,
    device: torch.device | str = "cpu",
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample 6-DOF SE(3) tangent deltas."""
    std_R_t = _broadcast_std(std_R, device)
    std_t_t = _broadcast_std(std_t, device)
    dR = torch.randn(batch_size, 3, device=device, generator=generator) * std_R_t
    dt = torch.randn(batch_size, 3, device=device, generator=generator) * std_t_t
    return dR, dt


def params_to_pose(R_euler: torch.Tensor, t: torch.Tensor):
    """Convert Euler-ZYX rotations and translations to DiffDRR poses."""
    return convert(R_euler, t, parameterization="euler_angles", convention="ZYX")


def delta_to_pose(dR: torch.Tensor, dt: torch.Tensor):
    """Convert tangent-space pose deltas to DiffDRR poses."""
    return convert(dR, dt, parameterization="axis_angle", convention=None)
