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


def _roundtrip_check() -> None:
    """Sanity check the SE(3) compose pipeline:"""
    torch.manual_seed(0)
    B = 4

    R = torch.tensor(
        [
            [1.497, 0.106, 0.222],
            [-0.053, 0.195, -0.076],
            [1.50, -0.10, 0.05],
            [0.02, 0.30, -0.12],
        ]
    )
    t = torch.tensor(
        [
            [3.93, 647.2, 2.11],
            [-0.70, 585.6, -5.55],
            [10.0, 700.0, -5.0],
            [-2.0, 900.0, 1.0],
        ]
    )
    dR, dt = sample_delta(B, std_R=0.05, std_t=15.0)

    optimal_pose = params_to_pose(R, t)
    delta_pose = delta_to_pose(dR, dt)
    noisy_pose = optimal_pose.compose(delta_pose)
    corrected_pose = noisy_pose.compose(delta_pose.inverse())

    noisy_gap = (noisy_pose.matrix - optimal_pose.matrix).abs().max().item()
    assert noisy_gap > 1.0, f"noisy unexpectedly close to optimal: {noisy_gap}"

    recovery_err = (corrected_pose.matrix - optimal_pose.matrix).abs().max().item()
    assert recovery_err < 1e-3, f"SE(3) recovery failed: max err {recovery_err}"

    id_pose = delta_to_pose(torch.zeros(1, 3), torch.zeros(1, 3))
    id_err = (id_pose.matrix - torch.eye(4)).abs().max().item()
    assert id_err < 1e-6, f"zero tangent didn't produce identity: {id_err}"

    print("pose_utils SE(3) compose check OK")
    print(f"  ||noisy − optimal||_max     = {noisy_gap:.4f}  (should be > 1)")
    print(f"  ||corrected − optimal||_max = {recovery_err:.2e}  (must be ≈ 0)")
    print(f"  ||δ=0 pose − I||_max         = {id_err:.2e}  (must be ≈ 0)")


if __name__ == "__main__":
    _roundtrip_check()
