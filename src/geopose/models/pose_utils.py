"""Pose composition and delta sampling for the refinement CNN.

Refinement framing
------------------
The optimal pose comes from the manifest in **Euler-ZYX (rad) + translation
(mm)** — the same convention as the initial-guess head in
:mod:`geopose.models.resnet_pose`. The noisy training pose is built via **SE(3)
right-composition** with a sampled delta::

    noisy_pose = optimal_pose ∘ delta_pose

The delta is parameterised as **axis-angle (rad) + translation (mm)** — the
natural SE(3) tangent — so the materialisation in
:func:`delta_to_pose` is the only step that fixes a convention. Both poses
end up as :class:`diffdrr.pose.RigidTransform` (4x4 SE(3) matrices) and
:meth:`RigidTransform.compose` does the multiplication on the manifold.

At inference the network's predicted ``(dR_hat, dt_hat)`` is interpreted as
the same axis-angle/translation tangent and composed-with-inverse to recover
optimal::

    corrected_pose = noisy_pose ∘ delta_pred^{-1}

Why SE(3) compose instead of parameter-space addition? With ``M = R · T(t)``
and a 650 mm translation lever arm, additive (R, t) perturbations don't
correspond to a clean SE(3) operation — a small rotation applied at that
lever arm produces tens of mm of displacement. The matched pair "right-
compose to generate noise, right-compose-with-inverse to correct" recovers
optimal exactly when δ_pred = δ; the older "additive in parameter space"
pair did too, but only because the encoder and decoder were both
parameter-space, leaving no SE(3) tangent semantics for the network output.

This module stays small. Matrix construction, composition, and geodesic
metrics live in DiffDRR; we own only the noise sampling and a couple of
helpers that pin the convention in one place.
"""

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
    """Sample 6-DOF SE(3) tangent deltas.

    ``dR ~ N(0, diag(std_R)^2)`` (axis-angle, radians; magnitude = rotation
    angle, direction = rotation axis) and ``dt ~ N(0, diag(std_t)^2)``
    (translation, millimetres).

    The std-per-axis interpretation matches the per-axis std used in the
    older Euler-ZYX convention to first order — magnitudes of the same
    order produce poses of the same order of magnitude. Calibrated values
    in :file:`configs/refine_pose.yaml` were tuned against the Euler
    parameterisation but transfer cleanly because the std magnitudes
    (≤ 0.14 rad) are well within the small-angle regime where axis-angle
    and Euler agree.

    Args:
        batch_size: number of poses to sample.
        std_R: scalar or 3-tuple of per-axis std for axis-angle rotation (rad).
        std_t: scalar or 3-tuple of per-axis std for translation (mm).
        device: torch device for the output tensors.
        generator: optional ``torch.Generator`` for reproducibility.

    Returns:
        ``(dR, dt)`` each of shape ``[batch_size, 3]``.
    """
    std_R_t = _broadcast_std(std_R, device)
    std_t_t = _broadcast_std(std_t, device)
    dR = torch.randn(batch_size, 3, device=device, generator=generator) * std_R_t
    dt = torch.randn(batch_size, 3, device=device, generator=generator) * std_t_t
    return dR, dt


def params_to_pose(R_euler: torch.Tensor, t: torch.Tensor):
    """Materialise ``(R_euler, t)`` (Euler-ZYX rad + translation mm) to a
    DiffDRR :class:`RigidTransform`.

    Used for poses that come from the manifest or from the initial-guess
    network — both live in Euler-ZYX by convention. Delta tangents (network
    outputs, sampled noise) use :func:`delta_to_pose` instead.
    """
    return convert(R_euler, t, parameterization="euler_angles", convention="ZYX")


def delta_to_pose(dR: torch.Tensor, dt: torch.Tensor):
    """Materialise an SE(3) tangent ``(dR axis-angle rad, dt mm)`` to a
    DiffDRR :class:`RigidTransform`.

    The tangent rotation is parameterised in axis-angle (3-vector, magnitude
    = angle in radians, direction = rotation axis). Identity tangent
    (``dR = dt = 0``) materialises to the identity matrix exactly, which
    means an all-zero network output applied via right-compose-with-inverse
    leaves the noisy pose unchanged — exactly the desired "no correction"
    behaviour at init.
    """
    return convert(dR, dt, parameterization="axis_angle", convention=None)


def _roundtrip_check() -> None:
    """Sanity check the SE(3) compose pipeline:

    * For sampled δ, ``noisy = optimal ∘ delta_to_pose(δ)`` is materially
      different from ``optimal`` (noise actually moves us).
    * ``noisy ∘ delta_to_pose(δ)^{-1} == optimal`` exactly (matched
      composition recovers the target).
    * Identity tangent ``δ = 0`` materialises to the SE(3) identity.

    Run with ``python -m geopose.models.pose_utils``.
    """
    torch.manual_seed(0)
    B = 4
    # Plausible per-view poses drawn from the manifest's pose range.
    R = torch.tensor(
        [
            [1.497, 0.106, 0.222],   # sub-stroke0001 pre LAT
            [-0.053, 0.195, -0.076], # sub-stroke0001 pre PA
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

    # Noisy should differ materially from optimal.
    noisy_gap = (noisy_pose.matrix - optimal_pose.matrix).abs().max().item()
    assert noisy_gap > 1.0, f"noisy unexpectedly close to optimal: {noisy_gap}"

    # And the matched correction must recover optimal to float tolerance.
    # 1e-3 absolute is float32 chained-compose noise at ~700 mm translation scale
    # (matrix entries ~1e3 → ~1e-7 relative precision per op, ×3 ops, ×lever arm).
    recovery_err = (corrected_pose.matrix - optimal_pose.matrix).abs().max().item()
    assert recovery_err < 1e-3, f"SE(3) recovery failed: max err {recovery_err}"

    # Identity tangent → identity SE(3).
    id_pose = delta_to_pose(torch.zeros(1, 3), torch.zeros(1, 3))
    id_err = (id_pose.matrix - torch.eye(4)).abs().max().item()
    assert id_err < 1e-6, f"zero tangent didn't produce identity: {id_err}"

    print("pose_utils SE(3) compose check OK")
    print(f"  ||noisy − optimal||_max     = {noisy_gap:.4f}  (should be > 1)")
    print(f"  ||corrected − optimal||_max = {recovery_err:.2e}  (must be ≈ 0)")
    print(f"  ||δ=0 pose − I||_max         = {id_err:.2e}  (must be ≈ 0)")


if __name__ == "__main__":
    _roundtrip_check()
