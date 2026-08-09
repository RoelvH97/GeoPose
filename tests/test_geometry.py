import torch

from geopose.inference import matrix_to_pose, pose_matrix, view_label_from_alpha
from geopose.models.pose_utils import delta_to_pose, params_to_pose
from geopose.preregistration import closest_rotation


def test_closest_rotation_is_proper_and_orthonormal():
    affine = torch.tensor(
        [[[1.2, 0.2, 0.0], [0.1, 0.8, -0.1], [0.0, 0.3, -1.1]]],
        dtype=torch.float64,
    )
    rotation = closest_rotation(affine)
    identity = torch.eye(3, dtype=torch.float64)[None]
    assert torch.allclose(rotation @ rotation.transpose(1, 2), identity, atol=1e-12)
    assert torch.allclose(torch.linalg.det(rotation), torch.ones(1, dtype=torch.float64))


def test_pose_matrix_roundtrip_preserves_transform():
    rotation = torch.tensor([[0.31, -0.12, 0.08], [-0.7, 0.2, -0.1]])
    translation = torch.tensor([[3.0, 650.0, -4.0], [-5.0, 800.0, 7.0]])
    matrix = pose_matrix(rotation, translation)
    decoded_rotation, decoded_translation = matrix_to_pose(matrix)
    assert torch.allclose(
        pose_matrix(decoded_rotation, decoded_translation), matrix, atol=1e-4
    )


def test_delta_inverse_recovers_optimal_pose():
    optimal = params_to_pose(
        torch.tensor([[0.5, -0.2, 0.1]]),
        torch.tensor([[2.0, 650.0, -3.0]]),
    )
    delta = delta_to_pose(
        torch.tensor([[0.04, -0.03, 0.01]]),
        torch.tensor([[4.0, -12.0, 2.0]]),
    )
    recovered = optimal.compose(delta).compose(delta.inverse())
    assert torch.allclose(recovered.matrix, optimal.matrix, atol=1e-4)


def test_signed_view_label_boundaries():
    assert view_label_from_alpha(80) == 0
    assert view_label_from_alpha(45) == 1
    assert view_label_from_alpha(0) == 1
    assert view_label_from_alpha(-45) == 1
    assert view_label_from_alpha(-80) == 2

