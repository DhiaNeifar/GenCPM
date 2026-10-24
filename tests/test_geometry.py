import numpy as np

from gencpm.evaluation.geometry import pose_to_matrix


def test_pose_matrix_is_a_rigid_transform():
    rng = np.random.default_rng(0)
    for _ in range(100):
        m = pose_to_matrix(list(rng.uniform(-100, 100, 3)) + list(rng.uniform(-180, 180, 3)))
        np.testing.assert_allclose(m[:3, :3] @ m[:3, :3].T, np.eye(3), atol=1e-12)
        assert np.isclose(np.linalg.det(m[:3, :3]), 1.0)


def test_yaw_rotates_x_to_y():
    m = pose_to_matrix([1.0, 2.0, 3.0, 0.0, 90.0, 0.0])
    np.testing.assert_allclose(m @ [1, 0, 0, 1], [1, 3, 3, 1], atol=1e-12)
