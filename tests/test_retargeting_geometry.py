import pytest

np = pytest.importorskip("numpy")

from slai_mi.retargeting import estimate_rigid_transform, fuse_landmarks, hand_frame_from_keypoints


def test_rigid_transform_recovery():
    child = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    parent = child + np.array([1, 2, 3])
    transform, residuals = estimate_rigid_transform(child, parent)
    assert np.allclose(transform[:3, 3], [1, 2, 3])
    assert np.allclose(residuals, 0)


def test_hand_frame_is_rigid():
    points = np.zeros((21, 3))
    points[0] = [0, 0, 0]
    points[5] = [1, 1, 0]
    points[9] = [0, 2, 0]
    points[13] = [-0.5, 1, 0]
    points[17] = [-1, 1, 0]
    transform = hand_frame_from_keypoints(points)
    assert np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3))


def test_multiview_fusion_transforms_secondary_camera():
    primary = np.zeros((21, 3))
    secondary = np.zeros((21, 3))
    transform = np.eye(4)
    transform[0, 3] = 2.0
    fused, confidence = fuse_landmarks(primary, secondary, np.ones(21), np.ones(21), transform)
    assert np.allclose(fused[:, 0], 1.0)
    assert np.allclose(confidence, 1.0)
