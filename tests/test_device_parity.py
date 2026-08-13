import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from slai_mi.devices.iphone.mapping import RelativeIPhoneToTcpMapper, rate_limit_tcp_target
from slai_mi.devices.iphone.visualization import static_pose_metrics
from slai_mi.devices.wujihand.grasp_pose import load_grasp_pose, save_grasp_pose
from slai_mi.devices.wujihand.pose import transform_to_ur_pose, ur_pose_to_transform
from slai_mi.devices.wujihand.tracking import LandmarkGate, reconstruct_missing_keypoints
from slai_mi.retargeting.calibration import (
    DualCameraExtrinsics,
    load_dual_camera_extrinsics,
    robust_extrinsic_fit,
)


def test_ur_pose_round_trip() -> None:
    pose = np.array([0.1, -0.2, 0.3, 0.2, -0.1, 0.05])
    assert np.allclose(transform_to_ur_pose(ur_pose_to_transform(pose)), pose)


def test_landmark_reconstruction_and_gate() -> None:
    points = np.arange(63, dtype=float).reshape(21, 3) / 1000
    expected = points[2].copy()
    points[2] = np.nan
    assert np.allclose(reconstruct_missing_keypoints(points)[2], expected)
    gate = LandmarkGate(max_speed_m_s=1)
    assert gate.filter(np.zeros((21, 3)), timestamp=1, confidence=1) is not None
    assert gate.filter(np.ones((21, 3)), timestamp=1.01, confidence=1) is None


def test_grasp_pose_is_device_bound(tmp_path) -> None:
    path = tmp_path / "pose.json"
    save_grasp_pose(path, np.arange(20) / 100, device_serial="hand-a")
    assert load_grasp_pose(path, device_serial="hand-a").shape == (20,)
    with pytest.raises(RuntimeError):
        load_grasp_pose(path, device_serial="hand-b")


def test_iphone_mapping_rate_limits_and_metrics() -> None:
    mapper = RelativeIPhoneToTcpMapper(workspace_radius_m=np.full(3, 0.1))
    neutral = np.eye(4)
    mapper.reset(neutral, neutral)
    moved = neutral.copy()
    moved[:3, 3] = [1, 0, 0]
    assert np.max(np.abs(mapper.target(moved)[:3, 3])) <= 0.1
    limited = rate_limit_tcp_target(neutral, moved, 0.1, maximum_translation_speed_m_s=0.2, maximum_rotation_speed_rad_s=1)
    assert np.linalg.norm(limited[:3, 3]) == pytest.approx(0.02)
    assert static_pose_metrics([neutral, neutral]).rotation_peak_deg == 0


def test_robust_extrinsics_and_persistence(tmp_path) -> None:
    rng = np.random.default_rng(4)
    source = rng.normal(size=(30, 3))
    rotation = Rotation.from_euler("xyz", [0.1, -0.2, 0.3]).as_matrix()
    target = source @ rotation.T + [0.2, 0.3, -0.1]
    transform, residuals = robust_extrinsic_fit(source, target)
    assert np.nanmax(residuals) < 1e-10
    artifact = DualCameraExtrinsics("a", "b", transform, 0, 30, 30)
    path = tmp_path / "extrinsics.json"
    artifact.save(path)
    assert np.allclose(load_dual_camera_extrinsics(path).primary_from_secondary, transform)
