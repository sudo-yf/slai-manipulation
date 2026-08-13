from datetime import UTC, datetime

import numpy as np
import pytest

from slai_mi.devices.ur5.config import HARDWARE_CONFIRMATION, UR5TeleopConfig
from slai_mi.devices.ur5.geometry import (
    apply_relative_workspace_guard,
    home_twist,
    joint_home_velocity,
)
from slai_mi.devices.ur5.zero_pose import load_zero_pose, save_zero_pose


def test_hardware_motion_requires_exact_confirmation() -> None:
    with pytest.raises(ValueError, match="hardware motion requires"):
        UR5TeleopConfig(robot_host="192.0.2.10", enable_hardware=True).validate()

    with pytest.raises(ValueError, match="robot host must not be empty"):
        UR5TeleopConfig().validate()

    UR5TeleopConfig(
        robot_host="192.0.2.10",
        enable_hardware=True,
        confirmation=HARDWARE_CONFIRMATION,
    ).validate()


def test_workspace_guard_blocks_outward_motion_but_allows_recovery() -> None:
    start = np.zeros(6)
    current = np.array([0.099, 0.0, 0.0, 0.0, 0.0, 0.0])

    outward, reason = apply_relative_workspace_guard(
        np.array([0.02, 0.0, 0.0, 0.0, 0.0, 0.0]), current, start, 0.1, 0.0, 0.25
    )
    inward, inward_reason = apply_relative_workspace_guard(
        np.array([-0.02, 0.0, 0.0, 0.0, 0.0, 0.0]), current, start, 0.1, 0.0, 0.25
    )

    np.testing.assert_array_equal(outward[:3], np.zeros(3))
    assert reason == "translation boundary"
    assert inward[0] < 0.0
    assert inward_reason is None


def test_home_commands_are_bounded() -> None:
    twist, reached = home_twist(np.zeros(6), np.ones(6), 0.1, 0.2)
    joints, joints_reached = joint_home_velocity(np.zeros(6), np.ones(6), 0.3)

    assert not reached
    assert not joints_reached
    assert np.linalg.norm(twist[:3]) == pytest.approx(0.1)
    assert np.linalg.norm(twist[3:]) == pytest.approx(0.2)
    assert np.max(np.abs(joints)) == pytest.approx(0.3)


def test_zero_pose_round_trip_and_robot_identity(tmp_path) -> None:
    path = tmp_path / "zero.json"
    pose = np.arange(6, dtype=np.float64) / 10.0
    save_zero_pose(
        path,
        "192.0.2.10",
        pose,
        recorded_at=datetime(2026, 1, 2, tzinfo=UTC),
    )

    np.testing.assert_array_equal(load_zero_pose(path, "192.0.2.10"), pose)
    with pytest.raises(RuntimeError, match="robot host"):
        load_zero_pose(path, "192.0.2.11")
