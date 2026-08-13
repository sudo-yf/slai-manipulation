import json

import numpy as np
import pytest

from slai_mi.datasets.lerobot_v3.combined_zero_pose import (
    load_ur5_button4_joints,
    load_wuji_button4_joints,
)
from slai_mi.datasets.lerobot_v3.schema import UR5_JOINT_NAMES, WUJI_JOINT_NAMES
from slai_mi.datasets.lerobot_v3.task_start_pose import load_task_start


def test_combined_zero_pose_checks_device_identity(tmp_path) -> None:
    path = tmp_path / "home.json"
    path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "actual_dimension": 26,
                "ur5": {
                    "dimension": 6,
                    "robot_host": "192.0.2.1",
                    "joint_names": list(UR5_JOINT_NAMES),
                    "actual_joint_positions_rad": [0.0] * 6,
                },
                "wuji_hand1": {
                    "dimension": 20,
                    "usb_serial": "usb-1",
                    "product_serial": "hand-1",
                    "joint_names": list(WUJI_JOINT_NAMES),
                    "actual_joint_positions_rad": [0.0] * 20,
                },
            }
        )
    )
    assert load_ur5_button4_joints(path, "192.0.2.1").shape == (6,)
    assert load_wuji_button4_joints(path, usb_serial="usb-1", product_serial="hand-1").shape == (20,)
    with pytest.raises(RuntimeError, match="host"):
        load_ur5_button4_joints(path, "192.0.2.2")


def test_task_start_pose_uses_named_26_dof_schema(tmp_path) -> None:
    hand_state = {
        "joint_names": list(WUJI_JOINT_NAMES),
        "target_joint_positions_rad": [0.0] * 20,
    }
    payload = {
        "format_version": 1,
        "task": {"id": "block_into_box"},
        "actual_dimension": 26,
        "joint_names": [*UR5_JOINT_NAMES, *WUJI_JOINT_NAMES],
        "zero_target_joint_positions_rad": [0.0] * 26,
        "initial_actual_joint_positions_rad": [0.0] * 26,
        "wuji_hand_state_0": hand_state,
        "wuji_hand_state_1": hand_state,
    }
    path = tmp_path / "task.yaml"
    import yaml

    path.write_text(yaml.safe_dump(payload))
    pose = load_task_start(path)
    assert pose.task_id == "block_into_box"
    np.testing.assert_array_equal(pose.ur5_zero, np.zeros(6))
