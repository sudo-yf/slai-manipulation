import json

import pytest

from slai_mi.devices.iphone import parse_pose_line


def test_parse_iphone_pose_packet():
    packet = {
        "format_version": 1,
        "sequence": 4,
        "timestamp_s": 1.2,
        "sent_at_unix_s": 2.3,
        "tracking": "normal",
        "world_from_camera": [[1, 0, 0, 0.1], [0, 1, 0, 0.2], [0, 0, 1, 0.3], [0, 0, 0, 1]],
    }
    pose = parse_pose_line(json.dumps(packet))
    assert pose.position_m == pytest.approx((0.1, 0.2, 0.3))


def test_rejects_bad_homogeneous_transform():
    packet = {
        "format_version": 1,
        "sequence": 0,
        "timestamp_s": 0,
        "sent_at_unix_s": 0,
        "tracking": "normal",
        "world_from_camera": [[1, 0, 0, 0]] * 4,
    }
    with pytest.raises(ValueError, match="bottom row"):
        parse_pose_line(json.dumps(packet))


def test_parse_pose_hub_flat_packet_with_deadman_state():
    packet = {
        "format_version": 2,
        "sequence": 5,
        "timestamp_s": 1.2,
        "sent_at_unix_s": 2.3,
        "tracking": "normal",
        "teleop_enabled": False,
        "teleop_epoch": 7,
        "world_from_camera": [
            1,
            0,
            0,
            0.1,
            0,
            1,
            0,
            0.2,
            0,
            0,
            1,
            0.3,
            0,
            0,
            0,
            1,
        ],
    }
    pose = parse_pose_line(json.dumps(packet))
    assert pose.position_m == pytest.approx((0.1, 0.2, 0.3))
    assert pose.teleop_enabled is False
    assert pose.teleop_epoch == 7
