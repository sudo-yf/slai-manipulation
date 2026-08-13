"""Minimal LeRobot v3 contract for UR5 + Wuji demonstrations."""

from __future__ import annotations

import numpy as np

RECORDED_BUTTON_NAMES = (
    "menu", "fit", "r", "f", "one", "two", "three", "home", "esc", "alt", "shift", "ctrl"
)

FPS = 30
MAX_CAMERA_SKEW_MS = 20.0
IMAGE_HEIGHT = 480
IMAGE_WIDTH = 640
UR5_DIM = 6
WUJI_DIM = 20
STATE_DIM = UR5_DIM + WUJI_DIM
ACTION_DIM = UR5_DIM + WUJI_DIM

D435_PRIMARY_RGB = "observation.images.d435_primary_rgb"
D405_RGB = "observation.images.d405_rgb"
D435_SECONDARY_RGB = "observation.images.d435_secondary_rgb"
OBSERVATION_STATE = "observation.state"
OBSERVATION_TCP_POSE = "observation.tcp_pose"
ACTION = "action"
UR5_TARGET_QD = "telemetry.ur5_target_qd"
ACTUAL_TCP_SPEED = "telemetry.actual_tcp_speed"
CAMERA_SKEW_MS = "telemetry.camera_skew_ms"
SOURCE_AGE_MS = "telemetry.source_age_ms"
DEVICE_TIMESTAMPS_S = "telemetry.device_timestamps_s"
HOST_RECEIVE_TIMESTAMPS_S = "telemetry.host_receive_timestamps_s"
SOURCE_SEQUENCES = "telemetry.source_sequence_numbers"
SOURCE_DROPS = "telemetry.source_dropped_before"
SOURCE_RESTARTS = "telemetry.source_restart_counts"
VALIDITY_MASK = "telemetry.validity_mask"
SPACEMOUSE_AXES = "telemetry.spacemouse_axes"
SPACEMOUSE_BUTTONS = "telemetry.spacemouse_buttons"
CAMERA_SOURCE_NAMES = ("d435_primary_rgb", "d405_rgb", "d435_secondary_rgb")
SOURCE_NAMES = (*CAMERA_SOURCE_NAMES, "ur5", "wuji", "spacemouse")
SPACEMOUSE_BUTTON_NAMES = RECORDED_BUTTON_NAMES

UR5_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
WUJI_JOINT_NAMES = tuple(
    f"right_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5)
)
STATE_NAMES = tuple(f"ur5.{name}.position" for name in UR5_JOINT_NAMES) + tuple(
    f"wuji.{name}.position" for name in WUJI_JOINT_NAMES
)
TCP_POSE_NAMES = ("x", "y", "z", "rx", "ry", "rz")
TCP_TWIST_NAMES = ("vx", "vy", "vz", "wx", "wy", "wz")
ACTION_NAMES = tuple(f"ur5.tcp.{name}" for name in TCP_TWIST_NAMES) + tuple(
    f"wuji.{name}.target_position" for name in WUJI_JOINT_NAMES
)

def lerobot_features() -> dict[str, dict]:
    """Return the exact v3 feature declaration used by the real-data writer."""
    rgb = {
        "dtype": "video",
        "shape": (IMAGE_HEIGHT, IMAGE_WIDTH, 3),
        "names": ["height", "width", "channels"],
        "info": {"is_depth_map": False},
    }
    return {
        D435_PRIMARY_RGB: {**rgb, "info": dict(rgb["info"])},
        D405_RGB: {**rgb, "info": dict(rgb["info"])},
        D435_SECONDARY_RGB: {**rgb, "info": dict(rgb["info"])},
        OBSERVATION_STATE: {
            "dtype": "float32",
            "shape": (STATE_DIM,),
            "names": list(STATE_NAMES),
        },
        OBSERVATION_TCP_POSE: {
            "dtype": "float32",
            "shape": (6,),
            "names": list(TCP_POSE_NAMES),
        },
        ACTION: {
            "dtype": "float32",
            "shape": (ACTION_DIM,),
            "names": list(ACTION_NAMES),
        },
        UR5_TARGET_QD: {
            "dtype": "float32",
            "shape": (6,),
            "names": list(UR5_JOINT_NAMES),
        },
        ACTUAL_TCP_SPEED: {
            "dtype": "float32",
            "shape": (6,),
            "names": list(TCP_TWIST_NAMES),
        },
        CAMERA_SKEW_MS: {
            "dtype": "float32",
            "shape": (2,),
            "names": ["d405_to_primary", "d435_secondary_to_primary"],
        },
        SOURCE_AGE_MS: {"dtype": "float32", "shape": (6,), "names": list(SOURCE_NAMES)},
        DEVICE_TIMESTAMPS_S: {
            "dtype": "float64",
            "shape": (6,),
            "names": list(SOURCE_NAMES),
        },
        HOST_RECEIVE_TIMESTAMPS_S: {
            "dtype": "float64",
            "shape": (6,),
            "names": list(SOURCE_NAMES),
        },
        SOURCE_SEQUENCES: {"dtype": "int64", "shape": (6,), "names": list(SOURCE_NAMES)},
        SOURCE_DROPS: {"dtype": "int64", "shape": (6,), "names": list(SOURCE_NAMES)},
        SOURCE_RESTARTS: {"dtype": "int64", "shape": (6,), "names": list(SOURCE_NAMES)},
        VALIDITY_MASK: {"dtype": "int64", "shape": (6,), "names": list(SOURCE_NAMES)},
        SPACEMOUSE_AXES: {
            "dtype": "float32",
            "shape": (6,),
            "names": ["x", "y", "z", "rx", "ry", "rz"],
        },
        SPACEMOUSE_BUTTONS: {
            "dtype": "int64",
            "shape": (12,),
            "names": list(SPACEMOUSE_BUTTON_NAMES),
        },
    }


def compose_state(ur5_actual_q: object, wuji_actual_q: object) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(ur5_actual_q, dtype=np.float32),
            np.asarray(wuji_actual_q, dtype=np.float32),
        ),
        dtype=np.float32,
    )


def compose_action(ur5_target_tcp_speed: object, wuji_target_q: object) -> np.ndarray:
    return np.concatenate(
        (
            np.asarray(ur5_target_tcp_speed, dtype=np.float32),
            np.asarray(wuji_target_q, dtype=np.float32),
        ),
        dtype=np.float32,
    )
