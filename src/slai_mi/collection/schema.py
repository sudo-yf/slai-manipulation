"""Stable schema for drawer retrieval demonstrations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

FPS = 15
SIM_HZ = 120
PHYSICS_STEPS_PER_FRAME = SIM_HZ // FPS
IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
TASK_PROMPT = "Grasp the object inside the drawer and take it out."

UR5_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
WRIST_JOINT_NAMES = ("wrist_fe_joint", "wrist_ru_joint")
HAND_JOINT_NAMES = tuple(
    f"right_finger{finger}_joint{joint}" for finger in range(1, 6) for joint in range(1, 5)
)
JOINT_NAMES = UR5_JOINT_NAMES + WRIST_JOINT_NAMES + HAND_JOINT_NAMES
ACTION_DIM = len(JOINT_NAMES)

if ACTION_DIM != 28:
    raise RuntimeError(f"drawer schema must contain 28 joints, got {ACTION_DIM}")


@dataclass(frozen=True)
class Frame:
    """One synchronized 15 Hz sample sent from Isaac to the writer."""

    far_rgb: np.ndarray
    near_rgb: np.ndarray
    joint_position: np.ndarray
    actions: np.ndarray

    def validated(self) -> Frame:
        far = _rgb(self.far_rgb, "far_rgb")
        near = _rgb(self.near_rgb, "near_rgb")
        state = _vector(self.joint_position, "joint_position")
        action = _vector(self.actions, "actions")
        return Frame(far, near, state, action)


def _rgb(value: np.ndarray, name: str) -> np.ndarray:
    image = np.asarray(value)
    expected = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    if image.shape != expected or image.dtype != np.uint8:
        raise ValueError(f"{name} must be uint8{expected}, got {image.dtype}{image.shape}")
    return np.ascontiguousarray(image)


def _vector(value: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (ACTION_DIM,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite float32[{ACTION_DIM}]")
    return np.ascontiguousarray(vector)
