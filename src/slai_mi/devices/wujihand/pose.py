"""Rigid transforms used by hand tracking and robot control."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from slai_mi.retargeting.geometry import hand_frame_from_keypoints, validate_transform


def transform_points(parent_from_child: np.ndarray, points: np.ndarray) -> np.ndarray:
    transform = validate_transform(parent_from_child)
    values = np.asarray(points, dtype=float)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("points must be a finite (N, 3) array")
    return values @ transform[:3, :3].T + transform[:3, 3]


def ur_pose_to_transform(pose: np.ndarray) -> np.ndarray:
    values = np.asarray(pose, dtype=float).reshape(-1)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("UR pose must be a finite 6-vector")
    transform = np.eye(4)
    transform[:3, :3] = Rotation.from_rotvec(values[3:]).as_matrix()
    transform[:3, 3] = values[:3]
    return validate_transform(transform)


def transform_to_ur_pose(transform: np.ndarray) -> np.ndarray:
    matrix = validate_transform(transform)
    return np.concatenate((matrix[:3, 3], Rotation.from_matrix(matrix[:3, :3]).as_rotvec()))


@dataclass(frozen=True)
class HandUR5Calibration:
    camera_serial: str
    base_from_camera: np.ndarray
    hand_from_tcp: np.ndarray

    def __post_init__(self) -> None:
        if not self.camera_serial.strip():
            raise ValueError("camera serial must not be empty")
        object.__setattr__(self, "base_from_camera", validate_transform(self.base_from_camera))
        object.__setattr__(self, "hand_from_tcp", validate_transform(self.hand_from_tcp))

    def base_coordinates(self, keypoints: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        camera_from_hand = hand_frame_from_keypoints(keypoints)
        base_from_hand = self.base_from_camera @ camera_from_hand
        base_from_tcp = base_from_hand @ self.hand_from_tcp
        return transform_points(self.base_from_camera, keypoints), base_from_hand, base_from_tcp


def load_hand_ur5_calibration(path: Path) -> HandUR5Calibration:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("format_version") != 1:
            raise ValueError("format_version must be 1")

        def matrix(name: str) -> np.ndarray:
            item = payload[name]
            result = np.eye(4)
            result[:3, 3] = item["translation_m"]
            result[:3, :3] = item["rotation_matrix"]
            return validate_transform(result, name)

        return HandUR5Calibration(str(payload["camera_serial"]), matrix("base_from_camera"), matrix("hand_from_tcp"))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid hand-to-UR5 calibration {path}: {exc}") from exc
