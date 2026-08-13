"""Validation for newline-delimited iPhone ARKit transforms."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class IPhonePose:
    sequence: int
    timestamp_s: float
    sent_at_unix_s: float
    tracking: str
    world_from_camera: tuple[tuple[float, ...], ...]
    teleop_enabled: bool = True
    teleop_epoch: int = 0

    @property
    def position_m(self) -> tuple[float, float, float]:
        return tuple(self.world_from_camera[row][3] for row in range(3))


def _number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return float(value)


def parse_pose_line(line: bytes | str) -> IPhonePose:
    try:
        payload = json.loads(line)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError("invalid pose JSON") from exc
    if not isinstance(payload, dict) or payload.get("format_version") not in {1, 2}:
        raise ValueError("pose packet format_version must be 1 or 2")
    sequence = payload.get("sequence")
    tracking = payload.get("tracking")
    if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence < 0:
        raise ValueError("sequence must be a non-negative integer")
    if not isinstance(tracking, str) or not tracking:
        raise ValueError("tracking must be a non-empty string")
    raw = payload.get("world_from_camera")
    try:
        if isinstance(raw, list) and len(raw) == 16:
            flat = tuple(_number(value, "transform value") for value in raw)
        else:
            flat = tuple(_number(value, "transform value") for row in raw for value in row)
    except (TypeError, ValueError) as exc:
        raise ValueError("world_from_camera must be a 4x4 finite matrix") from exc
    if len(flat) != 16:
        raise ValueError("world_from_camera must be a 4x4 finite matrix")
    matrix = tuple(tuple(flat[row * 4 : row * 4 + 4]) for row in range(4))
    if any(abs(a - b) > 1e-4 for a, b in zip(matrix[3], (0.0, 0.0, 0.0, 1.0), strict=True)):
        raise ValueError("invalid homogeneous bottom row")
    teleop_enabled = payload.get("teleop_enabled", True)
    teleop_epoch = payload.get("teleop_epoch", 0)
    if not isinstance(teleop_enabled, bool):
        raise TypeError("teleop_enabled must be a boolean")
    if isinstance(teleop_epoch, bool) or not isinstance(teleop_epoch, int) or teleop_epoch < 0:
        raise ValueError("teleop_epoch must be a non-negative integer")
    return IPhonePose(
        sequence,
        _number(payload.get("timestamp_s"), "timestamp_s"),
        _number(payload.get("sent_at_unix_s"), "sent_at_unix_s"),
        tracking,
        matrix,
        teleop_enabled,
        teleop_epoch,
    )


def make_iphone_pose(
    *,
    sequence: int,
    timestamp_s: float,
    sent_at_unix_s: float,
    tracking: str,
    world_from_camera: object,
    teleop_enabled: bool = True,
    teleop_epoch: int = 0,
) -> IPhonePose:
    payload = {
        "format_version": 2,
        "sequence": sequence,
        "timestamp_s": timestamp_s,
        "sent_at_unix_s": sent_at_unix_s,
        "tracking": tracking,
        "world_from_camera": world_from_camera,
        "teleop_enabled": teleop_enabled,
        "teleop_epoch": teleop_epoch,
    }
    if hasattr(world_from_camera, "tolist"):
        payload["world_from_camera"] = world_from_camera.tolist()
    return parse_pose_line(json.dumps(payload))
