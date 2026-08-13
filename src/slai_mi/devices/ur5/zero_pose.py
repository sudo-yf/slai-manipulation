"""Persistent, validated UR5 TCP zero-pose calibration."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

from .geometry import vector6


def load_zero_pose(path: Path, robot_host: str) -> np.ndarray | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise TypeError("root must be an object")
        if payload.get("format_version") != 1:
            raise ValueError("unsupported format version")
        if payload.get("robot_host") != robot_host:
            raise ValueError(
                f"robot host is {payload.get('robot_host')!r}, expected {robot_host!r}"
            )
        return vector6(payload["tcp_pose"], "recorded zero pose")
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid UR5 zero pose file {path}: {exc}") from exc


def save_zero_pose(
    path: Path,
    robot_host: str,
    tcp_pose: np.ndarray,
    *,
    recorded_at: datetime | None = None,
) -> None:
    pose = vector6(tcp_pose, "TCP zero pose")
    timestamp = recorded_at or datetime.now().astimezone()
    payload = {
        "format_version": 1,
        "robot_host": robot_host,
        "tcp_pose": [float(value) for value in pose],
        "recorded_at": timestamp.isoformat(timespec="seconds"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
