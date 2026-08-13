"""Portable persistence for measured WujiHand poses."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


def save_grasp_pose(path: Path, joints_rad: np.ndarray, *, device_serial: str) -> None:
    joints = np.asarray(joints_rad, dtype=float).reshape(-1)
    if joints.shape != (20,) or not np.isfinite(joints).all() or not device_serial:
        raise ValueError("pose requires 20 finite joints and a device serial")
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps({"schema_version": 1, "device_serial": device_serial, "joint_positions_rad": joints.tolist()}, indent=2) + "\n", encoding="utf-8")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def load_grasp_pose(path: Path, *, device_serial: str) -> np.ndarray:
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1 or payload.get("device_serial") != device_serial:
            raise ValueError("schema or device serial mismatch")
        joints = np.asarray(payload["joint_positions_rad"], dtype=float)
        if joints.shape != (20,) or not np.isfinite(joints).all():
            raise ValueError("invalid joint vector")
        return joints.copy()
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid WujiHand grasp pose {path}: {exc}") from exc
