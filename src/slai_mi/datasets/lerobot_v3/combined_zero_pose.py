"""Validated Button 4 targets shared by the UR5 and WujiHand1 runtimes."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .schema import UR5_JOINT_NAMES, WUJI_JOINT_NAMES

BUTTON4_ZERO_POSE_FILE_DEFAULT = Path(__file__).resolve().parents[4] / "configs" / "poses" / "home.json"


def load_ur5_button4_joints(path: Path, robot_host: str) -> np.ndarray | None:
    payload = _load(path)
    if payload is None:
        return None
    section = _section(payload, "ur5", 6)
    if section.get("robot_host") != robot_host:
        raise RuntimeError(
            f"Button 4 UR5 host is {section.get('robot_host')!r}, expected {robot_host!r}"
        )
    if section.get("joint_names") != list(UR5_JOINT_NAMES):
        raise RuntimeError("Button 4 UR5 joint names do not match this controller")
    return _vector(section, 6, "UR5")


def load_wuji_button4_joints(
    path: Path,
    *,
    usb_serial: str,
    product_serial: str,
) -> np.ndarray | None:
    payload = _load(path)
    if payload is None:
        return None
    section = _section(payload, "wuji_hand1", 20)
    if section.get("usb_serial") != usb_serial:
        raise RuntimeError(f"Button 4 Wuji USB serial does not match {usb_serial}")
    if section.get("product_serial") != product_serial:
        raise RuntimeError(f"Button 4 Wuji product serial does not match {product_serial}")
    if section.get("joint_names") != list(WUJI_JOINT_NAMES):
        raise RuntimeError("Button 4 Wuji joint names do not match this controller")
    return _vector(section, 20, "Wuji")


def _load(path: Path) -> dict | None:
    source = Path(path)
    if not source.exists():
        return None
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"cannot load Button 4 zero pose {source}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise RuntimeError(f"unsupported Button 4 zero pose format in {source}")
    if payload.get("actual_dimension") != 26:
        raise RuntimeError(f"Button 4 zero pose must contain 26 DoF: {source}")
    return payload


def _section(payload: dict, name: str, dimension: int) -> dict:
    section = payload.get(name)
    if not isinstance(section, dict) or section.get("dimension") != dimension:
        raise RuntimeError(f"Button 4 {name} section must contain {dimension} DoF")
    return section


def _vector(section: dict, dimension: int, label: str) -> np.ndarray:
    vector = np.asarray(section.get("actual_joint_positions_rad"), dtype=np.float64)
    if vector.shape != (dimension,) or not np.isfinite(vector).all():
        raise RuntimeError(f"Button 4 {label} target must be finite float[{dimension}]")
    return vector.copy()
