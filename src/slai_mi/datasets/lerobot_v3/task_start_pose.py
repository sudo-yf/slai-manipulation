"""Validated task-owned UR5 and Wuji start poses."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

from .schema import UR5_JOINT_NAMES, WUJI_JOINT_NAMES

PROJECT_ROOT = Path(__file__).resolve().parents[4]


@dataclass(frozen=True)
class TaskStartPose:
    """The named 26 DoF zero and Wuji state 1 for one task."""

    path: Path
    task_id: str
    ur5_zero: np.ndarray
    wuji_zero: np.ndarray
    wuji_state_0: np.ndarray
    wuji_state_1: np.ndarray
    wuji_circled_state_0: np.ndarray
    wuji_circled_state_1: np.ndarray


def load_task_start(path: Path) -> TaskStartPose:
    """Load one task's YAML-defined states without hardware side effects."""
    source = Path(path).expanduser().resolve()
    try:
        with source.open(encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"cannot load task start pose {source}: {exc}") from exc
    if isinstance(payload, dict) and isinstance(payload.get("start_pose"), dict):
        task = payload.get("task")
        payload = {**payload["start_pose"], "task": task}
    if not isinstance(payload, dict) or payload.get("format_version") != 1:
        raise RuntimeError(f"unsupported task start pose format in {source}")
    task = payload.get("task")
    if not isinstance(task, dict) or not isinstance(task.get("id"), str) or not task["id"].strip():
        raise RuntimeError(f"task start pose requires task.id: {source}")
    expected_joints = [*UR5_JOINT_NAMES, *WUJI_JOINT_NAMES]
    if payload.get("actual_dimension") != 26 or payload.get("joint_names") != expected_joints:
        raise RuntimeError(f"task start pose must use canonical 26 DoF joint names: {source}")
    zero = _vector(payload.get("zero_target_joint_positions_rad"), 26, "zero target", source)
    _vector(payload.get("initial_actual_joint_positions_rad"), 26, "initial actual state", source)
    state_0 = _wuji_state(payload, "wuji_hand_state_0", source)
    state_1 = _wuji_state(payload, "wuji_hand_state_1", source)
    circled_state_0 = _wuji_state(payload, "wuji_circled_state_0", source, fallback=state_0)
    circled_state_1 = _wuji_state(payload, "wuji_circled_state_1", source, fallback=state_1)
    return TaskStartPose(
        path=source,
        task_id=task["id"].strip(),
        ur5_zero=zero[:6].copy(),
        wuji_zero=zero[6:].copy(),
        wuji_state_0=state_0,
        wuji_state_1=state_1,
        wuji_circled_state_0=circled_state_0,
        wuji_circled_state_1=circled_state_1,
    )


def _wuji_state(
    payload: dict, name: str, source: Path, *, fallback: np.ndarray | None = None
) -> np.ndarray:
    state = payload.get(name)
    if state is None and fallback is not None:
        return fallback.copy()
    if not isinstance(state, dict) or state.get("joint_names") != list(WUJI_JOINT_NAMES):
        raise RuntimeError(f"task start pose must define canonical Wuji {name}: {source}")
    return _vector(state.get("target_joint_positions_rad"), 20, name, source)


def task_start_file_from_task_config(path: Path) -> Path:
    """Resolve ``start_pose_file`` from a task config, relative to the project root."""
    source = Path(path).expanduser().resolve()
    try:
        with source.open(encoding="utf-8") as stream:
            payload = yaml.safe_load(stream)
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"cannot load task config {source}: {exc}") from exc
    if isinstance(payload, dict) and isinstance(payload.get("start_pose"), dict):
        return source
    if not isinstance(payload, dict) or not isinstance(payload.get("start_pose_file"), str):
        raise TypeError(f"task config requires start_pose or start_pose_file: {source}")
    configured = Path(payload["start_pose_file"]).expanduser()
    return (configured if configured.is_absolute() else PROJECT_ROOT / configured).resolve()


def _vector(value: object, size: int, label: str, source: Path) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (size,) or not np.isfinite(vector).all():
        raise RuntimeError(f"task start pose {label} must be finite float[{size}]: {source}")
    return vector
