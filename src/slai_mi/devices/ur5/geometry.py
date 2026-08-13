"""Cartesian pose math and startup-relative workspace guards."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


def vector6(values: np.ndarray, name: str) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.shape != (6,) or not np.isfinite(vector).all():
        raise ValueError(f"invalid {name}: {vector}")
    return vector


def tcp_angular_velocity_to_base(twist: np.ndarray, tcp_pose: np.ndarray) -> np.ndarray:
    """Express a TCP-local angular velocity in the robot base frame."""
    transformed = vector6(twist, "twist").copy()
    pose = vector6(tcp_pose, "TCP pose")
    transformed[3:] = Rotation.from_rotvec(pose[3:]).apply(transformed[3:])
    return transformed


def project_pose(pose: np.ndarray, twist: np.ndarray, horizon_s: float) -> np.ndarray:
    """Project a UR rotation-vector pose using a base-frame spatial twist."""
    if horizon_s < 0.0:
        raise ValueError("projection horizon must be non-negative")
    current = vector6(pose, "TCP pose")
    velocity = vector6(twist, "twist")
    projected = current.copy()
    projected[:3] += velocity[:3] * horizon_s
    current_rotation = Rotation.from_rotvec(current[3:])
    delta_rotation = Rotation.from_rotvec(velocity[3:] * horizon_s)
    projected[3:] = (delta_rotation * current_rotation).as_rotvec()
    return projected


def rotation_offset_rad(reference_pose: np.ndarray, pose: np.ndarray) -> float:
    reference = Rotation.from_rotvec(vector6(reference_pose, "reference pose")[3:])
    current = Rotation.from_rotvec(vector6(pose, "TCP pose")[3:])
    return float((current * reference.inv()).magnitude())


def home_twist(
    current_pose: np.ndarray,
    target_pose: np.ndarray,
    max_translation_speed: float,
    max_rotation_speed: float,
    *,
    translation_tolerance: float = 0.0005,
    rotation_tolerance: float = np.deg2rad(0.5),
    proportional_gain: float = 2.0,
) -> tuple[np.ndarray, bool]:
    """Generate a bounded base-frame twist toward a calibrated TCP pose."""
    current = vector6(current_pose, "TCP pose")
    target = vector6(target_pose, "home pose")
    if max_translation_speed <= 0.0 or max_rotation_speed <= 0.0:
        raise ValueError("home speeds must be positive")
    if translation_tolerance <= 0.0 or rotation_tolerance <= 0.0:
        raise ValueError("home tolerances must be positive")

    translation_error = target[:3] - current[:3]
    current_rotation = Rotation.from_rotvec(current[3:])
    target_rotation = Rotation.from_rotvec(target[3:])
    rotation_error = (target_rotation * current_rotation.inv()).as_rotvec()
    reached = bool(
        np.linalg.norm(translation_error) <= translation_tolerance
        and np.linalg.norm(rotation_error) <= rotation_tolerance
    )
    if reached:
        return np.zeros(6, dtype=np.float64), True

    twist = np.concatenate((translation_error, rotation_error)) * proportional_gain
    for values, limit in ((twist[:3], max_translation_speed), (twist[3:], max_rotation_speed)):
        norm = float(np.linalg.norm(values))
        if norm > limit:
            values *= limit / norm
    return twist, False


def joint_home_velocity(
    current_joints: np.ndarray,
    target_joints: np.ndarray,
    max_speed: float,
    *,
    tolerance: float = 0.005,
    proportional_gain: float = 1.5,
) -> tuple[np.ndarray, bool]:
    """Generate synchronized bounded joint velocity toward a recorded configuration."""
    current = vector6(current_joints, "current joints")
    target = vector6(target_joints, "Button 4 joints")
    if max_speed <= 0.0 or tolerance <= 0.0 or proportional_gain <= 0.0:
        raise ValueError("joint home speed, tolerance, and gain must be positive")
    error = target - current
    if float(np.abs(error).max()) <= tolerance:
        return np.zeros(6, dtype=np.float64), True
    velocity = error * proportional_gain
    peak = float(np.abs(velocity).max())
    if peak > max_speed:
        velocity *= max_speed / peak
    return velocity, False


def apply_relative_workspace_guard(
    twist: np.ndarray,
    current_pose: np.ndarray,
    start_pose: np.ndarray,
    max_offset_m: float,
    max_rotation_rad: float,
    horizon_s: float,
) -> tuple[np.ndarray, str | None]:
    """Block command components that move farther outside a relative envelope."""
    if max_offset_m < 0.0 or max_rotation_rad < 0.0:
        raise ValueError("workspace limits must be non-negative")

    guarded = vector6(twist, "twist").copy()
    current = vector6(current_pose, "TCP pose")
    start = vector6(start_pose, "startup pose")
    projected = project_pose(current, guarded, horizon_s)
    blocked: list[str] = []

    current_distance = float(np.linalg.norm(current[:3] - start[:3]))
    projected_distance = float(np.linalg.norm(projected[:3] - start[:3]))
    if (
        max_offset_m > 0.0
        and projected_distance > max_offset_m
        and projected_distance >= current_distance
    ):
        guarded[:3] = 0.0
        blocked.append("translation boundary")

    current_angle = rotation_offset_rad(start, current)
    projected_angle = rotation_offset_rad(start, projected)
    if (
        max_rotation_rad > 0.0
        and projected_angle > max_rotation_rad
        and projected_angle >= current_angle
    ):
        guarded[3:] = 0.0
        blocked.append("rotation boundary")

    return guarded, ", ".join(blocked) if blocked else None
