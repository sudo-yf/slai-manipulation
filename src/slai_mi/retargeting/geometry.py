"""Hand-frame reconstruction and rigid calibration."""

from __future__ import annotations

import numpy as np


def validate_transform(matrix: np.ndarray, label: str = "transform") -> np.ndarray:
    transform = np.asarray(matrix, dtype=float)
    if transform.shape != (4, 4) or not np.isfinite(transform).all():
        raise ValueError(f"{label} must be a finite 4x4 matrix")
    rotation = transform[:3, :3]
    if not np.allclose(transform[3], (0, 0, 0, 1), atol=1e-8):
        raise ValueError(f"{label} has an invalid bottom row")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or np.linalg.det(rotation) < 0:
        raise ValueError(f"{label} rotation is invalid")
    return transform.copy()


def hand_frame_from_keypoints(keypoints_m: np.ndarray) -> np.ndarray:
    points = np.asarray(keypoints_m, dtype=float)
    if points.shape != (21, 3) or not np.isfinite(points).all():
        raise ValueError("keypoints must be a finite (21, 3) array")
    x = points[5] - points[17]
    x /= np.linalg.norm(x)
    y = points[9] - points[0]
    y -= np.dot(y, x) * x
    y /= np.linalg.norm(y)
    z = np.cross(x, y)
    if not np.isfinite((x, y, z)).all():
        raise ValueError("palm landmarks are degenerate")
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((x, np.cross(z, x), z))
    transform[:3, 3] = points[[0, 5, 9, 13, 17]].mean(axis=0)
    return validate_transform(transform)


def estimate_rigid_transform(child_points: np.ndarray, parent_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    child, parent = np.asarray(child_points, dtype=float), np.asarray(parent_points, dtype=float)
    if child.shape != parent.shape or child.ndim != 2 or child.shape[1] != 3 or len(child) < 3:
        raise ValueError("calibration needs matching (N, 3) arrays with N >= 3")
    child_center, parent_center = child.mean(axis=0), parent.mean(axis=0)
    u, _, vt = np.linalg.svd((child - child_center).T @ (parent - parent_center))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ correction @ u.T
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = parent_center - rotation @ child_center
    residuals = np.linalg.norm(child @ rotation.T + transform[:3, 3] - parent, axis=1)
    return validate_transform(transform), residuals
