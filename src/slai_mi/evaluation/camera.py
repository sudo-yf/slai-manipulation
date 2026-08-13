"""Landmark-based camera pose fitting with optional OpenCV support."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class CameraFitResult:
    eye: np.ndarray
    rotation_vector: np.ndarray
    translation_vector: np.ndarray
    projected_pixels: np.ndarray
    pixel_errors: np.ndarray
    weighted_rmse_px: float


def camera_matrix(intrinsics: dict[str, float]) -> np.ndarray:
    required = {"fx", "fy", "cx", "cy"}
    if missing := required.difference(intrinsics):
        raise ValueError(f"missing camera intrinsics: {sorted(missing)}")
    values = np.asarray([intrinsics[name] for name in ("fx", "fy", "cx", "cy")], dtype=float)
    if not np.isfinite(values).all() or np.any(values[:2] <= 0):
        raise ValueError("camera intrinsics must be finite and focal lengths positive")
    fx, fy, cx, cy = values
    return np.asarray(((fx, 0, cx), (0, fy, cy), (0, 0, 1)), dtype=np.float64)


def fit_camera_pose(
    intrinsics: dict[str, float],
    world_points: object,
    image_points: object,
    *,
    weights: object | None = None,
) -> CameraFitResult:
    """Fit world-to-camera extrinsics from at least four 3D/2D correspondences."""
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("OpenCV is required for camera pose fitting") from exc
    world = np.asarray(world_points, dtype=np.float64)
    pixels = np.asarray(image_points, dtype=np.float64)
    if world.ndim != 2 or world.shape[1:] != (3,) or world.shape[0] < 4:
        raise ValueError("world_points must have shape (N, 3) with N >= 4")
    if pixels.shape != (world.shape[0], 2):
        raise ValueError("image_points must have shape (N, 2)")
    point_weights = np.ones(world.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    if point_weights.shape != (world.shape[0],) or np.any(point_weights <= 0):
        raise ValueError("weights must be a positive vector matching the points")
    if not all(np.isfinite(value).all() for value in (world, pixels, point_weights)):
        raise ValueError("camera correspondences must be finite")
    success, rotation, translation = cv2.solvePnP(
        world, pixels, camera_matrix(intrinsics), None, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        raise RuntimeError("OpenCV could not fit the camera pose")
    projected = cv2.projectPoints(world, rotation, translation, camera_matrix(intrinsics), None)[0]
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - pixels, axis=1)
    rotation_matrix = cv2.Rodrigues(rotation)[0]
    eye = (-rotation_matrix.T @ translation).reshape(3)
    rmse = float(np.sqrt(np.sum(point_weights * errors**2) / np.sum(point_weights)))
    return CameraFitResult(
        eye=eye,
        rotation_vector=rotation.reshape(3),
        translation_vector=translation.reshape(3),
        projected_pixels=projected,
        pixel_errors=errors,
        weighted_rmse_px=rmse,
    )
