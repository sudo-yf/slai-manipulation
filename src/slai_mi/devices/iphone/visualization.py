"""Dependency-free geometry and repeatability metrics for iPhone poses."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation

from slai_mi.retargeting.geometry import validate_transform


@dataclass(frozen=True)
class StaticPoseMetrics:
    sample_count: int
    position_std_mm: np.ndarray
    position_span_mm: np.ndarray
    rotation_rms_deg: float
    rotation_peak_deg: float


def static_pose_metrics(transforms: list[np.ndarray]) -> StaticPoseMetrics:
    if len(transforms) < 2:
        raise ValueError("at least two pose samples are required")
    poses = np.stack([validate_transform(item) for item in transforms])
    residuals = (
        Rotation.from_matrix(poses[:, :3, :3]).mean().inv() * Rotation.from_matrix(poses[:, :3, :3])
    ).magnitude()
    return StaticPoseMetrics(
        len(poses),
        np.std(poses[:, :3, 3], axis=0) * 1000,
        np.ptp(poses[:, :3, 3], axis=0) * 1000,
        float(np.rad2deg(np.sqrt(np.mean(residuals**2)))),
        float(np.rad2deg(residuals.max())),
    )


def cuboid_vertices(size_m: np.ndarray) -> np.ndarray:
    size = np.asarray(size_m, dtype=float)
    if size.shape != (3,) or np.any(size <= 0):
        raise ValueError("cuboid size must be a positive 3-vector")
    signs = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ]
    )
    return signs * size / 2
