"""Calibrated confidence-weighted dual-camera hand landmark fusion."""

from __future__ import annotations

import numpy as np

from .geometry import validate_transform


def transform_points(parent_from_child: np.ndarray, points: np.ndarray) -> np.ndarray:
    transform = validate_transform(parent_from_child)
    values = np.asarray(points, dtype=float)
    if values.shape != (21, 3) or not np.isfinite(values).all():
        raise ValueError("landmarks must be a finite (21, 3) array")
    return values @ transform[:3, :3].T + transform[:3, 3]


def fuse_landmarks(primary: np.ndarray, secondary: np.ndarray, primary_weights: np.ndarray, secondary_weights: np.ndarray, primary_from_secondary: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first, second = np.asarray(primary, dtype=float), transform_points(primary_from_secondary, secondary)
    wa, wb = np.asarray(primary_weights, dtype=float), np.asarray(secondary_weights, dtype=float)
    if first.shape != (21, 3) or wa.shape != (21,) or wb.shape != (21,) or np.any(wa < 0) or np.any(wb < 0):
        raise ValueError("invalid landmark fusion inputs")
    total = wa + wb
    safe = np.where(total > 0, total, 1.0)
    fused = (first * wa[:, None] + second * wb[:, None]) / safe[:, None]
    fused[total == 0] = 0
    return fused, np.maximum(wa, wb)
