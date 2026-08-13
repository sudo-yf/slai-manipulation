"""Framework-independent action prediction metrics."""

from __future__ import annotations

import numpy as np


def select_horizon_anchors(
    episode_indices: object,
    frame_indices: object,
    *,
    count: int,
    horizon_span: int,
) -> np.ndarray:
    if count <= 0 or horizon_span < 0:
        raise ValueError("count must be positive and horizon_span non-negative")
    episodes = np.asarray(episode_indices)
    frames = np.asarray(frame_indices)
    if episodes.shape != frames.shape or episodes.ndim != 1:
        raise ValueError("episode and frame indices must be matching vectors")
    candidates: list[int] = []
    for episode in np.unique(episodes):
        rows = np.flatnonzero(episodes == episode)
        candidates.extend(rows[frames[rows] + horizon_span <= frames[rows[-1]]].tolist())
    if not candidates:
        raise ValueError("dataset has no complete action horizons")
    positions = np.linspace(0, len(candidates) - 1, min(count, len(candidates)), dtype=np.int64)
    return np.asarray(candidates, dtype=np.int64)[positions]


def action_metrics(
    predicted: object, target: object, ranges: object, *, arm_dim: int = 6
) -> dict[str, float | bool]:
    predicted_array = np.asarray(predicted, dtype=np.float64)
    target_array = np.asarray(target, dtype=np.float64)
    scale = np.asarray(ranges, dtype=np.float64)
    if predicted_array.shape != target_array.shape or predicted_array.ndim < 1:
        raise ValueError("predicted and target actions must have matching shapes")
    if scale.shape != (predicted_array.shape[-1],) or not np.isfinite(scale).all():
        raise ValueError("ranges must be a finite vector matching action dimension")
    if not 0 < arm_dim < predicted_array.shape[-1]:
        raise ValueError("arm_dim must split the action vector into non-empty groups")
    error = np.abs(predicted_array - target_array)
    normalized = error / np.maximum(scale, 1e-6)
    return {
        "mae": float(error.mean()),
        "normalized_mae": float(normalized.mean()),
        "arm_mae": float(error[..., :arm_dim].mean()),
        "hand_mae": float(error[..., arm_dim:].mean()),
        "arm_normalized_mae": float(normalized[..., :arm_dim].mean()),
        "hand_normalized_mae": float(normalized[..., arm_dim:].mean()),
        "finite": bool(np.isfinite(predicted_array).all()),
    }
