"""Episode-safe sampling for PI0.5 held-out action-horizon evaluation."""

from __future__ import annotations

import numpy as np


def select_horizon_rows(
    episode_indices: object,
    frame_indices: object,
    *,
    count: int,
    action_horizon: int = 15,
    source_fps: int = 30,
    policy_fps: int = 15,
) -> np.ndarray:
    episodes = np.asarray(episode_indices, dtype=np.int64)
    frames = np.asarray(frame_indices, dtype=np.int64)
    if episodes.ndim != 1 or frames.shape != episodes.shape:
        raise ValueError("episode and frame indices must be matching vectors")
    if count <= 0 or action_horizon <= 0 or source_fps % policy_fps:
        raise ValueError("count/horizon must be positive and FPS ratio integral")
    span = (action_horizon - 1) * (source_fps // policy_fps)
    candidates: list[int] = []
    for episode in np.unique(episodes):
        rows = np.flatnonzero(episodes == episode)
        valid = rows[frames[rows] + span <= frames[rows[-1]]]
        candidates.extend(valid.tolist())
    if not candidates:
        raise ValueError("held-out dataset has no complete action horizons")
    positions = np.linspace(0, len(candidates) - 1, min(count, len(candidates)), dtype=np.int64)
    return np.asarray(candidates, dtype=np.int64)[positions]
