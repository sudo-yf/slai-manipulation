"""Validation and temporal gating for metric hand landmarks."""

from __future__ import annotations

import numpy as np

HAND_CHAINS = ((0, 1, 2, 3, 4), (0, 5, 6, 7, 8), (0, 9, 10, 11, 12), (0, 13, 14, 15, 16), (0, 17, 18, 19, 20))
HAND_BONES = tuple((chain[i - 1], chain[i]) for chain in HAND_CHAINS for i in range(1, 5))


def reconstruct_missing_keypoints(keypoints: np.ndarray) -> np.ndarray:
    result = np.asarray(keypoints, dtype=float).copy()
    if result.shape != (21, 3):
        raise ValueError("hand keypoints must have shape (21, 3)")
    valid = np.isfinite(result).all(axis=1)
    for chain in HAND_CHAINS:
        for offset, index in enumerate(chain):
            if valid[index] or offset == 0:
                continue
            left = [i for i in range(offset - 1, -1, -1) if valid[chain[i]]]
            right = [i for i in range(offset + 1, len(chain)) if valid[chain[i]]]
            if left and right:
                a, b = left[0], right[0]
                result[index] = result[chain[a]] + (offset - a) / (b - a) * (result[chain[b]] - result[chain[a]])
            elif len(left) >= 2:
                a, b = left[:2]
                result[index] = result[chain[a]] + (result[chain[a]] - result[chain[b]]) / (a - b) * (offset - a)
            valid[index] = np.isfinite(result[index]).all()
    return result.astype(np.float32)


class LandmarkGate:
    def __init__(self, *, min_confidence: float = 0.7, deadband_m: float = 0.0015, max_speed_m_s: float = 4.0, reset_gap_s: float = 0.5) -> None:
        if not 0 <= min_confidence <= 1 or deadband_m < 0 or max_speed_m_s <= 0 or reset_gap_s <= 0:
            raise ValueError("invalid landmark gate parameters")
        self.min_confidence, self.deadband_m = min_confidence, deadband_m
        self.max_speed_m_s, self.reset_gap_s = max_speed_m_s, reset_gap_s
        self.reset()

    def reset(self) -> None:
        self._points: np.ndarray | None = None
        self._time: float | None = None
        self.last_reason = "reset"

    def filter(self, keypoints: np.ndarray, *, timestamp: float, confidence: float) -> np.ndarray | None:
        points = np.asarray(keypoints, dtype=float)
        if points.shape != (21, 3) or not np.isfinite(points).all():
            raise ValueError("landmarks must be finite (21, 3)")
        if confidence < self.min_confidence:
            self.last_reason = "low_confidence"
            return None
        if self._points is None or self._time is None or timestamp - self._time >= self.reset_gap_s:
            return self._accept(points, timestamp, "initial")
        dt = timestamp - self._time
        if dt <= 0:
            raise ValueError("timestamps must increase")
        displacement = np.linalg.norm(points - self._points, axis=1)
        if displacement.max() / dt > self.max_speed_m_s:
            self.last_reason = "motion_outlier"
            return None
        if displacement.max() < self.deadband_m:
            self._time = timestamp
            self.last_reason = "deadband"
            return self._points.astype(np.float32).copy()
        return self._accept(points, timestamp, "accepted")

    def _accept(self, points: np.ndarray, timestamp: float, reason: str) -> np.ndarray:
        self._points, self._time, self.last_reason = points.copy(), timestamp, reason
        return points.astype(np.float32).copy()
