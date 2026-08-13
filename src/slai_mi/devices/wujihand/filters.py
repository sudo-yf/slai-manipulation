"""Stateful filters for dexterous-hand joint targets."""

from __future__ import annotations

import math
from collections.abc import Sequence


class OneEuroFilter:
    """Dependency-free vector One Euro filter with monotonic timestamps."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5, derivative_cutoff: float = 1.0):
        if min_cutoff <= 0 or beta < 0 or derivative_cutoff <= 0:
            raise ValueError("invalid One Euro filter parameters")
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.derivative_cutoff = float(derivative_cutoff)
        self.reset()

    def reset(self) -> None:
        self._timestamp: float | None = None
        self._raw: list[float] | None = None
        self._value: list[float] | None = None
        self._derivative: list[float] | None = None

    @staticmethod
    def _alpha(cutoff: float, elapsed: float) -> float:
        return elapsed / (elapsed + 1.0 / (2.0 * math.pi * cutoff))

    def filter(self, value: Sequence[float], timestamp: float) -> tuple[float, ...]:
        sample = [float(item) for item in value]
        if not sample or not all(math.isfinite(item) for item in sample):
            raise ValueError("filter input must be a non-empty finite vector")
        timestamp = float(timestamp)
        if not math.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if self._timestamp is None:
            self._timestamp = timestamp
            self._raw = sample.copy()
            self._value = sample.copy()
            self._derivative = [0.0] * len(sample)
            return tuple(sample)
        assert self._raw is not None and self._value is not None and self._derivative is not None
        if len(sample) != len(self._raw):
            raise ValueError("filter input dimension changed")
        elapsed = timestamp - self._timestamp
        if elapsed <= 0:
            raise ValueError("timestamps must be strictly increasing")
        derivative_alpha = self._alpha(self.derivative_cutoff, elapsed)
        for index, current in enumerate(sample):
            derivative = (current - self._raw[index]) / elapsed
            self._derivative[index] += derivative_alpha * (derivative - self._derivative[index])
            cutoff = self.min_cutoff + self.beta * abs(self._derivative[index])
            alpha = self._alpha(cutoff, elapsed)
            self._value[index] += alpha * (current - self._value[index])
        self._timestamp = timestamp
        self._raw = sample.copy()
        return tuple(self._value)
