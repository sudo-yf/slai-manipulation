"""Joint-limit and slew-rate guards applied before WujiHand I/O."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class JointLimits:
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.lower or len(self.lower) != len(self.upper):
            raise ValueError("joint limits must have matching non-empty dimensions")
        if any(not math.isfinite(v) for v in self.lower + self.upper):
            raise ValueError("joint limits must be finite")
        if any(low >= high for low, high in zip(self.lower, self.upper, strict=True)):
            raise ValueError("each lower limit must be below its upper limit")

    @property
    def dimension(self) -> int:
        return len(self.lower)


class SafeCommandLimiter:
    """Clamp absolute limits and per-second motion between accepted commands."""

    def __init__(self, limits: JointLimits, max_velocity: Sequence[float]):
        velocity = tuple(float(item) for item in max_velocity)
        if len(velocity) != limits.dimension or any(v <= 0 or not math.isfinite(v) for v in velocity):
            raise ValueError("max velocity must be positive and match joint dimension")
        self.limits = limits
        self.max_velocity = velocity
        self._last: tuple[float, ...] | None = None
        self._timestamp: float | None = None

    def reset(self, position: Sequence[float], timestamp: float) -> tuple[float, ...]:
        self._last = self._clamp(position)
        self._timestamp = float(timestamp)
        return self._last

    def _clamp(self, command: Sequence[float]) -> tuple[float, ...]:
        values = tuple(float(item) for item in command)
        if len(values) != self.limits.dimension or any(not math.isfinite(v) for v in values):
            raise ValueError("command must be finite and match joint dimension")
        return tuple(max(low, min(high, value)) for value, low, high in zip(values, self.limits.lower, self.limits.upper, strict=True))

    def limit(self, command: Sequence[float], timestamp: float) -> tuple[float, ...]:
        target = self._clamp(command)
        timestamp = float(timestamp)
        if self._last is None or self._timestamp is None:
            return self.reset(target, timestamp)
        elapsed = timestamp - self._timestamp
        if elapsed <= 0:
            raise ValueError("timestamps must be strictly increasing")
        output = tuple(previous + max(-speed * elapsed, min(speed * elapsed, desired - previous)) for desired, previous, speed in zip(target, self._last, self.max_velocity, strict=True))
        self._last, self._timestamp = output, timestamp
        return output
