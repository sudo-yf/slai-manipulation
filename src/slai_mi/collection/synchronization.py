"""Timestamp normalization and bounded-buffer synchronization for real collection."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import numpy as np

T = TypeVar("T")


@dataclass(frozen=True)
class TimedSample(Generic[T]):
    value: T
    monotonic_s: float
    device_s: float | None = None


class DeviceClockFit:
    """Fit ``monotonic = slope * device + offset`` over a bounded window."""

    def __init__(self, capacity: int = 128) -> None:
        if capacity < 2:
            raise ValueError("clock fit capacity must be at least two")
        self._pairs: deque[tuple[float, float]] = deque(maxlen=capacity)

    def observe(self, device_s: float, monotonic_s: float) -> float:
        if not np.isfinite((device_s, monotonic_s)).all():
            raise ValueError("timestamps must be finite")
        self._pairs.append((float(device_s), float(monotonic_s)))
        return self.to_monotonic(device_s)

    def to_monotonic(self, device_s: float) -> float:
        if not self._pairs:
            raise RuntimeError("clock fit has no observations")
        if len(self._pairs) == 1:
            device, host = self._pairs[0]
            return host + float(device_s) - device
        values = np.asarray(self._pairs, dtype=np.float64)
        device = values[:, 0]
        host = values[:, 1]
        centered = device - device.mean()
        denominator = float(centered @ centered)
        slope = 1.0 if denominator <= np.finfo(np.float64).eps else float(
            centered @ (host - host.mean()) / denominator
        )
        # A broken or reset device clock must not create an implausible timeline.
        if not 0.9 <= slope <= 1.1:
            slope = 1.0
        offset = float(np.median(host - slope * device))
        return slope * float(device_s) + offset


class BoundedBuffer(Generic[T]):
    """Timestamp-ordered, fixed-capacity buffer with nearest/ZOH/interpolation reads."""

    def __init__(self, capacity: int = 256) -> None:
        if capacity < 2:
            raise ValueError("buffer capacity must be at least two")
        self._items: deque[TimedSample[T]] = deque(maxlen=capacity)

    def append(self, sample: TimedSample[T]) -> None:
        if self._items and sample.monotonic_s < self._items[-1].monotonic_s:
            raise ValueError("samples must be appended in monotonic timestamp order")
        self._items.append(sample)

    def nearest(self, timestamp_s: float) -> TimedSample[T]:
        self._require_items()
        return min(self._items, key=lambda item: abs(item.monotonic_s - timestamp_s))

    def zoh(self, timestamp_s: float) -> TimedSample[T]:
        self._require_items()
        candidates = [item for item in self._items if item.monotonic_s <= timestamp_s]
        if not candidates:
            raise LookupError("no sample exists at or before requested timestamp")
        return candidates[-1]

    def interpolate(self, timestamp_s: float) -> TimedSample[np.ndarray]:
        self._require_items()
        before = [item for item in self._items if item.monotonic_s <= timestamp_s]
        after = [item for item in self._items if item.monotonic_s >= timestamp_s]
        if not before or not after:
            raise LookupError("requested timestamp is not bracketed")
        left, right = before[-1], after[0]
        left_value = np.asarray(left.value, dtype=np.float64)
        right_value = np.asarray(right.value, dtype=np.float64)
        if left_value.shape != right_value.shape:
            raise ValueError("interpolation sample shapes differ")
        span = right.monotonic_s - left.monotonic_s
        alpha = 0.0 if span == 0.0 else (timestamp_s - left.monotonic_s) / span
        value = left_value + alpha * (right_value - left_value)
        return TimedSample(value=value, monotonic_s=timestamp_s)

    def _require_items(self) -> None:
        if not self._items:
            raise LookupError("buffer is empty")


@dataclass(frozen=True)
class SourceDiagnostic:
    skew_ms: float
    age_ms: float
    valid: bool
    reason: str | None = None


@dataclass(frozen=True)
class SynchronizedFrame:
    timestamp_s: float
    cameras: Mapping[str, Any]
    ur5_state: np.ndarray
    wuji_state: np.ndarray
    command: Any
    diagnostics: Mapping[str, SourceDiagnostic]

    @property
    def valid(self) -> bool:
        return all(item.valid for item in self.diagnostics.values())


class RealFrameSynchronizer:
    """Use the primary camera as timeline and align all other real sources."""

    def __init__(
        self,
        *,
        camera_roles: Sequence[str] = ("primary", "secondary", "wrist"),
        capacity: int = 256,
        max_camera_skew_ms: float = 20.0,
        max_camera_age_ms: float = 100.0,
        max_state_age_ms: float = 100.0,
        max_command_age_ms: float = 250.0,
    ) -> None:
        if tuple(camera_roles) != ("primary", "secondary", "wrist"):
            raise ValueError("camera roles must be primary, secondary, and wrist")
        self.cameras = {role: BoundedBuffer[Any](capacity) for role in camera_roles}
        self.ur5 = BoundedBuffer[np.ndarray](capacity)
        self.wuji = BoundedBuffer[np.ndarray](capacity)
        self.commands = BoundedBuffer[Any](capacity)
        self.max_camera_skew_ms = max_camera_skew_ms
        self.max_camera_age_ms = max_camera_age_ms
        self.max_state_age_ms = max_state_age_ms
        self.max_command_age_ms = max_command_age_ms

    def synchronize(self, primary: TimedSample[Any], *, now_s: float) -> SynchronizedFrame:
        target = primary.monotonic_s
        camera_samples = {"primary": primary}
        diagnostics: dict[str, SourceDiagnostic] = {}
        for role in ("secondary", "wrist"):
            camera_samples[role] = self.cameras[role].nearest(target)
        for role, sample in camera_samples.items():
            skew = abs(sample.monotonic_s - target) * 1000.0
            diagnostics[f"camera.{role}"] = self._diagnostic(
                skew,
                max(0.0, now_s - sample.monotonic_s) * 1000.0,
                self.max_camera_skew_ms,
                self.max_camera_age_ms,
                "camera skew or age exceeded",
            )
        ur5 = self.ur5.interpolate(target)
        wuji = self.wuji.interpolate(target)
        for name, buffer in (("ur5", self.ur5), ("wuji", self.wuji)):
            nearest = buffer.nearest(target)
            age = abs(nearest.monotonic_s - target) * 1000.0
            diagnostics[name] = self._diagnostic(
                age, age, self.max_state_age_ms, self.max_state_age_ms, "state too old"
            )
        command = self.commands.zoh(target)
        command_age = (target - command.monotonic_s) * 1000.0
        diagnostics["command"] = self._diagnostic(
            0.0,
            command_age,
            float("inf"),
            self.max_command_age_ms,
            "command too old",
        )
        return SynchronizedFrame(
            timestamp_s=target,
            cameras={role: sample.value for role, sample in camera_samples.items()},
            ur5_state=np.asarray(ur5.value),
            wuji_state=np.asarray(wuji.value),
            command=command.value,
            diagnostics=diagnostics,
        )

    @staticmethod
    def _diagnostic(
        skew: float,
        age: float,
        skew_limit: float,
        age_limit: float,
        reason: str,
    ) -> SourceDiagnostic:
        valid = bool(
            np.isfinite((skew, age)).all()
            and skew <= skew_limit
            and age <= age_limit
        )
        return SourceDiagnostic(skew, age, valid, None if valid else reason)
