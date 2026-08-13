"""Hardware-neutral camera data contracts."""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass


@dataclass(frozen=True)
class CameraConfig:
    name: str
    serial: str
    width: int
    height: int
    fps: int
    enable_depth: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.serial.strip():
            raise ValueError("camera name and serial are required")
        if min(self.width, self.height, self.fps) <= 0:
            raise ValueError("camera dimensions and FPS must be positive")


@dataclass(frozen=True)
class CameraFrame:
    camera: str
    sequence: int
    device_timestamp_s: float
    host_timestamp_s: float
    color: object
    depth: object | None = None

    def __post_init__(self) -> None:
        if not self.camera or self.sequence < 0:
            raise ValueError("invalid camera frame identity")
        if not all(math.isfinite(v) for v in (self.device_timestamp_s, self.host_timestamp_s)):
            raise ValueError("camera timestamps must be finite")


def validate_camera_set(configs: Iterable[CameraConfig], expected_count: int | None = 3) -> tuple[CameraConfig, ...]:
    result = tuple(configs)
    if expected_count is not None and len(result) != expected_count:
        raise ValueError(f"expected {expected_count} cameras, got {len(result)}")
    if len({item.name for item in result}) != len(result):
        raise ValueError("camera names must be unique")
    if len({item.serial for item in result}) != len(result):
        raise ValueError("camera serials must be unique")
    return result
