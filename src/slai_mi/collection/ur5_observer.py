"""Read-only UR5 RTDE sampling for demonstration recording."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Self

import numpy as np


@dataclass(frozen=True)
class UR5Sample:
    actual_q: np.ndarray
    actual_qd: np.ndarray
    target_q: np.ndarray
    target_qd: np.ndarray
    actual_tcp_pose: np.ndarray
    actual_tcp_speed: np.ndarray
    target_tcp_speed: np.ndarray
    controller_timestamp_s: float
    host_timestamp_s: float


class UR5Observer:
    def __init__(self, host: str, *, receiver_factory: Any | None = None, monotonic=time.monotonic):
        if not host.strip():
            raise ValueError("UR5 host must not be empty")
        self.host = host
        self._receiver_factory = receiver_factory
        self._monotonic = monotonic
        self._receiver: Any | None = None

    def start(self) -> None:
        if self._receiver is not None:
            raise RuntimeError("UR5 observer is already running")
        if self._receiver_factory is None:
            import rtde_receive

            self._receiver_factory = rtde_receive.RTDEReceiveInterface
        self._receiver = self._receiver_factory(self.host)

    def sample(self) -> UR5Sample:
        if self._receiver is None:
            raise RuntimeError("UR5 observer is not running")
        receiver = self._receiver
        sample = UR5Sample(
            actual_q=_six(receiver.getActualQ(), "actual_q"),
            actual_qd=_six(receiver.getActualQd(), "actual_qd"),
            target_q=_six(receiver.getTargetQ(), "target_q"),
            target_qd=_six(receiver.getTargetQd(), "target_qd"),
            actual_tcp_pose=_six(receiver.getActualTCPPose(), "actual_tcp_pose"),
            actual_tcp_speed=_six(receiver.getActualTCPSpeed(), "actual_tcp_speed"),
            target_tcp_speed=_six(receiver.getTargetTCPSpeed(), "target_tcp_speed"),
            controller_timestamp_s=float(receiver.getTimestamp()),
            host_timestamp_s=self._monotonic(),
        )
        if not np.isfinite(sample.controller_timestamp_s):
            raise RuntimeError("UR5 returned a non-finite controller timestamp")
        return sample

    def stop(self) -> None:
        if self._receiver is not None:
            disconnect = getattr(self._receiver, "disconnect", None)
            if callable(disconnect):
                disconnect()
            self._receiver = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()


def _six(value: object, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32)
    if vector.shape != (6,) or not np.isfinite(vector).all():
        raise RuntimeError(f"UR5 {name} must be a finite float[6] vector")
    return vector
