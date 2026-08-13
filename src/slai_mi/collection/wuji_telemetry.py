"""Best-effort local telemetry channel from the Wuji safety runtime to recorders."""

from __future__ import annotations

import json
import select
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

from slai_mi.datasets.lerobot_v3.schema import WUJI_DIM

PROTOCOL_VERSION = 1
MAX_PACKET_BYTES = 8192


@dataclass(frozen=True)
class WujiTelemetrySample:
    sequence: int
    actual_q: np.ndarray
    command_q: np.ndarray
    actual_time: float
    command_time: float

    @classmethod
    def from_payload(cls, payload: object) -> WujiTelemetrySample:
        if not isinstance(payload, dict) or payload.get("protocol") != PROTOCOL_VERSION:
            raise ValueError("unsupported Wuji telemetry packet")
        sequence = payload.get("sequence")
        if not isinstance(sequence, int) or sequence < 0:
            raise ValueError("Wuji telemetry sequence must be a non-negative integer")
        actual_time = _timestamp(payload.get("actual_time"), "actual_time")
        command_time = _timestamp(payload.get("command_time"), "command_time")
        return cls(
            sequence=sequence,
            actual_q=_joint_vector(payload.get("actual_q"), "actual_q"),
            command_q=_joint_vector(payload.get("command_q"), "command_q"),
            actual_time=actual_time,
            command_time=command_time,
        )

    def ages_ms(self, now: float | None = None) -> tuple[float, float]:
        current = time.monotonic() if now is None else float(now)
        return (
            max(0.0, (current - self.command_time) * 1000.0),
            max(0.0, (current - self.actual_time) * 1000.0),
        )


class WujiTelemetryPublisher:
    """Send telemetry without ever making hand control depend on the recorder."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._socket = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        self._sequence = 0

    def publish(
        self,
        *,
        actual_q: object,
        command_q: object,
        actual_time: float,
        command_time: float,
    ) -> bool:
        payload = {
            "protocol": PROTOCOL_VERSION,
            "sequence": self._sequence,
            "actual_q": _joint_vector(actual_q, "actual_q").tolist(),
            "command_q": _joint_vector(command_q, "command_q").tolist(),
            "actual_time": _timestamp(actual_time, "actual_time"),
            "command_time": _timestamp(command_time, "command_time"),
        }
        packet = json.dumps(payload, separators=(",", ":"), allow_nan=False).encode("utf-8")
        self._sequence += 1
        try:
            self._socket.sendto(packet, str(self.path))
        except (FileNotFoundError, ConnectionRefusedError, OSError):
            return False
        return True

    def close(self) -> None:
        self._socket.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class WujiTelemetryReceiver:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._socket: socket.socket | None = None
        self._latest: WujiTelemetrySample | None = None

    def start(self) -> None:
        if self._socket is not None:
            raise RuntimeError("Wuji telemetry receiver is already running")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"Wuji telemetry socket already exists: {self.path}")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
            sock.bind(str(self.path))
            sock.setblocking(False)
        except Exception:
            sock.close()
            raise
        self._socket = sock

    def receive_latest(self, timeout_s: float = 0.0) -> WujiTelemetrySample | None:
        self.receive_available(timeout_s)
        return self._latest

    def receive_available(self, timeout_s: float = 0.0) -> tuple[WujiTelemetrySample, ...]:
        """Return every queued packet in sequence order for loss-aware recording."""
        if self._socket is None:
            raise RuntimeError("Wuji telemetry receiver is not running")
        if timeout_s < 0.0:
            raise ValueError("timeout_s must be non-negative")
        readable, _, _ = select.select([self._socket], [], [], timeout_s)
        if not readable:
            return ()
        received = []
        while True:
            try:
                packet = self._socket.recv(MAX_PACKET_BYTES)
            except BlockingIOError:
                break
            sample = WujiTelemetrySample.from_payload(json.loads(packet.decode("utf-8")))
            if self._latest is None or sample.sequence > self._latest.sequence:
                self._latest = sample
                received.append(sample)
        return tuple(received)

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self.path.unlink(missing_ok=True)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _joint_vector(value: object, name: str) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64)
    if vector.shape != (WUJI_DIM,) or not np.isfinite(vector).all():
        raise ValueError(f"{name} must be a finite float[{WUJI_DIM}] vector")
    return vector


def _timestamp(value: object, name: str) -> float:
    result = float(value)
    if not np.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} must be a finite non-negative timestamp")
    return result
