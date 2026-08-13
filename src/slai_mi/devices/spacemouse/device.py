"""Threaded SpaceMouse reader with normalization, deadzone, and stale-input handling."""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Sequence
from pathlib import Path
from typing import Self

import numpy as np

from . import spnav
from .evdev import EvdevReader

SPNAV_TO_Z_UP = np.array(
    [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    dtype=np.float32,
)


def deadzone_vector(deadzone: float | Sequence[float]) -> np.ndarray:
    values = np.asarray(deadzone, dtype=np.float32)
    if values.ndim == 0:
        values = np.full(6, float(values), dtype=np.float32)
    values = values.reshape(-1)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("deadzone must be one finite value or six finite values")
    if np.any(values < 0.0) or np.any(values >= 1.0):
        raise ValueError("deadzone values must be in [0, 1)")
    return values


def normalize_motion(
    raw_motion: Sequence[float],
    max_value: float,
    deadzone: np.ndarray,
) -> np.ndarray:
    """Normalize raw spnav values and transform them into the shared Z-up frame."""
    if not np.isfinite(max_value) or max_value <= 0.0:
        raise ValueError("max_value must be finite and positive")
    raw = np.asarray(raw_motion, dtype=np.float32).reshape(-1)
    if raw.shape != (6,) or not np.isfinite(raw).all():
        raise ValueError(f"invalid raw SpaceMouse motion: {raw}")

    normalized = np.clip(raw / max_value, -1.0, 1.0)
    normalized[np.abs(normalized) < deadzone] = 0.0
    transformed = np.empty(6, dtype=np.float32)
    transformed[:3] = SPNAV_TO_Z_UP @ normalized[:3]
    transformed[3:] = SPNAV_TO_Z_UP @ normalized[3:]
    return transformed


class SpaceMouse:
    """Read the latest normalized SpaceMouse state through spacenavd."""

    def __init__(
        self,
        max_value: float = 500.0,
        deadzone: float | Sequence[float] = 0.12,
        stale_timeout: float = 0.25,
        poll_interval: float = 0.001,
        backend: str = "spnav",
        event_device: Path | None = None,
    ) -> None:
        if not np.isfinite(max_value) or max_value <= 0.0:
            raise ValueError("max_value must be finite and positive")
        if not np.isfinite(stale_timeout) or stale_timeout <= 0.0:
            raise ValueError("stale_timeout must be finite and positive")
        if not np.isfinite(poll_interval) or poll_interval <= 0.0:
            raise ValueError("poll_interval must be finite and positive")
        if backend not in {"spnav", "evdev"}:
            raise ValueError("backend must be 'spnav' or 'evdev'")
        if backend == "evdev" and event_device is None:
            raise ValueError("event_device is required for the evdev backend")
        self.max_value = float(max_value)
        self.deadzone = deadzone_vector(deadzone)
        self.stale_timeout = float(stale_timeout)
        self.poll_interval = float(poll_interval)
        self.backend = backend
        self.event_device = Path(event_device) if event_device is not None else None
        self._raw_motion = np.zeros(6, dtype=np.float32)
        self._buttons: dict[int, bool] = {}
        self._pending_button_states: deque[tuple[dict[int, bool], float]] = deque()
        self._last_motion_time = 0.0
        self._last_button_time = 0.0
        self._failure: BaseException | None = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._run, name="SpaceMouse", daemon=True)

    def start(self) -> None:
        if self._thread.is_alive():
            raise RuntimeError("SpaceMouse reader is already running")
        self._thread.start()
        if not self._ready.wait(timeout=1.0):
            self.stop()
            raise RuntimeError(f"timed out connecting to SpaceMouse via {self.backend}")
        self._raise_failure()

    def stop(self) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()

    def state(self, now: float | None = None) -> tuple[np.ndarray, dict[int, bool]]:
        motion, buttons, _event_time = self.timestamped_state(now=now)
        return motion, buttons

    def timestamped_state(
        self, now: float | None = None
    ) -> tuple[np.ndarray, dict[int, bool], float]:
        """Return state plus the monotonic arrival time of its newest input event."""
        self._raise_failure()
        with self._lock:
            raw_motion = self._raw_motion.copy()
            if self._pending_button_states:
                buttons, button_time = self._pending_button_states.popleft()
            else:
                buttons, button_time = self._buttons.copy(), self._last_button_time
            last_motion_time = self._last_motion_time
            event_time = max(last_motion_time, button_time)
        timestamp = time.monotonic() if now is None else float(now)
        if timestamp - last_motion_time > self.stale_timeout:
            raw_motion.fill(0.0)
        return normalize_motion(raw_motion, self.max_value, self.deadzone), buttons, event_time

    def _raise_failure(self) -> None:
        with self._lock:
            failure = self._failure
        if failure is not None:
            raise RuntimeError(f"SpaceMouse reader failed: {failure}") from failure

    def _handle_event(
        self,
        event: spnav.SpnavMotionEvent | spnav.SpnavButtonEvent,
        timestamp: float,
    ) -> None:
        if isinstance(event, spnav.SpnavMotionEvent):
            raw = np.asarray(event.translation + event.rotation, dtype=np.float32)
            with self._lock:
                self._raw_motion[:] = raw
                self._last_motion_time = timestamp
            return
        if isinstance(event, spnav.SpnavButtonEvent):
            with self._lock:
                pressed = bool(event.press)
                if pressed != self._buttons.get(event.bnum, False):
                    self._buttons[event.bnum] = pressed
                    self._last_button_time = timestamp
                    self._pending_button_states.append((self._buttons.copy(), timestamp))

    def _run(self) -> None:
        opened = False
        evdev_reader: EvdevReader | None = None
        try:
            if self.backend == "evdev":
                assert self.event_device is not None
                evdev_reader = EvdevReader(self.event_device)
                evdev_reader.open()
                poll_event = evdev_reader.poll_event
            else:
                spnav.open_connection()
                poll_event = spnav.poll_event
            opened = True
            self._ready.set()
            while not self._stop.is_set():
                event = poll_event()
                if event is None:
                    time.sleep(self.poll_interval)
                    continue
                self._handle_event(event, time.monotonic())
        except Exception as exc:  # noqa: BLE001 - propagate reader-thread failures to caller
            with self._lock:
                self._failure = exc
            self._ready.set()
        finally:
            if opened:
                if evdev_reader is not None:
                    evdev_reader.close()
                else:
                    spnav.close_connection()
