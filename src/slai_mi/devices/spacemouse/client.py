"""Dependency-isolated SpaceMouse process for Isaac simulation."""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from collections import deque
from contextlib import suppress
from pathlib import Path
from typing import Any, Self

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[4]
SOURCE_ROOT = PROJECT_ROOT / "src"
WORKER_PYTHON = Path(
    os.environ.get(
        "SPACEMOUSE_WORKER_PYTHON",
        sys.executable,
    )
)
WORKER_MODULE = "slai_mi.devices.spacemouse.workers.events"


def _worker_environment() -> dict[str, str]:
    environment = os.environ.copy()
    current = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        f"{SOURCE_ROOT}{os.pathsep}{current}" if current else str(SOURCE_ROOT)
    )
    return environment


class SpaceMouseProcess:
    """Expose an external SpaceMouse reader through the regular state interface."""

    def __init__(
        self,
        *,
        deadzone: float = 0.12,
        stale_timeout: float = 0.10,
        rate_hz: float = 125.0,
        backend: str = "spnav",
        event_device: Path | None = None,
    ) -> None:
        self.deadzone = deadzone
        self.stale_timeout = stale_timeout
        self.rate_hz = rate_hz
        self.backend = backend
        self.event_device = Path(event_device) if event_device is not None else None
        self._sock: socket.socket | None = None
        self._process: subprocess.Popen[Any] | None = None
        self._motion = np.zeros(6, dtype=np.float32)
        self._buttons: dict[int, bool] = {}
        self._pending_button_states: deque[dict[int, bool]] = deque()
        self._error: str | None = None

    def start(self) -> None:
        if not WORKER_PYTHON.is_file():
            raise FileNotFoundError(f"SpaceMouse worker Python not found: {WORKER_PYTHON}")
        parent_sock, child_sock = socket.socketpair(socket.AF_UNIX, socket.SOCK_DGRAM)
        parent_sock.setblocking(False)
        command = [
            str(WORKER_PYTHON),
            "-m",
            WORKER_MODULE,
            "--socket-fd",
            str(child_sock.fileno()),
            "--deadzone",
            str(self.deadzone),
            "--stale-timeout",
            str(self.stale_timeout),
            "--rate-hz",
            str(self.rate_hz),
            "--backend",
            self.backend,
        ]
        if self.event_device is not None:
            command.extend(("--event-device", str(self.event_device)))
        try:
            self._process = subprocess.Popen(
                command,
                pass_fds=(child_sock.fileno(),),
                cwd=PROJECT_ROOT,
                env=_worker_environment(),
                start_new_session=True,
            )
        except Exception:
            parent_sock.close()
            child_sock.close()
            raise
        child_sock.close()
        self._sock = parent_sock

    def state(self) -> tuple[np.ndarray, dict[int, bool]]:
        if self._sock is None:
            raise RuntimeError("SpaceMouse worker is not running")
        while True:
            try:
                packet = json.loads(self._sock.recv(65536).decode("utf-8"))
            except BlockingIOError:
                break
            packet_type = packet.get("type")
            if packet_type == "state":
                motion = np.asarray(packet.get("motion"), dtype=np.float32)
                if motion.shape != (6,) or not np.isfinite(motion).all():
                    self._error = "SpaceMouse worker returned invalid motion"
                else:
                    self._motion = motion
                    buttons = {
                        int(button): bool(pressed)
                        for button, pressed in packet.get("buttons", {}).items()
                    }
                    if buttons != self._buttons:
                        self._buttons = buttons
                        self._pending_button_states.append(buttons.copy())
            elif packet_type == "error":
                self._error = str(packet.get("message", "unknown SpaceMouse worker error"))
        if self._process is not None and self._process.poll() is not None and self._error is None:
            self._error = f"SpaceMouse worker exited with code {self._process.returncode}"
        if self._error is not None:
            raise RuntimeError(self._error)
        buttons = (
            self._pending_button_states.popleft()
            if self._pending_button_states
            else self._buttons.copy()
        )
        return self._motion.copy(), buttons

    def stop(self) -> None:
        if self._process is not None and self._process.poll() is None:
            if self._sock is not None:
                with suppress(OSError):
                    self._sock.send(b"stop")
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.terminate()
                self._process.wait(timeout=2.0)
        if self._sock is not None:
            self._sock.close()
        self._process = None
        self._sock = None

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()
