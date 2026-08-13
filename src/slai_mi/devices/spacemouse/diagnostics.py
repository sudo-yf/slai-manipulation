"""SpaceMouse service and live-input diagnostics."""

from __future__ import annotations

import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

SPACEMOUSE_BY_ID = Path("/dev/input/by-id")
RESTART_COMMAND = "sudo systemctl restart spacenavd"


@dataclass(frozen=True)
class ServiceBinding:
    """Boot-relative timestamps for the daemon and current input device."""

    event_device: Path
    service_started_us: int
    device_initialized_us: int

    @property
    def device_is_newer(self) -> bool:
        return self.device_initialized_us > self.service_started_us


def locate_spacemouse_device(search_root: Path = SPACEMOUSE_BY_ID) -> Path:
    candidates = sorted(search_root.glob("*3Dconnexion*event-mouse"))
    if not candidates:
        raise RuntimeError("SpaceMouse input device was not found under /dev/input/by-id")
    if len(candidates) > 1:
        names = ", ".join(str(path) for path in candidates)
        raise RuntimeError(f"multiple SpaceMouse input devices found: {names}")
    return candidates[0]


def _capture(command: list[str], runner: Callable[..., Any]) -> str:
    result = runner(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip() or "no diagnostic output"
        raise RuntimeError(f"command failed ({' '.join(command)}): {detail}")
    return result.stdout.strip()


def _properties(text: str) -> Mapping[str, str]:
    values: dict[str, str] = {}
    for line in text.splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values[key] = value
    return values


def read_service_binding(
    event_device: Path | None = None,
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> ServiceBinding:
    device = event_device or locate_spacemouse_device()
    active = _capture(["systemctl", "is-active", "spacenavd"], runner)
    if active != "active":
        raise RuntimeError(f"spacenavd is not active (state={active!r})")

    service_started = _capture(
        ["systemctl", "show", "spacenavd", "-p", "ActiveEnterTimestampMonotonic", "--value"],
        runner,
    )
    properties = _properties(
        _capture(["udevadm", "info", "--query=property", f"--name={device}"], runner)
    )
    initialized = properties.get("USEC_INITIALIZED", "")
    try:
        return ServiceBinding(device, int(service_started), int(initialized))
    except ValueError as exc:
        raise RuntimeError("could not read SpaceMouse/spacenavd monotonic timestamps") from exc


def validate_service_binding(
    event_device: Path | None = None,
    *,
    runner: Callable[..., Any] = subprocess.run,
) -> str:
    binding = read_service_binding(event_device, runner=runner)
    if binding.device_is_newer:
        raise RuntimeError(
            "SpaceMouse was connected after spacenavd started, so the daemon has no live "
            f"device binding. Run `{RESTART_COMMAND}`, then retry."
        )
    return f"SpaceMouse service check passed: device={binding.event_device.name}"


def wait_for_live_input(
    spacemouse: Any,
    required_released_buttons: tuple[int, ...],
    *,
    timeout_s: float = 12.0,
    hold_s: float = 0.3,
    monotonic: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    emit: Callable[[str], None] = print,
    on_state: Callable[[np.ndarray, Mapping[int, bool]], None] | None = None,
) -> None:
    """Require one deliberate cap motion followed by a centered hold."""
    deadline = monotonic() + timeout_s
    activity_seen = False
    centered_since: float | None = None
    emit("INPUT CHECK: move the SpaceMouse cap once, then release it to center.")

    while monotonic() < deadline:
        motion, buttons = spacemouse.state()
        if on_state is not None:
            on_state(motion, buttons)
        moving = bool(np.any(np.abs(motion) > 0.0))
        buttons_released = all(
            not buttons.get(button, False) for button in required_released_buttons
        )
        if not activity_seen:
            if moving:
                activity_seen = True
                emit("SpaceMouse motion detected; release the cap and Shift/Ctrl.")
        elif not moving and buttons_released:
            if centered_since is None:
                centered_since = monotonic()
            elif monotonic() - centered_since >= hold_s:
                emit("SpaceMouse live-input check passed.")
                return
        else:
            centered_since = None
        sleep(0.01)

    if not activity_seen:
        raise RuntimeError(
            "no SpaceMouse motion events were received; move the cap or run "
            f"`{RESTART_COMMAND}` after reconnecting the device"
        )
    raise RuntimeError("SpaceMouse did not return to center with Shift/Ctrl released")
