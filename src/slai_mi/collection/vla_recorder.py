"""Episode recording loop for canonical three-camera LeRobot v3 data."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from slai_mi.datasets.lerobot_v3.contract import validate_frame
from slai_mi.datasets.lerobot_v3.schema import (
    ACTION,
    ACTUAL_TCP_SPEED,
    CAMERA_SKEW_MS,
    D405_RGB,
    D435_PRIMARY_RGB,
    D435_SECONDARY_RGB,
    DEVICE_TIMESTAMPS_S,
    HOST_RECEIVE_TIMESTAMPS_S,
    OBSERVATION_STATE,
    OBSERVATION_TCP_POSE,
    SOURCE_AGE_MS,
    SOURCE_DROPS,
    SOURCE_RESTARTS,
    SOURCE_SEQUENCES,
    SPACEMOUSE_AXES,
    SPACEMOUSE_BUTTONS,
    UR5_TARGET_QD,
    VALIDITY_MASK,
    compose_action,
    compose_state,
)


@dataclass(frozen=True)
class SourceSample:
    """Transport-neutral sample metadata used to align device streams."""

    value: Any
    device_timestamp_s: float
    host_timestamp_s: float
    sequence: int
    dropped_before: int = 0
    restart_count: int = 0
    valid: bool = True


@dataclass(frozen=True)
class SynchronizedInputs:
    primary: SourceSample
    d405: SourceSample
    secondary: SourceSample
    ur5: SourceSample
    wuji: SourceSample
    spacemouse: SourceSample


def assemble_frame(inputs: SynchronizedInputs, task_prompt: str, *, now: float | None = None) -> dict[str, object]:
    """Build and validate one canonical frame from timestamp-aligned inputs."""
    current = time.monotonic() if now is None else float(now)
    sources = (
        inputs.primary,
        inputs.d405,
        inputs.secondary,
        inputs.ur5,
        inputs.wuji,
        inputs.spacemouse,
    )
    ur5 = inputs.ur5.value
    wuji = inputs.wuji.value
    mouse = inputs.spacemouse.value
    frame: dict[str, object] = {
        D435_PRIMARY_RGB: np.asarray(inputs.primary.value),
        D405_RGB: np.asarray(inputs.d405.value),
        D435_SECONDARY_RGB: np.asarray(inputs.secondary.value),
        OBSERVATION_STATE: compose_state(ur5.actual_q, wuji.actual_q),
        OBSERVATION_TCP_POSE: np.asarray(ur5.actual_tcp_pose, dtype=np.float32),
        ACTION: compose_action(ur5.target_tcp_speed, wuji.command_q),
        UR5_TARGET_QD: np.asarray(ur5.target_qd, dtype=np.float32),
        ACTUAL_TCP_SPEED: np.asarray(ur5.actual_tcp_speed, dtype=np.float32),
        CAMERA_SKEW_MS: np.asarray(
            [
                abs(inputs.d405.host_timestamp_s - inputs.primary.host_timestamp_s) * 1000.0,
                abs(inputs.secondary.host_timestamp_s - inputs.primary.host_timestamp_s) * 1000.0,
            ],
            dtype=np.float32,
        ),
        SOURCE_AGE_MS: np.asarray(
            [max(0.0, current - source.host_timestamp_s) * 1000.0 for source in sources],
            dtype=np.float32,
        ),
        DEVICE_TIMESTAMPS_S: np.asarray(
            [source.device_timestamp_s for source in sources], dtype=np.float64
        ),
        HOST_RECEIVE_TIMESTAMPS_S: np.asarray(
            [source.host_timestamp_s for source in sources], dtype=np.float64
        ),
        SOURCE_SEQUENCES: np.asarray([source.sequence for source in sources], dtype=np.int64),
        SOURCE_DROPS: np.asarray([source.dropped_before for source in sources], dtype=np.int64),
        SOURCE_RESTARTS: np.asarray([source.restart_count for source in sources], dtype=np.int64),
        VALIDITY_MASK: np.asarray([source.valid for source in sources], dtype=np.int64),
        SPACEMOUSE_AXES: np.asarray(mouse.axes, dtype=np.float32),
        SPACEMOUSE_BUTTONS: np.asarray(mouse.buttons, dtype=np.int64),
        "task": task_prompt,
    }
    validate_frame(frame)
    return frame


@dataclass
class EpisodeRecorder:
    dataset: Any
    synchronizer: Any
    task_prompt: str
    timeout_s: float = 1.0
    monotonic: Any = time.monotonic
    frame_count: int = field(default=0, init=False)

    def record(self, stop_event: threading.Event) -> int:
        """Record until requested to stop; errors propagate to the owning app."""
        self.frame_count = 0
        while not stop_event.is_set():
            inputs = self.synchronizer.read(timeout_s=self.timeout_s)
            frame = assemble_frame(inputs, self.task_prompt, now=self.monotonic())
            self.dataset.add_frame(frame)
            self.frame_count += 1
        return self.frame_count
