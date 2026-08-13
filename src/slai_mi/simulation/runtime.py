"""Backend-neutral loops shared by simulation applications and tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class SimulationCommand:
    """One simulator command in the shared Z-up tool frame."""

    twist: np.ndarray
    buttons: Mapping[int, bool]


class SimulationScene(Protocol):
    """Task plugin boundary; task assets remain outside the generic runtime."""

    def reset(self, *, seed: int) -> None: ...

    def step(self, command: SimulationCommand) -> Mapping[str, Any]: ...

    def episode_done(self) -> bool: ...

    def episode_success(self) -> bool: ...


class EpisodeWriter(Protocol):
    def begin_episode(self, *, index: int, seed: int) -> None: ...

    def add_frame(self, frame: Mapping[str, Any]) -> None: ...

    def end_episode(self, *, success: bool) -> None: ...


CommandSource = Callable[[], SimulationCommand]


def zero_command() -> SimulationCommand:
    return SimulationCommand(np.zeros(6, dtype=np.float32), {})


def run_teleoperation(
    scene: SimulationScene,
    command_source: CommandSource,
    *,
    should_continue: Callable[[], bool],
) -> int:
    """Step a scene until its host application asks the loop to stop."""
    steps = 0
    scene.reset(seed=0)
    while should_continue():
        scene.step(command_source())
        steps += 1
    return steps


def collect_episodes(
    scene: SimulationScene,
    writer: EpisodeWriter,
    *,
    episodes: int,
    seed: int,
    max_steps: int,
    command_source: CommandSource = zero_command,
) -> list[bool]:
    """Collect deterministic episodes, with all persistence injected by the caller."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if max_steps < 1:
        raise ValueError("max_steps must be at least 1")

    outcomes: list[bool] = []
    for index in range(episodes):
        episode_seed = seed + index
        scene.reset(seed=episode_seed)
        writer.begin_episode(index=index, seed=episode_seed)
        for _ in range(max_steps):
            writer.add_frame(scene.step(command_source()))
            if scene.episode_done():
                break
        success = scene.episode_done() and scene.episode_success()
        writer.end_episode(success=success)
        outcomes.append(success)
    return outcomes
