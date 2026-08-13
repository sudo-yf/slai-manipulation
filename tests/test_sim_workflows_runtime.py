from __future__ import annotations

from typing import Any

import numpy as np

from slai_mi.simulation.runtime import SimulationCommand, collect_episodes


class FakeScene:
    def __init__(self) -> None:
        self.seeds: list[int] = []
        self.step_count = 0

    def reset(self, *, seed: int) -> None:
        self.seeds.append(seed)
        self.step_count = 0

    def step(self, command: SimulationCommand) -> dict[str, Any]:
        self.step_count += 1
        return {"step": self.step_count, "twist": command.twist.copy()}

    def episode_done(self) -> bool:
        return self.step_count >= 2

    def episode_success(self) -> bool:
        return True


class FakeWriter:
    def __init__(self) -> None:
        self.events: list[tuple[Any, ...]] = []

    def begin_episode(self, *, index: int, seed: int) -> None:
        self.events.append(("begin", index, seed))

    def add_frame(self, frame: dict[str, Any]) -> None:
        self.events.append(("frame", frame["step"]))

    def end_episode(self, *, success: bool) -> None:
        self.events.append(("end", success))


def test_collection_loop_has_deterministic_seeds_and_writer_boundary() -> None:
    scene = FakeScene()
    writer = FakeWriter()
    outcomes = collect_episodes(scene, writer, episodes=2, seed=41, max_steps=5)

    assert outcomes == [True, True]
    assert scene.seeds == [41, 42]
    assert writer.events == [
        ("begin", 0, 41), ("frame", 1), ("frame", 2), ("end", True),
        ("begin", 1, 42), ("frame", 1), ("frame", 2), ("end", True),
    ]
    assert np.asarray(writer.events[1][1]).shape == ()
