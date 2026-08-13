from __future__ import annotations

import threading
from contextlib import contextmanager

import pytest

from slai_mi.runtime import (
    CollectionDependencies,
    RealCollectionWorkflow,
    RealTeleopWorkflow,
    TeleopDependencies,
    validate_real_hardware_config,
)


def hardware_config(*, cameras: bool = True) -> dict:
    return {
        "configured": True,
        "ur5": {"enabled": True, "host": "robot.invalid"},
        "wujihand": {"enabled": True, "usb_serial": "fake"},
        "spacemouse": {"enabled": True},
        "cameras": {
            "enabled": cameras,
            "devices": [
                {"role": "primary", "serial": "fake-1"},
                {"role": "secondary", "serial": "fake-2"},
                {"role": "wrist", "serial": "fake-3"},
            ],
        },
    }


def test_config_gate_runs_before_factories() -> None:
    config = hardware_config()
    config["configured"] = False
    with pytest.raises(ValueError, match="configured must be true"):
        validate_real_hardware_config(config, required=("ur5",))


def test_teleop_supervises_workers_and_closes_input() -> None:
    events: list[str] = []

    @contextmanager
    def mouse_factory(_config):
        events.append("mouse-open")
        try:
            yield object()
        finally:
            events.append("mouse-close")

    class Runtime:
        def __init__(self, name: str) -> None:
            self.name = name

        def run(self, stop: threading.Event) -> None:
            events.append(f"{self.name}-run")
            stop.set()

    dependencies = TeleopDependencies(
        ur5_factory=lambda _config, _mouse, _stop: Runtime("ur5"),
        wuji_factory=lambda _config, _mouse, _stop: Runtime("wuji"),
        spacemouse_factory=mouse_factory,
        preflight=lambda _config: events.append("preflight"),
    )
    RealTeleopWorkflow(hardware_config(), dependencies).run()
    assert events[0:2] == ["preflight", "mouse-open"]
    assert "ur5-run" in events
    assert "wuji-run" in events
    assert events[-1] == "mouse-close"


def test_collection_saves_episode_and_finalizes_all_resources() -> None:
    events: list[str] = []

    class Mouse:
        def __init__(self) -> None:
            self.states = [
                {0: True},
                {0: False},
                {1: True},
            ]

        def state(self):
            buttons = self.states.pop(0)
            return (), buttons

    @contextmanager
    def resource(name: str, value=None):
        events.append(f"{name}-open")
        try:
            yield value if value is not None else object()
        finally:
            events.append(f"{name}-close")

    class Dataset:
        def save_episode(self) -> None:
            events.append("save")

        def clear_episode_buffer(self) -> None:
            events.append("clear")

        def finalize(self) -> None:
            events.append("finalize")

    class Recorder:
        def record(self, stop: threading.Event) -> None:
            events.append("record")
            stop.wait()

    mouse = Mouse()
    dependencies = CollectionDependencies(
        ur5_factory=lambda _config: resource("ur5"),
        wuji_factory=lambda _config: resource("wuji"),
        spacemouse_factory=lambda _config: resource("mouse", mouse),
        cameras_factory=lambda _config: resource("cameras"),
        dataset_factory=lambda _dataset, _task: Dataset(),
        synchronizer_factory=lambda _sources, _dataset: object(),
        recorder_factory=lambda _dataset, _sync, _prompt: Recorder(),
        sleep=lambda _seconds: None,
    )
    workflow = RealCollectionWorkflow(
        hardware_config(),
        {"format": "lerobot_v3"},
        {"task": {"instruction": "Do the task."}},
        dependencies,
        episode_limit=1,
    )
    assert workflow.run() == 1
    assert "record" in events
    assert events.count("save") == 1
    assert events[-6:] == [
        "clear",
        "finalize",
        "cameras-close",
        "mouse-close",
        "wuji-close",
        "ur5-close",
    ]
    assert events[-1] == "ur5-close"


def test_collection_rejects_missing_camera_identity_before_opening() -> None:
    config = hardware_config()
    config["cameras"]["devices"][1]["serial"] = None
    opened = False

    def forbidden(_config):
        nonlocal opened
        opened = True
        raise AssertionError("factory must not run")

    dependencies = CollectionDependencies(
        ur5_factory=forbidden,
        wuji_factory=forbidden,
        spacemouse_factory=forbidden,
        cameras_factory=forbidden,
        dataset_factory=lambda _dataset, _task: None,
        synchronizer_factory=lambda _sources, _dataset: None,
        recorder_factory=lambda _dataset, _sync, _prompt: None,
    )
    workflow = RealCollectionWorkflow(
        config,
        {},
        {"task": {"instruction": "task"}},
        dependencies,
        episode_limit=1,
    )
    with pytest.raises(ValueError, match="requires a serial"):
        workflow.run()
    assert not opened
