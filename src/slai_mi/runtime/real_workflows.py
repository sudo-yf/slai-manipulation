"""Dependency-injected orchestration for physical teleoperation and collection."""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass, field
from types import FrameType
from typing import Any, Protocol, Self

from slai_mi.collection.operator_control import EpisodeAction, SpaceMouseEpisodeControls


class StoppableRuntime(Protocol):
    def run(self, stop_event: threading.Event) -> None: ...


def validate_real_hardware_config(
    config: Mapping[str, Any], *, required: Sequence[str]
) -> None:
    """Validate the identity gate before any dependency may open hardware."""
    if config.get("configured") is not True:
        raise ValueError("hardware configuration is not confirmed (configured must be true)")
    for name in required:
        section = config.get(name)
        if not isinstance(section, Mapping) or section.get("enabled") is not True:
            raise ValueError(f"required device is not enabled: {name}")
    ur5 = config.get("ur5", {})
    if "ur5" in required and not str(ur5.get("host") or "").strip():
        raise ValueError("ur5.host must be configured")
    cameras = config.get("cameras", {})
    if "cameras" in required:
        devices = cameras.get("devices")
        if not isinstance(devices, list) or len(devices) != 3:
            raise ValueError("cameras.devices must define exactly three camera roles")
        roles = {item.get("role") for item in devices if isinstance(item, Mapping)}
        if roles != {"primary", "secondary", "wrist"}:
            raise ValueError("camera roles must be primary, secondary, and wrist")
        if any(not str(item.get("serial") or "").strip() for item in devices):
            raise ValueError("every enabled camera requires a serial")


@dataclass(frozen=True)
class TeleopDependencies:
    ur5_factory: Callable[[Mapping[str, Any], Any, threading.Event], StoppableRuntime]
    wuji_factory: Callable[[Mapping[str, Any], Any, threading.Event], StoppableRuntime]
    spacemouse_factory: Callable[[Mapping[str, Any]], Any]
    preflight: Callable[[Mapping[str, Any]], None] = lambda _config: None


@dataclass(frozen=True)
class CollectionDependencies:
    ur5_factory: Callable[[Mapping[str, Any]], Any]
    wuji_factory: Callable[[Mapping[str, Any]], Any]
    spacemouse_factory: Callable[[Mapping[str, Any]], Any]
    cameras_factory: Callable[[Mapping[str, Any]], Any]
    dataset_factory: Callable[[Mapping[str, Any], Mapping[str, Any]], Any]
    synchronizer_factory: Callable[[Mapping[str, Any], Mapping[str, Any]], Any]
    recorder_factory: Callable[[Any, Any, str], Any]
    preflight: Callable[[Mapping[str, Any]], None] = lambda _config: None
    sleep: Callable[[float], None] = time.sleep


@dataclass
class _Worker:
    name: str
    target: Callable[[threading.Event], None]
    stop_event: threading.Event
    failure: list[BaseException]
    thread: threading.Thread = field(init=False)

    def __post_init__(self) -> None:
        def guarded() -> None:
            try:
                self.target(self.stop_event)
            except BaseException as exc:  # noqa: BLE001 - transfer worker failures
                self.failure.append(exc)
                self.stop_event.set()

        self.thread = threading.Thread(target=guarded, name=self.name)


class _SignalStop:
    def __init__(self, stop_event: threading.Event) -> None:
        self.stop_event = stop_event
        self.previous: dict[int, Any] = {}

    def __enter__(self) -> Self:
        if threading.current_thread() is threading.main_thread():
            for number in (signal.SIGINT, signal.SIGTERM):
                self.previous[number] = signal.getsignal(number)
                signal.signal(number, self._stop)
        return self

    def _stop(self, _number: int, _frame: FrameType | None) -> None:
        self.stop_event.set()

    def __exit__(self, *_args: object) -> None:
        for number, handler in self.previous.items():
            signal.signal(number, handler)


class RealTeleopWorkflow:
    """Supervise UR5 and Wuji runtimes with one shared stop condition."""

    def __init__(self, hardware: Mapping[str, Any], dependencies: TeleopDependencies) -> None:
        self.hardware = hardware
        self.dependencies = dependencies

    def run(self) -> None:
        validate_real_hardware_config(
            self.hardware, required=("ur5", "wujihand", "spacemouse")
        )
        self.dependencies.preflight(self.hardware)
        stop_event = threading.Event()
        failures: list[BaseException] = []
        with ExitStack() as resources, _SignalStop(stop_event):
            mouse = resources.enter_context(self.dependencies.spacemouse_factory(self.hardware))
            ur5 = self.dependencies.ur5_factory(self.hardware, mouse, stop_event)
            wuji = self.dependencies.wuji_factory(self.hardware, mouse, stop_event)
            workers = [
                _Worker("ur5-teleop", ur5.run, stop_event, failures),
                _Worker("wuji-teleop", wuji.run, stop_event, failures),
            ]
            for worker in workers:
                worker.thread.start()
            while not stop_event.wait(0.05) and any(w.thread.is_alive() for w in workers):
                pass
            stop_event.set()
            for worker in workers:
                worker.thread.join(timeout=5.0)
            alive = [worker.name for worker in workers if worker.thread.is_alive()]
            if alive:
                raise RuntimeError(f"hardware workers did not stop: {', '.join(alive)}")
            if failures:
                raise RuntimeError(f"hardware worker failed: {failures[0]}") from failures[0]


class RealCollectionWorkflow:
    """Own real sensors, episode recording, dataset commit/discard, and cleanup."""

    def __init__(
        self,
        hardware: Mapping[str, Any],
        dataset_config: Mapping[str, Any],
        task: Mapping[str, Any],
        dependencies: CollectionDependencies,
        *,
        episode_limit: int,
    ) -> None:
        if episode_limit < 1:
            raise ValueError("episode_limit must be at least one")
        self.hardware = hardware
        self.dataset_config = dataset_config
        self.task = task
        self.dependencies = dependencies
        self.episode_limit = episode_limit

    def run(self) -> int:
        validate_real_hardware_config(
            self.hardware, required=("ur5", "wujihand", "spacemouse", "cameras")
        )
        instruction = str(self.task.get("task", {}).get("instruction") or "").strip()
        if not instruction:
            raise ValueError("task.task.instruction must be configured")
        self.dependencies.preflight(self.hardware)
        stop_event = threading.Event()
        with ExitStack() as resources, _SignalStop(stop_event):
            ur5 = resources.enter_context(self.dependencies.ur5_factory(self.hardware))
            wuji = resources.enter_context(self.dependencies.wuji_factory(self.hardware))
            mouse = resources.enter_context(self.dependencies.spacemouse_factory(self.hardware))
            cameras = resources.enter_context(self.dependencies.cameras_factory(self.hardware))
            dataset = self.dependencies.dataset_factory(self.dataset_config, self.task)
            resources.callback(dataset.finalize)
            synchronizer = self.dependencies.synchronizer_factory(
                {"ur5": ur5, "wuji": wuji, "spacemouse": mouse, "cameras": cameras},
                self.dataset_config,
            )
            recorder = self.dependencies.recorder_factory(dataset, synchronizer, instruction)
            controls = SpaceMouseEpisodeControls()
            saved = 0
            active_stop: threading.Event | None = None
            worker: _Worker | None = None
            failures: list[BaseException] = []
            try:
                while not stop_event.is_set() and saved < self.episode_limit:
                    _motion, buttons = mouse.state()
                    action = controls.update(buttons)
                    if action is EpisodeAction.START:
                        active_stop = threading.Event()
                        worker = _Worker("episode-recorder", recorder.record, active_stop, failures)
                        worker.thread.start()
                    elif action in {EpisodeAction.SAVE, EpisodeAction.DISCARD}:
                        if active_stop is None or worker is None:
                            raise RuntimeError("episode control changed without an active recorder")
                        active_stop.set()
                        worker.thread.join(timeout=5.0)
                        if worker.thread.is_alive():
                            raise RuntimeError("episode recorder did not stop")
                        if failures:
                            raise RuntimeError(f"episode recorder failed: {failures[0]}") from failures[0]
                        if action is EpisodeAction.SAVE:
                            dataset.save_episode()
                            saved += 1
                        else:
                            dataset.clear_episode_buffer()
                        active_stop = None
                        worker = None
                    self.dependencies.sleep(0.005)
            finally:
                if active_stop is not None:
                    active_stop.set()
                if worker is not None:
                    worker.thread.join(timeout=5.0)
                dataset.clear_episode_buffer()
            return saved
