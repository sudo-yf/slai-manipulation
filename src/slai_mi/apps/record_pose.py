"""Teleoperate UR5/WujiHand while recording named groups of measured poses."""

from __future__ import annotations

import argparse
import select
import sys
import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from slai_mi.collection.pose_recorder import HoldGestureDetector, PoseJournal
from slai_mi.datasets.lerobot_v3.schema import UR5_JOINT_NAMES, WUJI_JOINT_NAMES
from slai_mi.devices.spacemouse.buttons import Button
from slai_mi.devices.spacemouse.client import SpaceMouseProcess
from slai_mi.runtime.real_workflows import validate_real_hardware_config
from slai_mi.site_adapter import (
    CachedSpaceMouse,
    StationSession,
    UR5TeleopLoop,
    WujiSupervisionLoop,
)

from ._common import (
    load_yaml,
    print_plan,
    project_path,
    require_real_robot_confirmation,
)

JOINT_NAMES = (*UR5_JOINT_NAMES, *WUJI_JOINT_NAMES)
RECORD_CONTROLS = {
    "capture": int(Button.MENU),
    "finish": int(Button.FIT),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-config", default="configs/hardware.yaml")
    parser.add_argument("--task", default="configs/tasks/block_into_box.yaml")
    parser.add_argument("--output-root", default="data/pose-recordings")
    parser.add_argument("--hold-seconds", type=float, default=0.8)
    parser.add_argument("--execute-real", action="store_true", help="Enable physical teleoperation")
    parser.add_argument("--confirm", help="Required physical-motion confirmation phrase")
    return parser


def _wait_for_release(mouse: SpaceMouseProcess, codes: tuple[int, ...]) -> None:
    while True:
        _motion, buttons = mouse.state()
        if not any(buttons.get(code, False) for code in codes):
            return
        time.sleep(0.02)


def _read_positions(session: StationSession) -> np.ndarray:
    session.check()
    arm = np.asarray(session.read_ur5_state()["joints"], dtype=float)
    hand = np.asarray(session.read_wuji_positions(), dtype=float)
    positions = np.concatenate((arm, hand))
    if positions.shape != (len(JOINT_NAMES),) or not np.isfinite(positions).all():
        raise RuntimeError(f"station returned an invalid {positions.size}-DoF pose")
    return positions


def _run_recorder(
    args: argparse.Namespace, hardware: dict[str, Any], task: dict[str, Any]
) -> Path:
    if not sys.stdin.isatty():
        raise RuntimeError("pose recording requires an interactive terminal")
    validate_real_hardware_config(
        hardware, required=("ur5", "wujihand", "spacemouse")
    )
    for section, key in (
        ("ur5", "driver_python"),
        ("wujihand", "driver_python"),
        ("wujihand", "retarget_python"),
    ):
        executable = Path(str(hardware[section].get(key, "")))
        if not executable.is_file():
            raise FileNotFoundError(f"{section}.{key} is missing: {executable}")
    output_root = project_path(args.output_root)
    journal: PoseJournal | None = None
    controls = RECORD_CONTROLS
    detector = HoldGestureDetector(controls, hold_seconds=args.hold_seconds)
    codes = tuple(controls.values())
    print(f"保存目录: {output_root}")
    print("在终端输入分组名称并回车；之后长按 Menu 记录，长按 Fit 保存退出。")
    print("再次输入名称会立即新增或切换分组。")
    session = StationSession(hardware, task)
    mouse = CachedSpaceMouse(
        SpaceMouseProcess(
            deadzone=session.spacemouse_deadzone,
            stale_timeout=0.1,
            rate_hz=1.0 / session.control_period_s,
        ),
        session.control_period_s,
    )
    stop_event = threading.Event()
    failures: list[BaseException] = []
    ur5_loop = UR5TeleopLoop(session, mouse)
    wuji_loop = WujiSupervisionLoop(session, hardware, mouse)
    loops = (
        ("record-ur5-teleop", ur5_loop),
        ("record-wuji-retarget", wuji_loop),
    )

    def guarded(loop: Any) -> None:
        try:
            loop.run(stop_event)
        except BaseException as exc:  # noqa: BLE001 - hardware failure crosses thread boundary
            failures.append(exc)
            stop_event.set()

    threads = [
        threading.Thread(target=guarded, args=(loop,), name=name) for name, loop in loops
    ]
    started_threads: list[threading.Thread] = []
    alive: list[str] = []
    try:
        with mouse:
            try:
                for thread in threads:
                    thread.start()
                    started_threads.append(thread)
                deadline = time.monotonic() + 10.0
                while (
                    not session.supervisor.armed or not wuji_loop.ready.is_set()
                ) and not failures:
                    if time.monotonic() >= deadline:
                        raise RuntimeError(
                            "SpaceMouse/Wuji retargeting did not become ready within 10 seconds"
                        )
                    time.sleep(0.02)
                if failures:
                    raise RuntimeError(f"teleoperation failed: {failures[0]}") from failures[0]
                print(
                    "SpaceMouse UR5 遥操作已启动；"
                    f"Wuji retargeting 已启动（4K USB {wuji_loop.camera_serial}）。"
                )
                _wait_for_release(mouse, codes)
                while not stop_event.is_set():
                    readable, _writable, _exceptional = select.select(
                        [sys.stdin], [], [], 0.0
                    )
                    if readable:
                        line = sys.stdin.readline()
                        if line == "":
                            break
                        group_name = line.strip()
                        if group_name:
                            if journal is None:
                                journal = PoseJournal.create(
                                    output_root,
                                    recording_name=group_name,
                                    joint_names=JOINT_NAMES,
                                    exact_filename=True,
                                )
                                print(f"保存位置: {journal.path}")
                            created = journal.select_group(group_name)
                            journal.save()
                            print(f"{'新建' if created else '切换到'}分组: {group_name}")
                    _motion, buttons = mouse.state()
                    events = detector.update(buttons, time.monotonic())
                    if "finish" in events:
                        break
                    if "capture" in events:
                        if journal is None or journal.payload["active_group"] is None:
                            print("未记录：请先在终端输入分组名称并回车。")
                            detector.reset()
                            _wait_for_release(mouse, codes)
                            continue
                        name = journal.next_pose_name()
                        journal.record(name, _read_positions(session))
                        journal.save()
                        print(f"已记录: {journal.active_group['name']} / {name}")
                        detector.reset()
                        _wait_for_release(mouse, codes)
                    time.sleep(0.02)
                if failures:
                    raise RuntimeError(f"teleoperation failed: {failures[0]}") from failures[0]
            finally:
                stop_event.set()
                for thread in started_threads:
                    thread.join(timeout=5.0)
                alive = [thread.name for thread in started_threads if thread.is_alive()]
    except KeyboardInterrupt:
        print("\n收到 Ctrl-C，保存已有记录。")
    finally:
        if journal is not None:
            journal.save()
    if alive:
        raise RuntimeError(f"teleoperation threads did not stop: {', '.join(alive)}")
    if journal is None:
        raise RuntimeError("没有输入分组名称，因此没有创建记录文件")
    print(f"记录完成: {journal.path}")
    return journal.path


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.hold_seconds <= 0.0:
        raise SystemExit("--hold-seconds must be positive")
    hardware = load_yaml(args.hardware_config)
    task = load_yaml(args.task)
    require_real_robot_confirmation(args.execute_real, args.confirm)
    print_plan(
        {
            "app": "record_pose",
            "mode": "execute" if args.execute_real else "dry-run",
            "hardware_config": str(project_path(args.hardware_config)),
            "task": task.get("task", {}).get("id"),
            "output_root": str(project_path(args.output_root)),
            "measured_schema": "real_v1",
            "measured_dimension": len(JOINT_NAMES),
            "simulation_dimension": 28,
            "hardware_motion": "enabled" if args.execute_real else "disabled",
            "wujihand_control": "camera_retargeting",
        }
    )
    if not args.execute_real:
        return 0
    try:
        _run_recorder(args, hardware, task)
    except (OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"Pose recording failed: {exc}") from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
