from __future__ import annotations

import argparse
import threading

import numpy as np
import pytest
import yaml

from slai_mi.apps import record_pose
from slai_mi.apps.record_pose import JOINT_NAMES, RECORD_CONTROLS, main
from slai_mi.collection.pose_recorder import HoldGestureDetector, PoseJournal, safe_filename
from slai_mi.devices.spacemouse.buttons import Button


def test_hold_detector_fires_once_until_release() -> None:
    detector = HoldGestureDetector({"capture": 1}, hold_seconds=0.8)
    assert detector.update({1: True}, 10.0) == []
    assert detector.update({1: True}, 10.79) == []
    assert detector.update({1: True}, 10.8) == ["capture"]
    assert detector.update({1: True}, 12.0) == []
    assert detector.update({1: False}, 12.1) == []
    assert detector.update({1: True}, 13.0) == []
    assert detector.update({1: True}, 13.8) == ["capture"]


def test_record_buttons_are_menu_to_capture_and_fit_to_finish() -> None:
    assert RECORD_CONTROLS == {
        "capture": int(Button.MENU),
        "finish": int(Button.FIT),
    }


def test_pose_journal_groups_and_atomically_saves(tmp_path) -> None:
    journal = PoseJournal.create(
        tmp_path,
        recording_name="抓取姿态",
        joint_names=JOINT_NAMES,
    )
    assert journal.select_group("箱内") is True
    journal.record("起点", np.arange(26, dtype=float))
    assert journal.select_group("箱外") is True
    journal.record("终点", np.arange(26, dtype=float) + 1.0)
    assert journal.select_group("箱内") is False
    journal.save()

    payload = yaml.safe_load(journal.path.read_text(encoding="utf-8"))
    assert payload["source"]["dimension"] == 26
    assert payload["source"]["missing_from_simulation_v1"] == [
        "wrist_pitch_joint",
        "wrist_yaw_joint",
    ]
    assert [group["name"] for group in payload["groups"]] == ["箱内", "箱外"]
    assert payload["groups"][0]["poses"][0]["name"] == "起点"
    assert payload["groups"][1]["poses"][0]["joint_positions"][-1] == 26.0
    assert not journal.path.with_suffix(".yaml.tmp").exists()


def test_exact_group_filename_reopens_without_overwriting(tmp_path) -> None:
    journal = PoseJournal.create(
        tmp_path,
        recording_name="箱内",
        joint_names=JOINT_NAMES,
        exact_filename=True,
    )
    journal.select_group("箱内")
    journal.record("pose_001", np.zeros(26))
    journal.save()

    reopened = PoseJournal.create(
        tmp_path,
        recording_name="箱内",
        joint_names=JOINT_NAMES,
        exact_filename=True,
    )
    assert reopened.path.name == "箱内.yaml"
    assert reopened.active_group["poses"][0]["name"] == "pose_001"


def test_pose_journal_rejects_wrong_dimension(tmp_path) -> None:
    journal = PoseJournal.create(
        tmp_path,
        recording_name="test",
        joint_names=JOINT_NAMES,
    )
    journal.select_group("default")
    with pytest.raises(ValueError, match="26 finite"):
        journal.record("bad", np.zeros(28))


def test_pose_journal_requires_a_typed_group(tmp_path) -> None:
    journal = PoseJournal.create(
        tmp_path,
        recording_name="test",
        joint_names=JOINT_NAMES,
    )
    with pytest.raises(RuntimeError, match="分组名称"):
        journal.record("pose_001", np.zeros(26))


def test_safe_filename_and_cli_dry_run(capsys) -> None:
    assert safe_filename("  one / two  ") == "one-two"
    assert main([]) == 0
    output = capsys.readouterr().out
    assert '"measured_dimension": 26' in output
    assert '"hardware_motion": "disabled"' in output


def test_record_runtime_teleoperates_and_groups_terminal_input(tmp_path, monkeypatch) -> None:
    lifecycle = []

    class Supervisor:
        armed = False

    class Session:
        control_period_s = 0.001
        spacemouse_deadzone = 0.12

        def __init__(self, _hardware, _task):
            self.supervisor = Supervisor()

        def check(self):
            pass

        def read_ur5_state(self):
            return {"joints": np.arange(6, dtype=float)}

        def read_wuji_positions(self):
            return np.arange(20, dtype=float) + 6.0

    session = Session({}, {})

    class Loop:
        def __init__(self, _session, *_args):
            self.ready = threading.Event()
            self.camera_serial = "HB202400001"

        def run(self, stop_event):
            lifecycle.append("loop_start")
            session.supervisor.armed = True
            self.ready.set()
            stop_event.wait()
            lifecycle.append("loop_stop")

    class Mouse:
        def __enter__(self):
            lifecycle.append("mouse_start")
            return self

        def __exit__(self, *_args):
            lifecycle.append("mouse_stop")

        def state(self):
            return np.zeros(6), {}

    class Detector:
        def __init__(self, *_args, **_kwargs):
            self.calls = 0

        def update(self, _buttons, _now):
            self.calls += 1
            return ["capture"] if self.calls == 1 else ["finish"]

        def reset(self):
            pass

    class Input:
        def isatty(self):
            return True

        def readline(self):
            return "箱内\n"

    readable = iter(([object()], []))
    monkeypatch.setattr(record_pose, "StationSession", lambda *_args: session)
    monkeypatch.setattr(record_pose, "UR5TeleopLoop", Loop)
    monkeypatch.setattr(record_pose, "WujiSupervisionLoop", Loop)
    monkeypatch.setattr(record_pose, "SpaceMouseProcess", lambda **_kwargs: object())
    monkeypatch.setattr(record_pose, "CachedSpaceMouse", lambda *_args: Mouse())
    monkeypatch.setattr(record_pose, "HoldGestureDetector", Detector)
    monkeypatch.setattr(record_pose.sys, "stdin", Input())
    monkeypatch.setattr(
        record_pose.select,
        "select",
        lambda *_args: (next(readable), [], []),
    )
    hardware = {
        "configured": True,
        "ur5": {
            "enabled": True,
            "host": "fake",
            "driver_python": "/bin/true",
        },
        "wujihand": {
            "enabled": True,
            "driver_python": "/bin/true",
            "retarget_python": "/bin/true",
        },
        "spacemouse": {"enabled": True},
    }
    args = argparse.Namespace(
        output_root=str(tmp_path),
        name="runtime",
        hold_seconds=0.8,
    )
    path = record_pose._run_recorder(args, hardware, {})

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert payload["active_group"] == "箱内"
    assert payload["groups"][0]["poses"][0]["joint_positions"] == list(
        np.arange(26, dtype=float)
    )
    assert lifecycle.count("loop_start") == 2
    assert lifecycle.count("loop_stop") == 2
    assert lifecycle[-1] == "mouse_stop"
