from __future__ import annotations

import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from slai_mi.collection.vla_recorder import (
    SourceSample,
    SynchronizedInputs,
    assemble_configured_frame,
)
from slai_mi.datasets.lerobot_v3.configured import ConfiguredDatasetContract
from slai_mi.devices.wrist_sensor.teleop import WristMasterSlaveController
from slai_mi.input_schema import load_input_schema, select_transformed_vector, select_vector
from slai_mi.ui.collection_dashboard import CollectionDashboardProvider


def _sample(value, sequence: int = 1) -> SourceSample:
    return SourceSample(value, 10.0, 10.0, sequence)


def test_wrist_collection_frame_is_exactly_8d_and_pi05_view_is_11d() -> None:
    schema = load_input_schema("configs/input_schemas/ur5e_wrist_8dof.yaml")
    contract = ConfiguredDatasetContract(schema)
    ur5 = SimpleNamespace(
        actual_q=np.arange(6, dtype=np.float32),
        actual_tcp_pose=np.zeros(6, dtype=np.float32),
        actual_tcp_speed=np.zeros(6, dtype=np.float32),
        target_qd=np.zeros(6, dtype=np.float32),
        target_tcp_speed=np.arange(6, dtype=np.float32) / 10.0,
    )
    wrist = SimpleNamespace(
        actual_q=np.asarray([0.1, -0.2], dtype=np.float32),
        target_q=np.asarray([0.3, -0.4], dtype=np.float32),
    )
    mouse = SimpleNamespace(
        axes=np.zeros(6, dtype=np.float32), buttons=np.zeros(12, dtype=np.int64)
    )
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    inputs = SynchronizedInputs(
        cameras={role: _sample(image) for role in ("primary", "wrist", "secondary")},
        channels={"ur5": _sample(ur5), "wrist": _sample(wrist), "spacemouse": _sample(mouse)},
    )

    frame = assemble_configured_frame(
        inputs,
        "put the block in the box",
        schema=schema,
        validator=contract.validate_frame,
        now=10.0,
    )

    np.testing.assert_allclose(frame["observation.state"], [0, 1, 2, 3, 4, 5, 0.1, -0.2])
    np.testing.assert_allclose(frame["action"], [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.3, -0.4])
    policy_parts = [
        select_transformed_vector(frame[item["key"]], item, f"state[{index}]")
        for index, item in enumerate(schema["pi05"]["state"]["sources"])
    ]
    assert len(np.concatenate(policy_parts)) == 11
    assert len(select_vector(frame["action"], schema["pi05"]["action"], "action")) == 8


def test_wrist_controller_automatically_starts_master_stream(monkeypatch) -> None:
    commands: list[str] = []
    targets: list[tuple[float, float]] = []

    class Response:
        def __init__(self):
            self.ok = True
            self.fields = {
                "zero_valid": "1",
                "enc0_deg": "1000",
                "enc1_deg": "2000",
            }

    class Teleop:
        def __init__(self, *_args):
            self.sent_frame = False

        def command(self, command, _expected, timeout_s=2.0):
            commands.append(command)
            return Response()

        def read_line(self, _timeout):
            if not self.sent_frame:
                self.sent_frame = True
                return "OK TELE enc0_deg=1100 enc1_deg=1800"
            time.sleep(0.001)
            return None

        def write_command(self, command):
            commands.append(command)

        def close(self):
            return None

    class Unwrapper:
        def reset(self, value):
            self.value = value

        def update(self, value):
            return value - self.value

    class Filter:
        def __init__(self, **_kwargs):
            pass

        def reset(self, _value):
            pass

        def filter(self, value, _dt):
            return value

    class Shaper:
        def __init__(self, *_args):
            pass

        def reset(self, *_args):
            pass

        def update(self, fe, ru, _dt):
            return fe, ru

    state = SimpleNamespace(fe_deg=0.0, ru_deg=0.0, target_fe_deg=0.0, target_ru_deg=0.0)

    class Controller:
        bounds = object()

        def __init__(self, *_args):
            pass

        def prepare(self, **_kwargs):
            return state

        def stream_target_deg(self, fe, ru):
            targets.append((fe, ru))

        def read_state(self):
            return state

        def shutdown(self, **_kwargs):
            return None

    class Client:
        def __init__(self, *_args, **_kwargs):
            pass

        def connect(self):
            return self

        def close(self):
            return None

    config = SimpleNamespace(
        stream=SimpleNamespace(master_period_ms=10, command_hz=50.0, state_hz=20.0),
        filter=SimpleNamespace(one_euro_min_cutoff=8.0, one_euro_beta=0.15, one_euro_d_cutoff=1.0),
        settling=SimpleNamespace(tolerance_deg=0.3),
    )

    monkeypatch.setattr(
        "slai_mi.devices.wrist_sensor.teleop._vendor_imports",
        lambda: (
            Unwrapper,
            Teleop,
            lambda fields, key: int(fields[key]) / 100.0,
            lambda line: SimpleNamespace(
                ok=True,
                command="TELE",
                fields={"enc0_deg": "1100", "enc1_deg": "1800"},
            ),
            Filter,
            Shaper,
            Controller,
            lambda _path: config,
            lambda enc0, enc1, _config: (-enc1, enc0),
            Client,
            lambda value: value,
            lambda value: value,
        ),
    )
    controller = WristMasterSlaveController(Path("unused.yaml"))
    with controller:
        deadline = time.monotonic() + 1.0
        while not targets and time.monotonic() < deadline:
            time.sleep(0.005)
        assert targets == [(2.0, 1.0)]
        assert controller.state().actual_q.shape == (2,)

    assert commands[:3] == ["STOP", "SET_PERIOD 10", "START"]


def test_collection_dashboard_accepts_wrist_group_without_wuji() -> None:
    schema_path = "configs/input_schemas/ur5e_wrist_8dof.yaml"
    provider = CollectionDashboardProvider(
        {
            "input_schema": schema_path,
            "ur5": {"enabled": True},
            "wujihand": {"enabled": False},
            "wrist_sensor": {"enabled": True},
            "cameras": {
                "enabled": True,
                "devices": [
                    {"role": "primary", "serial": "1"},
                    {"role": "wrist", "serial": "2"},
                    {"role": "secondary", "serial": "3"},
                ],
            },
        },
        "task",
    )
    now = time.monotonic()
    image = np.zeros((480, 640, 3), dtype=np.uint8)
    inputs = SynchronizedInputs(
        cameras={
            role: SourceSample(image, now, now, index)
            for index, role in enumerate(("primary", "wrist", "secondary"))
        },
        channels={
            "ur5": SourceSample(
                SimpleNamespace(actual_q=np.zeros(6, dtype=np.float32)), now, now, 1
            ),
            "wrist": SourceSample(
                SimpleNamespace(actual_q=np.zeros(2, dtype=np.float32)), now, now, 1
            ),
            "spacemouse": SourceSample(
                SimpleNamespace(
                    axes=np.zeros(6, dtype=np.float32),
                    buttons=np.zeros(12, dtype=np.int64),
                ),
                now,
                now,
                1,
            ),
        },
    )

    provider.observe_inputs(inputs)

    status = provider.status()
    assert status["devices"]["ur5"]["state"] == "active"
    assert status["devices"]["wrist"]["state"] == "active"
    assert status["devices"]["wuji"]["state"] == "inactive"
    assert len(status["dof"]["values"]) == 8
