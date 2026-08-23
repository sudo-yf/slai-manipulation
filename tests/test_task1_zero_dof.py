from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from slai_mi.input_schema import compose_capture_vector, load_input_schema


def test_task1_capture_has_two_explicit_zero_wrist_dimensions() -> None:
    schema = load_input_schema()
    state = compose_capture_vector(
        schema,
        "state",
        {
            "ur5": SimpleNamespace(actual_q=np.arange(6, dtype=np.float32)),
            "wuji": SimpleNamespace(actual_q=np.arange(20, dtype=np.float32)),
        },
    )
    action = compose_capture_vector(
        schema,
        "action",
        {
            "ur5": SimpleNamespace(target_tcp_speed=np.arange(6, dtype=np.float32)),
            "wuji": SimpleNamespace(command_q=np.arange(20, dtype=np.float32)),
        },
    )
    assert state.shape == (28,)
    assert action.shape == (28,)
    np.testing.assert_array_equal(state[-2:], [0.0, 0.0])
    np.testing.assert_array_equal(action[-2:], [0.0, 0.0])


def test_default_real_collection_task_is_task1() -> None:
    task = yaml.safe_load(
        Path("configs/tasks/task1.yaml").read_text(encoding="utf-8")
    )
    assert task["task"]["id"] == "task1"
