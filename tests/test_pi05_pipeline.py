from __future__ import annotations

import json

import numpy as np
import pytest

from slai_mi.apps.pi05 import main
from slai_mi.datasets.pi05 import (
    discover_source_roots,
    policy_rgb,
    stage_episode,
    validate_pi05_source,
)
from slai_mi.evaluation.heldout import select_horizon_rows


class _Dataset:
    def __init__(self) -> None:
        self.items = []
        for index in range(4):
            self.items.append(
                {
                    "observation.state": np.full(26, index, dtype=np.float32),
                    "observation.tcp_pose": np.full(6, index + 10, dtype=np.float32),
                    "action": np.full(26, index + 20, dtype=np.float32),
                    "observation.images.primary_rgb": np.zeros((224, 224, 3), dtype=np.uint8),
                    "observation.images.secondary_rgb": np.zeros((3, 224, 224), dtype=np.uint8),
                    "task": "pick block",
                }
            )

    def __getitem__(self, index: int) -> dict:
        return self.items[index]


def test_stage_episode_builds_32d_state_and_15hz_view(tmp_path) -> None:
    path = tmp_path / "episode.npz"
    assert stage_episode(_Dataset(), 0, 4, 2, path) == 2
    with np.load(path) as episode:
        assert episode["state"].shape == (2, 32)
        assert episode["actions"].shape == (2, 26)
        assert episode["primary_rgb"].shape == (2, 224, 224, 3)
        assert episode["task"].item() == "pick block"


def test_policy_rgb_rejects_non_rgb() -> None:
    with pytest.raises(ValueError, match="RGB"):
        policy_rgb(np.zeros((224, 224)))


def test_legacy_two_camera_contract_is_an_explicit_pi05_source(tmp_path) -> None:
    root = tmp_path / "batch"
    (root / "meta").mkdir(parents=True)
    features = {
        key: {"shape": shape}
        for key, shape in {
            "observation.images.primary_rgb": [480, 640, 3],
            "observation.images.secondary_rgb": [480, 640, 3],
            "observation.state": [26],
            "observation.tcp_pose": [6],
            "action": [26],
        }.items()
    }
    (root / "meta" / "info.json").write_text(
        json.dumps(
            {
                "codebase_version": "v3.0",
                "fps": 30,
                "features": features,
                "total_episodes": 2,
                "total_frames": 100,
            }
        )
    )
    (root / "meta" / "robot_teleoperation_contract.json").write_text(
        json.dumps({"contract_id": "robot_teleoperation.ur5_wuji.pi05_cartesian.v1"})
    )

    assert discover_source_roots(tmp_path) == (root,)
    assert validate_pi05_source(root)["episodes"] == 2


def test_heldout_sampling_never_crosses_episode_boundary() -> None:
    episodes = np.repeat(np.arange(2), 40)
    frames = np.tile(np.arange(40), 2)
    selected = select_horizon_rows(episodes, frames, count=6)
    assert len(selected) == 6
    assert np.all(frames[selected] <= 11)


def test_pi05_cli_defaults_to_a_non_executing_plan(capsys) -> None:
    assert main(["train"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["app"] == "pi05"
    assert plan["mode"] == "dry-run"
    assert plan["repo_id"] == "local/slai-ur5-wujihand-pi05"
