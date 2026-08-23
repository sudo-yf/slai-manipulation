import json
from pathlib import Path

import numpy as np

from slai_mi.apps.evaluate import main


def test_action_cli_writes_json(tmp_path) -> None:
    source = tmp_path / "actions.npz"
    output = tmp_path / "report.json"
    np.savez(source, predicted=np.ones((2, 8)), target=np.zeros((2, 8)), ranges=np.ones(8))
    assert main(["actions", str(source), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["mae"] == 1.0


def test_dataset_cli_dispatches_wrist_contract(monkeypatch, tmp_path, capsys) -> None:
    root = tmp_path / "wrist"
    meta = root / "meta"
    meta.mkdir(parents=True)
    (meta / "robot_teleoperation_contract.json").write_text(
        json.dumps(
            {"contract_id": ("robot_teleoperation.ur5_wrist.three_rgb_cartesian_rot6d_columns.v1")}
        )
    )
    monkeypatch.setattr(
        "slai_mi.datasets.lerobot_v3.configured.ConfiguredDatasetContract.validate_root",
        lambda _self, selected: {"root": str(selected), "frames": 5},
    )

    assert main(["dataset", str(root)]) == 0

    assert json.loads(capsys.readouterr().out) == {"frames": 5, "root": str(Path(root))}
