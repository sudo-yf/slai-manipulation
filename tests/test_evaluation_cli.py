import json

import numpy as np

from slai_mi.apps.evaluate import main


def test_action_cli_writes_json(tmp_path) -> None:
    source = tmp_path / "actions.npz"
    output = tmp_path / "report.json"
    np.savez(source, predicted=np.ones((2, 8)), target=np.zeros((2, 8)), ranges=np.ones(8))
    assert main(["actions", str(source), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["mae"] == 1.0
