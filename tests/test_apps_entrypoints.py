"""Tests for safe application entry points."""

import json

import pytest

from slai_mi.apps import collect_real, collect_sim, deploy_real, teleop_real, teleop_sim


@pytest.mark.parametrize(
    ("entrypoint", "name"),
    [
        (teleop_real.main, "teleop_real"),
        (collect_real.main, "collect_real"),
        (teleop_sim.main, "teleop_sim"),
        (collect_sim.main, "collect_sim"),
    ],
)
def test_entrypoints_default_to_dry_run(entrypoint, name, capsys) -> None:
    assert entrypoint([]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["app"] == name
    assert output["mode"] == "dry-run"


@pytest.mark.parametrize("entrypoint", [teleop_real.main, collect_real.main])
def test_real_entrypoints_require_confirmation(entrypoint) -> None:
    with pytest.raises(SystemExit, match="Real hardware remains disabled"):
        entrypoint(["--execute-real"])


def test_collection_rejects_zero_episodes() -> None:
    with pytest.raises(SystemExit, match="episodes must be at least 1"):
        collect_sim.main(["--episodes", "0"])


def test_real_collection_continuous_dry_run(capsys) -> None:
    assert collect_real.main(["--continuous"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["episodes"] == "continuous"
    assert output["dashboard"] == "http://0.0.0.0:8765"
    assert output["strategy"] == "ur5e_wujihand_26dof_collection"
    assert output["physical_dof"] == 26
    assert output["recorded_dof"] == 28
    assert output["synthetic_channels"] == ["wrist_fe_ru_zero"]
    assert "wrist_sensor" not in output["enabled_devices"]


def test_real_teleop_defaults_to_retargeting_group(capsys) -> None:
    assert teleop_real.main([]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["strategy"] == "ur5e_wujihand_retargeting"
    assert output["control_modes"]["wujihand"] == "camera_retargeting"


def test_wrist_teleop_group_is_selectable(capsys) -> None:
    strategy = "ur5e_wrist_8dof_teleop"
    assert teleop_real.main(["--strategy", strategy]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["strategy_status"] == "ready"
    assert output["physical_dof"] == 8
    assert output["synthetic_channels"] == []
    assert output["enabled_devices"] == ["cameras", "spacemouse", "ur5", "wrist_sensor"]
    assert output["control_modes"]["wrist"] == "wrist_master_to_slave"


def test_real_deployment_defaults_to_dry_run(capsys) -> None:
    assert deploy_real.main(["--config", "configs/inference_pi05_round10.yaml"]) == 0
    assert json.loads(capsys.readouterr().out)["mode"] == "dry-run"
