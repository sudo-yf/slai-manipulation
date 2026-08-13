from pathlib import Path

import yaml

ROOT = Path(__file__).parents[1]
CONFIGS = ROOT / "configs"


def load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    assert isinstance(value, dict), f"{path} must contain a YAML mapping"
    return value


def resolve_ref(source: Path, reference: str) -> Path:
    relative_path = reference.split("#", maxsplit=1)[0]
    return (source.parent / relative_path).resolve()


def test_all_yaml_configs_are_versioned_and_parseable() -> None:
    paths = sorted(CONFIGS.rglob("*.yaml"))
    assert paths
    for path in paths:
        config = load_yaml(path)
        assert config.get("schema_version") == 1, path


def test_dataset_joint_schemas_are_explicit_and_dimensionally_consistent() -> None:
    dataset = load_yaml(CONFIGS / "dataset.yaml")
    groups = dataset["joint_groups"]

    for name, group in groups.items():
        assert len(group["joint_names"]) == group["dimension"], name
        assert len(set(group["joint_names"])) == group["dimension"], name

    assert dataset["schemas"]["real_v1"]["dimension"] == 26
    assert dataset["schemas"]["real_v1"]["joint_groups"] == ["ur5", "wujihand"]
    assert dataset["schemas"]["simulation_v1"]["dimension"] == 28
    assert dataset["schemas"]["simulation_v1"]["joint_groups"] == [
        "ur5",
        "wrist",
        "wujihand",
    ]
    assert dataset["cross_schema_conversion"] == "disabled"


def test_pose_dimensions_and_safety_defaults() -> None:
    for path in sorted((CONFIGS / "poses").rglob("*.yaml")):
        pose = load_yaml(path)
        assert pose["configured"] is False, path
        assert pose["units"] == "rad", path
        assert len(pose["joint_names"]) == len(pose["joint_positions"]), path
        assert len(set(pose["joint_names"])) == len(pose["joint_names"]), path
        assert "source" in pose["provenance"], path

        expected_dimension = 20 if pose["kind"] == "hand_preset" else None
        if pose["kind"] == "task_start":
            expected_dimension = 26
        elif pose["kind"] == "device_home":
            expected_dimension = 28
        assert len(pose["joint_positions"]) == expected_dimension, path


def test_task_references_exist_and_match_dataset_schema() -> None:
    dataset = load_yaml(CONFIGS / "dataset.yaml")
    for path in sorted((CONFIGS / "tasks").glob("*.yaml")):
        task = load_yaml(path)
        references = [
            task["dataset_ref"],
            task["control_profile_ref"],
            task["start_pose_ref"],
            *task["hand_presets"].values(),
        ]
        for reference in references:
            assert resolve_ref(path, reference).is_file(), f"broken reference in {path}: {reference}"

        schema = dataset["schemas"][task["state_schema"]]
        start_pose = load_yaml(resolve_ref(path, task["start_pose_ref"]))
        assert len(start_pose["joint_positions"]) == schema["dimension"]


def test_hardware_template_contains_no_device_identity() -> None:
    hardware = load_yaml(CONFIGS / "hardware.yaml")
    assert hardware["configured"] is False
    assert hardware["ur5"]["enabled"] is False
    assert hardware["ur5"]["host"] is None
    assert hardware["wujihand"]["usb_serial"] is None
    assert hardware["wujihand"]["product_serial"] is None
    assert hardware["iphone"] == {
        "enabled": False,
        "source": "pose_hub",
        "local_host": "127.0.0.1",
        "pose_port": 5005,
        "robot_state_port": 5006,
    }
    assert all(device["serial"] is None for device in hardware["cameras"]["devices"])
    assert hardware["stereo_camera"]["left_serial"] is None
    assert hardware["stereo_camera"]["right_serial"] is None
