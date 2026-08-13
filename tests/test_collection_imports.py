def test_ur5_runtime_imports_without_optional_hardware_sdks() -> None:
    from slai_mi.devices.ur5 import runtime

    assert callable(runtime.validate_robot_health)
