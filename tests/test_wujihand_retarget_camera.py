import pytest

from slai_mi.devices.wujihand.retarget_camera import (
    dedicated_retarget_camera,
    require_connected_retarget_camera,
)


def test_dedicated_retarget_camera() -> None:
    hardware = {
        "wujihand": {
            "retarget_camera_serial": "retarget-1",
            "retarget_camera_device": "/dev/v4l/by-id/retarget-1-video-index0",
        },
        "cameras": {"devices": [{"role": "primary", "serial": "collection-1"}]},
    }
    assert dedicated_retarget_camera(hardware) == (
        "/dev/v4l/by-id/retarget-1-video-index0",
        "retarget-1",
    )


@pytest.mark.parametrize("serial", [None, "", "collection-1"])
def test_dedicated_retarget_camera_rejects_missing_or_shared_serial(serial) -> None:
    hardware = {
        "wujihand": {
            "retarget_camera_serial": serial,
            "retarget_camera_device": "/dev/v4l/by-id/retarget-1-video-index0",
        },
        "cameras": {"devices": [{"role": "primary", "serial": "collection-1"}]},
    }
    with pytest.raises(ValueError):
        dedicated_retarget_camera(hardware)


def test_connected_retarget_camera_preflight(tmp_path) -> None:
    device = tmp_path / "HB202400001-video-index0"
    device.touch()
    assert require_connected_retarget_camera(str(device), "HB202400001") == str(device)
    with pytest.raises(RuntimeError, match="does not contain configured serial"):
        require_connected_retarget_camera(str(device), "another-camera")
    with pytest.raises(RuntimeError, match="is not connected"):
        require_connected_retarget_camera(str(tmp_path / "missing"), "missing")
