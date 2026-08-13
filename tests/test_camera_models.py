import pytest

from slai_mi.devices.cameras import (
    CameraConfig,
    CameraFrame,
    FrameSynchronizer,
    validate_camera_set,
)


def test_three_camera_contract_requires_unique_serials():
    cameras = [CameraConfig(f"cam{i}", str(i), 640, 480, 30) for i in range(3)]
    assert len(validate_camera_set(cameras)) == 3
    cameras[2] = CameraConfig("cam2", "1", 640, 480, 30)
    with pytest.raises(ValueError, match="serials"):
        validate_camera_set(cameras)


def test_frame_synchronizer_pairs_nearest_host_times():
    sync = FrameSynchronizer(("left", "right"))
    sync.add(CameraFrame("left", 1, 1.0, 10.0, object()))
    sync.add(CameraFrame("right", 2, 1.0, 10.01, object()))
    result = sync.read(0.01)
    assert set(result) == {"left", "right"}
