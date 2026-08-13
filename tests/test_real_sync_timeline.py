from __future__ import annotations

import numpy as np
import pytest

from slai_mi.collection.synchronization import (
    BoundedBuffer,
    DeviceClockFit,
    RealFrameSynchronizer,
    TimedSample,
)


def sample(value, timestamp: float) -> TimedSample:
    return TimedSample(value=value, monotonic_s=timestamp)


def test_device_clock_fit_maps_drifted_device_time_to_monotonic() -> None:
    fit = DeviceClockFit(capacity=8)
    for device in range(6):
        fit.observe(float(device), 50.0 + 1.002 * device)
    assert fit.to_monotonic(8.0) == pytest.approx(58.016)


def test_bounded_buffer_interpolates_and_zoh_does_not_look_ahead() -> None:
    buffer = BoundedBuffer[np.ndarray](capacity=2)
    buffer.append(sample(np.array([0.0, 2.0]), 1.0))
    buffer.append(sample(np.array([2.0, 4.0]), 3.0))
    np.testing.assert_allclose(buffer.interpolate(2.0).value, [1.0, 3.0])
    assert buffer.zoh(2.0).monotonic_s == 1.0
    buffer.append(sample(np.array([4.0, 6.0]), 4.0))
    with pytest.raises(LookupError, match="at or before"):
        buffer.zoh(0.5)


def test_primary_camera_timeline_interpolates_state_and_holds_command() -> None:
    sync = RealFrameSynchronizer(max_camera_age_ms=50.0)
    primary = sample("p", 1.0)
    sync.cameras["secondary"].append(sample("s", 0.995))
    sync.cameras["wrist"].append(sample("w", 1.006))
    for buffer, low, high in (
        (sync.ur5, [0.0, 2.0], [2.0, 4.0]),
        (sync.wuji, [10.0], [14.0]),
    ):
        buffer.append(sample(np.asarray(low), 0.9))
        buffer.append(sample(np.asarray(high), 1.1))
    sync.commands.append(sample({"target": 1}, 0.98))

    frame = sync.synchronize(primary, now_s=1.01)

    assert frame.cameras == {"primary": "p", "secondary": "s", "wrist": "w"}
    np.testing.assert_allclose(frame.ur5_state, [1.0, 3.0])
    np.testing.assert_allclose(frame.wuji_state, [12.0])
    assert frame.command == {"target": 1}
    assert frame.valid
    assert frame.diagnostics["camera.wrist"].skew_ms == pytest.approx(6.0)
    assert frame.diagnostics["command"].age_ms == pytest.approx(20.0)


def test_sync_marks_excessive_camera_skew_and_stale_command_invalid() -> None:
    sync = RealFrameSynchronizer(max_camera_skew_ms=10, max_command_age_ms=20)
    primary = sample("p", 1.0)
    sync.cameras["secondary"].append(sample("s", 0.98))
    sync.cameras["wrist"].append(sample("w", 1.0))
    for buffer in (sync.ur5, sync.wuji):
        buffer.append(sample(np.array([0.0]), 0.99))
        buffer.append(sample(np.array([1.0]), 1.01))
    sync.commands.append(sample("old", 0.9))

    frame = sync.synchronize(primary, now_s=1.0)

    assert not frame.valid
    assert not frame.diagnostics["camera.secondary"].valid
    assert not frame.diagnostics["command"].valid
