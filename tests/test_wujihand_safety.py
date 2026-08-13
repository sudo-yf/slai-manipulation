import pytest

from slai_mi.devices.wujihand import (
    JointLimits,
    OneEuroFilter,
    SafeCommandLimiter,
    WujiHandClient,
    WujiHandRuntime,
)


def test_command_limiter_clamps_position_and_velocity():
    limiter = SafeCommandLimiter(JointLimits((-1.0, -2.0), (1.0, 2.0)), (0.5, 1.0))
    assert limiter.reset((0.0, 0.0), 0.0) == (0.0, 0.0)
    assert limiter.limit((10.0, -10.0), 1.0) == (0.5, -1.0)


def test_one_euro_rejects_non_monotonic_samples():
    filt = OneEuroFilter()
    filt.filter((0.0, 1.0), 1.0)
    with pytest.raises(ValueError, match="strictly increasing"):
        filt.filter((0.0, 1.0), 1.0)


def test_runtime_writes_only_limited_commands():
    class Backend:
        def __init__(self): self.writes = []
        def read_positions(self): return (0.0, 0.0)
        def write_positions(self, values): self.writes.append(tuple(values))
        def disable(self): pass

    backend = Backend()
    client = WujiHandClient(lambda: backend, expected_joints=2)
    client.connect()
    limiter = SafeCommandLimiter(JointLimits((-1.0, -1.0), (1.0, 1.0)), (0.5, 0.5))
    limiter.reset((0.0, 0.0), 0.0)
    runtime = WujiHandRuntime(client, limiter, lambda: (2.0, -2.0))
    assert runtime.step(1.0) == (0.5, -0.5)
    assert backend.writes == [(0.5, -0.5)]
