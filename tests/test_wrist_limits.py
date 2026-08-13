from slai_mi.devices.wrist_sensor import CONTROL_LIMITS_FE_RU


def test_wrist_control_limits_straddle_zero():
    for limits in CONTROL_LIMITS_FE_RU:
        lower, upper = limits.radians
        assert lower < 0 < upper
