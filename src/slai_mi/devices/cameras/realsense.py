"""Lazy RealSense discovery helpers; importing this module does not open hardware."""


def discover_serials() -> tuple[str, ...]:
    try:
        import pyrealsense2 as rs
    except ImportError as exc:
        raise RuntimeError("RealSense support requires pyrealsense2") from exc
    return tuple(device.get_info(rs.camera_info.serial_number) for device in rs.context().query_devices())
