"""Configuration checks for the dedicated Wuji hand-tracking camera."""

from __future__ import annotations

from pathlib import Path
from typing import Any


def dedicated_retarget_camera(hardware: dict[str, Any]) -> tuple[str, str]:
    """Return the stable V4L path and identity of the dedicated USB camera."""
    wuji = hardware.get("wujihand")
    if not isinstance(wuji, dict):
        raise TypeError("hardware config is missing wujihand mapping")
    serial = str(wuji.get("retarget_camera_serial") or "").strip()
    if not serial:
        raise ValueError("wujihand.retarget_camera_serial must identify the USB camera")
    device = str(wuji.get("retarget_camera_device") or "").strip()
    if not device:
        raise ValueError("wujihand.retarget_camera_device must select a stable V4L path")

    cameras = hardware.get("cameras", {})
    devices = cameras.get("devices", []) if isinstance(cameras, dict) else []
    collection_serials = {
        str(device.get("serial") or "").strip()
        for device in devices
        if isinstance(device, dict)
    }
    if serial in collection_serials:
        raise ValueError(
            "wujihand.retarget_camera_serial must not reuse a collection camera: "
            f"{serial}"
        )
    return device, serial


def require_connected_retarget_camera(device: str, serial: str) -> str:
    """Validate the stable USB-camera link before OpenCV stream startup."""
    path = Path(device)
    if not path.exists():
        raise RuntimeError(
            f"dedicated retarget USB camera {serial} is not connected: {device}"
        )
    if serial not in path.name:
        raise RuntimeError(
            f"retarget camera path does not contain configured serial {serial}: {device}"
        )
    return str(path)
