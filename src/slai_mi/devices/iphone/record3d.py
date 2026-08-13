"""Direct USB pose source for Record3D with an injectable stream backend."""

from __future__ import annotations

import threading
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .protocol import make_iphone_pose


class Record3DPoseClient:
    def __init__(
        self,
        *,
        udid: str | None = None,
        device_index: int = 0,
        read_timeout_s: float = 10.0,
        stream_type=None,
    ):
        if device_index < 0 or read_timeout_s <= 0:
            raise ValueError("invalid Record3D settings")
        self.udid, self.device_index, self.read_timeout_s = udid, device_index, read_timeout_s
        self._stream_type, self._stream = stream_type, None
        self._frame_ready, self._stream_stopped = threading.Event(), threading.Event()
        self._sequence = 0

    def connect(self) -> None:
        stream_type = self._stream_type
        if stream_type is None:
            try:
                from record3d import Record3DStream
            except ImportError as exc:
                raise RuntimeError("Record3D support requires the record3d package") from exc
            stream_type = Record3DStream
        devices = list(stream_type.get_connected_devices())
        matches = [item for item in devices if self.udid is not None and item.udid == self.udid]
        if self.udid is not None:
            if not matches:
                raise RuntimeError(f"Record3D cannot see iPhone {self.udid}")
            device = matches[0]
        elif self.device_index < len(devices):
            device = devices[self.device_index]
        else:
            raise RuntimeError(f"Record3D found {len(devices)} device(s)")
        self._stream = stream_type()
        self._stream.on_new_frame = self._frame_ready.set
        self._stream.on_stream_stopped = self._stream_stopped.set
        self._stream.connect(device)

    def receive(self):
        if self._stream is None:
            raise RuntimeError("Record3D client is not connected")
        if not self._frame_ready.wait(self.read_timeout_s):
            raise TimeoutError("no Record3D frame arrived")
        self._frame_ready.clear()
        if self._stream_stopped.is_set():
            raise ConnectionError("Record3D stream stopped")
        raw = self._stream.get_camera_pose()
        quaternion = np.array([raw.qx, raw.qy, raw.qz, raw.qw], dtype=float)
        norm = np.linalg.norm(quaternion)
        if not np.isfinite(norm) or norm < 1e-6:
            raise ValueError("Record3D returned a degenerate quaternion")
        transform = np.eye(4)
        transform[:3, :3] = Rotation.from_quat(quaternion / norm).as_matrix()
        transform[:3, 3] = [raw.tx, raw.ty, raw.tz]
        pose = make_iphone_pose(
            sequence=self._sequence,
            timestamp_s=time.monotonic(),
            sent_at_unix_s=time.time(),
            tracking="available_unverified",
            world_from_camera=transform,
        )
        self._sequence += 1
        return pose

    def close(self) -> None:
        if self._stream is not None:
            self._stream.disconnect()
        self._stream = None
