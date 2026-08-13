"""Synchronized RGB-D acquisition from the two model-facing D435I cameras."""

from __future__ import annotations

import threading
import time
from collections import deque
from contextlib import suppress
from dataclasses import dataclass
from typing import Any, Self

import numpy as np

from slai_mi.datasets.lerobot_v3.schema import FPS, IMAGE_HEIGHT, IMAGE_WIDTH, MAX_CAMERA_SKEW_MS

PRIMARY_ROLE = "primary"
SECONDARY_ROLE = "secondary"
ROLES = (PRIMARY_ROLE, SECONDARY_ROLE)
MAX_DEPTH_MM = 4095


@dataclass(frozen=True)
class RGBDFrame:
    rgb: np.ndarray
    depth_mm: np.ndarray
    color_timestamp_s: float
    depth_timestamp_s: float
    host_timestamp_s: float
    color_frame_number: int
    depth_frame_number: int
    valid_fraction: float
    clipped_fraction: float


@dataclass(frozen=True)
class RGBDPair:
    primary: RGBDFrame
    secondary: RGBDFrame

    @property
    def skew_ms(self) -> float:
        return abs(self.primary.host_timestamp_s - self.secondary.host_timestamp_s) * 1000.0


class FramePairer:
    """Pair frames by host arrival time while discarding irrecoverably old frames."""

    def __init__(self, max_skew_ms: float = MAX_CAMERA_SKEW_MS, queue_size: int = 8) -> None:
        if max_skew_ms <= 0.0 or queue_size < 2:
            raise ValueError("frame pairing limits must be positive")
        self.max_skew_s = max_skew_ms / 1000.0
        self._queues = {role: deque(maxlen=queue_size) for role in ROLES}
        self._condition = threading.Condition()

    def add(self, role: str, frame: RGBDFrame) -> None:
        if role not in self._queues:
            raise ValueError(f"unknown camera role: {role}")
        with self._condition:
            self._queues[role].append(frame)
            self._condition.notify_all()

    def read(self, timeout_s: float, errors: dict[str, BaseException] | None = None) -> RGBDPair:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if errors:
                    role, error = next(iter(errors.items()))
                    raise RuntimeError(f"{role} RealSense failed: {error}") from error
                pair = self._take_best_locked()
                if pair is not None:
                    return pair
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("timed out waiting for a synchronized RGB-D pair")
                self._condition.wait(min(remaining, 0.05))

    def wake(self) -> None:
        with self._condition:
            self._condition.notify_all()

    def _take_best_locked(self) -> RGBDPair | None:
        primary = self._queues[PRIMARY_ROLE]
        secondary = self._queues[SECONDARY_ROLE]
        while primary and secondary:
            best_i, best_j = min(
                ((i, j) for i in range(len(primary)) for j in range(len(secondary))),
                key=lambda indices: abs(
                    primary[indices[0]].host_timestamp_s - secondary[indices[1]].host_timestamp_s
                ),
            )
            first = primary[best_i]
            second = secondary[best_j]
            if abs(first.host_timestamp_s - second.host_timestamp_s) <= self.max_skew_s:
                for _ in range(best_i + 1):
                    primary.popleft()
                for _ in range(best_j + 1):
                    secondary.popleft()
                return RGBDPair(first, second)
            if primary[0].host_timestamp_s < secondary[0].host_timestamp_s:
                primary.popleft()
            else:
                secondary.popleft()
        return None


class DualRealSenseRGBD:
    def __init__(
        self,
        primary_serial: str,
        secondary_serial: str,
        *,
        fps: int = FPS,
        max_skew_ms: float = MAX_CAMERA_SKEW_MS,
        rs_module: Any | None = None,
        monotonic=time.monotonic,
    ) -> None:
        if not primary_serial or not secondary_serial or primary_serial == secondary_serial:
            raise ValueError("two distinct RealSense serials are required")
        self.serials = {PRIMARY_ROLE: primary_serial, SECONDARY_ROLE: secondary_serial}
        self.fps = int(fps)
        self._rs = rs_module
        self._monotonic = monotonic
        self._pairer = FramePairer(max_skew_ms=max_skew_ms)
        self._stop = threading.Event()
        self._threads: list[threading.Thread] = []
        self._errors: dict[str, BaseException] = {}
        self._calibration: dict[str, dict] = {}
        self._state_lock = threading.Lock()

    def start(self, timeout_s: float = 20.0) -> None:
        if self._threads:
            raise RuntimeError("dual RealSense source is already running")
        if self._rs is None:
            import pyrealsense2 as rs

            self._rs = rs
        try:
            deadline = self._monotonic() + timeout_s
            for role in ROLES:
                thread = threading.Thread(
                    target=self._run_camera,
                    args=(role, self.serials[role]),
                    name=f"VLA-D435I-{role}",
                    daemon=True,
                )
                thread.start()
                self._threads.append(thread)
                while self._monotonic() < deadline:
                    with self._state_lock:
                        if role in self._errors:
                            error = self._errors[role]
                            raise RuntimeError(
                                f"{role} RealSense startup failed: {error}"
                            ) from error
                        if role in self._calibration:
                            break
                    time.sleep(0.01)
                else:
                    raise TimeoutError(f"timed out waiting for the first {role} RGB-D frame")
        except BaseException:
            self.stop()
            raise

    def read_pair(self, timeout_s: float = 1.0) -> RGBDPair:
        deadline = self._monotonic() + timeout_s
        while True:
            with self._state_lock:
                errors = dict(self._errors)
            if errors:
                role, error = next(iter(errors.items()))
                raise RuntimeError(f"{role} RealSense failed: {error}") from error
            remaining = deadline - self._monotonic()
            if remaining <= 0.0:
                raise TimeoutError("timed out waiting for a synchronized RGB-D pair")
            try:
                return self._pairer.read(min(remaining, 0.1))
            except TimeoutError:
                pass

    def calibration(self) -> dict[str, dict]:
        with self._state_lock:
            if not all(role in self._calibration for role in ROLES):
                raise RuntimeError("camera calibration is unavailable before startup completes")
            return {role: dict(values) for role, values in self._calibration.items()}

    def stop(self) -> None:
        self._stop.set()
        self._pairer.wake()
        for thread in self._threads:
            thread.join(timeout=3.0)
        self._threads.clear()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.stop()

    def _run_camera(self, role: str, serial: str) -> None:
        rs = self._rs
        pipeline = rs.pipeline()
        started = False
        try:
            config = rs.config()
            config.enable_device(serial)
            config.enable_stream(
                rs.stream.color, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.rgb8, self.fps
            )
            config.enable_stream(
                rs.stream.depth, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.z16, self.fps
            )
            profile = pipeline.start(config)
            started = True
            calibration = _calibration(profile, serial, role, rs)
            align = rs.align(rs.stream.color)
            consecutive_timeouts = 0
            startup_restarts = 0
            ready = False
            while not self._stop.is_set():
                try:
                    frames = align.process(pipeline.wait_for_frames(1500))
                except RuntimeError as exc:
                    if "Frame didn't arrive" not in str(exc):
                        raise
                    consecutive_timeouts += 1
                    if consecutive_timeouts >= 5:
                        if not ready and startup_restarts < 1:
                            pipeline.stop()
                            started = False
                            time.sleep(1.0)
                            if self._stop.is_set():
                                break
                            profile = pipeline.start(config)
                            started = True
                            calibration = _calibration(profile, serial, role, rs)
                            align = rs.align(rs.stream.color)
                            consecutive_timeouts = 0
                            startup_restarts += 1
                            continue
                        raise RuntimeError(
                            f"no RGB-D frame arrived from D435I {serial} after "
                            f"{consecutive_timeouts * 1.5:.1f} s"
                        ) from exc
                    continue
                consecutive_timeouts = 0
                color = frames.get_color_frame()
                depth = frames.get_depth_frame()
                if not color or not depth:
                    continue
                host_time = self._monotonic()
                rgb = np.asanyarray(color.get_data()).copy()
                raw_depth = np.asanyarray(depth.get_data()).copy()
                clipped_fraction = float(
                    np.count_nonzero(raw_depth > MAX_DEPTH_MM) / raw_depth.size
                )
                depth_mm = np.minimum(raw_depth, MAX_DEPTH_MM).astype(np.uint16, copy=False)
                valid_fraction = float(np.count_nonzero(raw_depth) / raw_depth.size)
                self._pairer.add(
                    role,
                    RGBDFrame(
                        rgb=np.ascontiguousarray(rgb),
                        depth_mm=np.ascontiguousarray(depth_mm[..., None]),
                        color_timestamp_s=float(color.get_timestamp()) / 1000.0,
                        depth_timestamp_s=float(depth.get_timestamp()) / 1000.0,
                        host_timestamp_s=host_time,
                        color_frame_number=int(color.get_frame_number()),
                        depth_frame_number=int(depth.get_frame_number()),
                        valid_fraction=valid_fraction,
                        clipped_fraction=clipped_fraction,
                    ),
                )
                if not ready:
                    with self._state_lock:
                        self._calibration[role] = calibration
                    ready = True
        except Exception as exc:  # noqa: BLE001 - worker failures are relayed to the caller
            with self._state_lock:
                self._errors[role] = exc
            self._pairer.wake()
        finally:
            if started:
                with suppress(RuntimeError):
                    pipeline.stop()


def _calibration(profile: Any, expected_serial: str, role: str, rs: Any) -> dict:
    device = profile.get_device()
    serial = device.get_info(rs.camera_info.serial_number)
    if serial != expected_serial:
        raise RuntimeError(f"opened RealSense {serial}, expected {expected_serial}")
    usb_type = device.get_info(rs.camera_info.usb_type_descriptor).strip()
    if not usb_type.startswith("3"):
        raise RuntimeError(
            f"D435I {serial} is connected as USB {usb_type}; RGB-D 640x480@{FPS} "
            "requires a direct USB 3.x connection"
        )
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    depth_profile = profile.get_stream(rs.stream.depth).as_video_stream_profile()
    color_intrinsics = color_profile.get_intrinsics()
    depth_intrinsics = depth_profile.get_intrinsics()
    extrinsics = depth_profile.get_extrinsics_to(color_profile)
    return {
        "role": role,
        "serial": serial,
        "model": device.get_info(rs.camera_info.name),
        "firmware_version": device.get_info(rs.camera_info.firmware_version),
        "usb_type_descriptor": usb_type,
        "fps": FPS,
        "color_format": "rgb8",
        "depth_format": "z16_mm_aligned_to_color",
        "depth_scale_m_per_unit": float(device.first_depth_sensor().get_depth_scale()),
        "color_intrinsics": _intrinsics_dict(color_intrinsics),
        "depth_intrinsics": _intrinsics_dict(depth_intrinsics),
        "depth_to_color_extrinsics": {
            "rotation": [float(value) for value in extrinsics.rotation],
            "translation_m": [float(value) for value in extrinsics.translation],
        },
    }


def _intrinsics_dict(intrinsics: Any) -> dict:
    return {
        "width": int(intrinsics.width),
        "height": int(intrinsics.height),
        "fx": float(intrinsics.fx),
        "fy": float(intrinsics.fy),
        "ppx": float(intrinsics.ppx),
        "ppy": float(intrinsics.ppy),
        "distortion_model": str(intrinsics.model),
        "coefficients": [float(value) for value in intrinsics.coeffs],
    }
