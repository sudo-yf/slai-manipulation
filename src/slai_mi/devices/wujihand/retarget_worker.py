"""JSON-RPC MediaPipe/Wuji retarget worker for the pinned Python 3.11 environment."""

from __future__ import annotations

import contextlib
import json
import sys
import time
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path

import numpy as np

from slai_mi.devices.wujihand.retarget_camera import (
    dedicated_retarget_camera,
    require_connected_retarget_camera,
)
from slai_mi.devices.wujihand.tracking import LandmarkGate

PROJECT_ROOT = Path(__file__).resolve().parents[4]
LANDMARK_STALE_S = 0.2


class LocalRetargetProvider:
    def __init__(self, request: dict) -> None:
        hardware = request["hardware"]
        self.camera_device, self.camera_serial = dedicated_retarget_camera(hardware)
        require_connected_retarget_camera(self.camera_device, self.camera_serial)
        third_party = PROJECT_ROOT / "third_party/wuji-retargeting"
        sys.path.insert(0, str(third_party))
        with contextlib.redirect_stdout(sys.stderr):
            import cv2
            from example.input_devices.realsense_mediapipe import RealsenseMediaPipe
            from wuji_retargeting import Retargeter

            wuji = hardware["wujihand"]
            self.detector = RealsenseMediaPipe(
                hand_side="right",
                video_config=wuji.get("video_input"),
                external_frames=True,
            )
            self.retargeter = Retargeter.from_yaml(
                str(wuji["retarget_config"]), hand_side="right"
            )
            self.capture = None
            if not request["external_frames"]:
                self.capture = cv2.VideoCapture(self.camera_device)
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.capture.set(cv2.CAP_PROP_FPS, 30)
                if not self.capture.isOpened():
                    raise RuntimeError(
                        f"failed to open retarget USB camera: {self.camera_device}"
                    )
        self.gate = LandmarkGate(min_confidence=0.7)
        self.memory = None
        shape = request.get("shared_shape")
        if request.get("shared_memory"):
            self.memory = shared_memory.SharedMemory(name=request["shared_memory"])
            resource_tracker.unregister(self.memory._name, "shared_memory")
            self.shared_frame = np.ndarray(tuple(shape), dtype=np.uint8, buffer=self.memory.buf)

    @property
    def joint_limits(self) -> list[list[float]]:
        return np.asarray(self.retargeter.optimizer.robot.joint_limits, dtype=float).tolist()

    def process_frame(self) -> None:
        if self.memory is None:
            raise RuntimeError("external frame shared memory is not configured")
        with contextlib.redirect_stdout(sys.stderr):
            self.detector.process_bgr_frame(np.asarray(self.shared_frame).copy())

    def target(self, now: float) -> list[float] | None:
        if self.capture is not None:
            ok, frame = self.capture.read()
            if not ok:
                raise RuntimeError("failed to read retarget USB camera frame")
            with contextlib.redirect_stdout(sys.stderr):
                self.detector.process_bgr_frame(frame)
        detected_at = self.detector.get_detection_time()
        if detected_at is None or now - detected_at > LANDMARK_STALE_S:
            self.gate.reset()
            return None
        points = self.detector.get_fingers_data()["right_fingers"]
        if not np.any(points):
            return None
        accepted = self.gate.filter(points, timestamp=now, confidence=1.0)
        if accepted is None:
            return None
        with contextlib.redirect_stdout(sys.stderr):
            target = np.asarray(self.retargeter.retarget(accepted), dtype=float).reshape(-1)
        if target.ndim != 1 or not np.isfinite(target).all():
            raise RuntimeError("retargeter returned an invalid target")
        return target.tolist()

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()
        with contextlib.redirect_stdout(sys.stderr):
            self.detector.cleanup()
        if self.memory is not None:
            self.memory.close()


def _reply(payload: dict) -> None:
    print("SLAI_RETARGET_RPC " + json.dumps(payload, separators=(",", ":")), flush=True)


def main() -> None:
    provider = None
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                operation = request["op"]
                if operation == "init":
                    provider = LocalRetargetProvider(request)
                    result = {
                        "joint_limits": provider.joint_limits,
                        "camera_serial": provider.camera_serial,
                        "camera_device": provider.camera_device,
                    }
                elif provider is None:
                    raise RuntimeError("worker is not initialized")
                elif operation == "process_frame":
                    provider.process_frame()
                    result = None
                elif operation == "target":
                    result = provider.target(float(request.get("now", time.monotonic())))
                elif operation == "close":
                    provider.close()
                    _reply({"ok": True, "result": None})
                    return
                else:
                    raise ValueError(f"unknown operation: {operation}")
                _reply({"ok": True, "result": result})
            except BaseException as exc:  # noqa: BLE001 - RPC fault transfer
                _reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    finally:
        if provider is not None:
            with contextlib.suppress(Exception):
                provider.close()


if __name__ == "__main__":
    main()
