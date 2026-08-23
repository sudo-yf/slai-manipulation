"""Open a standalone Wuji MediaPipe viewer and validate all 21 hand landmarks."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from slai_mi.devices.wujihand.retarget_camera import (
    dedicated_retarget_camera,
    require_connected_retarget_camera,
)

from ._common import load_yaml, print_plan, reexec_with_python


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-config", default="configs/hardware.yaml")
    parser.add_argument("--duration", type=float, default=60.0, help="Viewer seconds; 0 is unlimited")
    parser.add_argument("--exit-on-detection", action="store_true")
    parser.add_argument("--run", action="store_true", help="Open the RealSense viewer")
    return parser


def _run(hardware: dict, duration: float, exit_on_detection: bool) -> bool:
    camera_device, camera_serial = dedicated_retarget_camera(hardware)
    require_connected_retarget_camera(camera_device, camera_serial)

    import cv2

    project_root = Path(__file__).resolve().parents[3]
    third_party = project_root / "third_party/wuji-retargeting"
    sys.path.insert(0, str(third_party))
    from example.input_devices.realsense_mediapipe import RealsenseMediaPipe
    from wuji_retargeting import Retargeter

    wuji = hardware["wujihand"]
    detector = RealsenseMediaPipe(
        hand_side="right",
        video_config=wuji.get("video_input"),
        show_video=True,
        external_frames=True,
    )
    capture = cv2.VideoCapture(camera_device)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_FPS, 30)
    if not capture.isOpened():
        detector.cleanup()
        raise RuntimeError(f"failed to open retarget USB camera: {camera_device}")
    try:
        retargeter = Retargeter.from_yaml(str(wuji["retarget_config"]), hand_side="right")
        started = time.monotonic()
        succeeded = False
        last_report = 0.0
        print("请把右手放入窗口；红点编号应完整显示 0-20。按 q 退出。")
        while duration == 0.0 or time.monotonic() - started < duration:
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError("failed to read retarget USB camera frame")
            detector.process_bgr_frame(frame)
            now = time.monotonic()
            detected_at = detector.get_detection_time()
            fresh = detected_at is not None and now - detected_at <= 0.3
            points = np.asarray(detector.get_fingers_data()["right_fingers"], dtype=float)
            if fresh and points.shape == (21, 3) and np.isfinite(points).all():
                joints = np.asarray(retargeter.retarget(points), dtype=float).reshape(-1)
                if joints.shape == (20,) and np.isfinite(joints).all():
                    if not succeeded:
                        print("LANDMARKS=21 RETARGET_DOF=20 PASS", flush=True)
                    succeeded = True
                    if exit_on_detection:
                        break
            elif now - last_report >= 1.0:
                print("等待完整右手 21 个关键点...", flush=True)
                last_report = now
            if detector._stop_event.is_set():
                break
            try:
                if cv2.getWindowProperty("Wuji Hand Retarget", cv2.WND_PROP_VISIBLE) < 1:
                    break
            except cv2.error:
                pass
            time.sleep(0.02)
        return succeeded
    finally:
        capture.release()
        detector.cleanup()


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.duration < 0.0:
        raise SystemExit("--duration must be non-negative")
    hardware = load_yaml(args.hardware_config)
    wuji = hardware.get("wujihand", {})
    try:
        camera_device, camera_serial = dedicated_retarget_camera(hardware)
    except ValueError as exc:
        raise SystemExit(f"Wuji retarget check failed: {exc}") from exc
    print_plan(
        {
            "app": "check_wuji_retarget",
            "mode": "viewer" if args.run else "dry-run",
            "camera": "dedicated_4k_usb",
            "camera_serial": camera_serial,
            "camera_device": camera_device,
            "landmarks": 21,
            "retarget_dof": 20,
            "robot_control": "disabled",
        }
    )
    if not args.run:
        return 0
    result = reexec_with_python(
        str(wuji.get("retarget_python", "")),
        "slai_mi.apps.check_wuji_retarget",
        list(sys.argv[1:] if argv is None else argv),
        "SLAI_WUJI_CHECK_REEXEC",
    )
    if result is not None:
        return result
    try:
        succeeded = _run(hardware, args.duration, args.exit_on_detection)
    except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
        raise SystemExit(f"Wuji retarget check failed: {exc}") from exc
    if not succeeded:
        raise SystemExit("Wuji retarget check ended without detecting all 21 landmarks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
