"""Non-blocking H264 preview publishing for the collection dashboard."""

from __future__ import annotations

import logging
import os
import queue
import subprocess
import threading
import time
from collections.abc import Iterable, Sequence
from contextlib import suppress
from typing import Any

import numpy as np

LOGGER = logging.getLogger(__name__)


class _EncoderChannel:
    """Own one FFmpeg process without ever blocking the capture caller."""

    def __init__(self, role: str, command: Sequence[str]) -> None:
        self.role = role
        self.command = tuple(command)
        self._frames: queue.Queue[np.ndarray] = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._process: subprocess.Popen[bytes] | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"webrtc-preview-{self.role}",
            daemon=True,
        )
        self._thread.start()

    def submit(self, frame: Any) -> None:
        image = np.asarray(frame, dtype=np.uint8)
        if not image.flags.c_contiguous:
            image = np.ascontiguousarray(image)
        try:
            self._frames.put_nowait(image)
            return
        except queue.Full:
            pass
        with suppress(queue.Empty):
            self._frames.get_nowait()
        with suppress(queue.Full):
            self._frames.put_nowait(image)

    def stop(self) -> None:
        self._stop.set()
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None
        self._close_process()

    def _start_process(self) -> subprocess.Popen[bytes]:
        process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._process = process
        return process

    def _close_process(self) -> None:
        process, self._process = self._process, None
        if process is None:
            return
        if process.stdin is not None:
            with suppress(BrokenPipeError, OSError):
                process.stdin.close()
        if process.poll() is None:
            process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)

    def _run(self) -> None:
        retry_after = 0.0
        while not self._stop.is_set():
            try:
                frame = self._frames.get(timeout=0.2)
            except queue.Empty:
                continue
            if time.monotonic() < retry_after:
                continue
            process = self._process
            if process is None or process.poll() is not None:
                self._close_process()
                try:
                    process = self._start_process()
                except OSError as exc:
                    LOGGER.warning("WebRTC preview encoder %s failed to start: %s", self.role, exc)
                    retry_after = time.monotonic() + 2.0
                    continue
            try:
                assert process.stdin is not None
                process.stdin.write(frame.tobytes())
                process.stdin.flush()
            except (BrokenPipeError, OSError):
                self._close_process()
                retry_after = time.monotonic() + 1.0


class H264PreviewPublisher:
    """Publish latest-only RGB frames to MediaMTX RTSP paths."""

    def __init__(
        self,
        roles: Iterable[str],
        *,
        width: int,
        height: int,
        fps: int,
        bitrate_kbps: int = 1000,
        rtsp_base_url: str | None = None,
        ffmpeg: str | None = None,
    ) -> None:
        if min(width, height, fps, bitrate_kbps) <= 0:
            raise ValueError("preview dimensions, FPS and bitrate must be positive")
        base_url = (rtsp_base_url or os.environ.get("SLAI_PREVIEW_RTSP_BASE")) or (
            "rtsp://127.0.0.1:8554"
        )
        executable = ffmpeg or os.environ.get("SLAI_PREVIEW_FFMPEG") or "/usr/bin/ffmpeg"
        self.width = int(width)
        self.height = int(height)
        self._channels = {
            str(role): _EncoderChannel(
                str(role),
                self._command(
                    executable,
                    f"{base_url.rstrip('/')}/{role}",
                    fps=int(fps),
                    bitrate_kbps=int(bitrate_kbps),
                ),
            )
            for role in roles
        }

    def _command(
        self,
        executable: str,
        output_url: str,
        *,
        fps: int,
        bitrate_kbps: int,
    ) -> tuple[str, ...]:
        return (
            executable,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{self.width}x{self.height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "h264_nvenc",
            "-preset",
            "p1",
            "-tune",
            "ull",
            "-profile:v",
            "baseline",
            "-rc",
            "cbr",
            "-b:v",
            f"{bitrate_kbps}k",
            "-maxrate",
            f"{bitrate_kbps}k",
            "-bufsize",
            f"{max(1, bitrate_kbps // 2)}k",
            "-g",
            str(max(1, fps // 2)),
            "-bf",
            "0",
            "-pix_fmt",
            "yuv420p",
            "-f",
            "rtsp",
            "-rtsp_transport",
            "tcp",
            output_url,
        )

    def start(self) -> None:
        for channel in self._channels.values():
            channel.start()

    def publish(self, role: str, frame: Any) -> None:
        try:
            channel = self._channels[role]
        except KeyError as exc:
            raise KeyError(f"unknown preview camera: {role}") from exc
        image = np.asarray(frame)
        if image.shape != (self.height, self.width, 3):
            raise ValueError(
                f"preview frame {role} has shape {image.shape}, "
                f"expected {(self.height, self.width, 3)}"
            )
        channel.submit(image)

    def stop(self) -> None:
        for channel in self._channels.values():
            channel.stop()

