from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from slai_mi.ui.webrtc_preview import H264PreviewPublisher, _EncoderChannel


def test_publisher_builds_low_latency_nvenc_commands() -> None:
    publisher = H264PreviewPublisher(
        ("primary", "wrist"), width=640, height=480, fps=30, bitrate_kbps=900
    )

    command = publisher._channels["primary"].command
    assert "h264_nvenc" in command
    assert command[command.index("-tune") + 1] == "ull"
    assert command[command.index("-bf") + 1] == "0"
    assert command[command.index("-g") + 1] == "15"
    assert command[-1] == "rtsp://127.0.0.1:8554/primary"


def test_publisher_validates_frame_shape_without_starting_ffmpeg() -> None:
    publisher = H264PreviewPublisher(("primary",), width=640, height=480, fps=30)

    with pytest.raises(ValueError, match="expected"):
        publisher.publish("primary", np.zeros((10, 10, 3), dtype=np.uint8))


def test_encoder_queue_keeps_only_latest_frame() -> None:
    channel = _EncoderChannel("primary", ("ffmpeg",))
    first = np.zeros((2, 2, 3), dtype=np.uint8)
    second = np.ones((2, 2, 3), dtype=np.uint8)

    channel.submit(first)
    channel.submit(second)

    assert channel._frames.qsize() == 1
    assert np.array_equal(channel._frames.get_nowait(), second)


def test_channel_stop_terminates_encoder() -> None:
    process = MagicMock()
    process.poll.return_value = None
    process.wait.return_value = 0
    with patch("slai_mi.ui.webrtc_preview.subprocess.Popen", return_value=process):
        channel = _EncoderChannel("primary", ("ffmpeg",))
        channel._start_process()
        channel.stop()

    process.terminate.assert_called()
