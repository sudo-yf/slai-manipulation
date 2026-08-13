"""Live-provider contract tests for the read-only collection dashboard."""

from __future__ import annotations

import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

from slai_mi.ui.collection_frontend import DashboardRuntime, build_handler


class FakeStatusProvider:
    def __init__(self) -> None:
        self.starts = 0
        self.stops = 0
        self.status_reads = 0
        self.frame_reads: list[str] = []

    def start(self) -> None:
        self.starts += 1

    def stop(self) -> None:
        self.stops += 1

    def status(self) -> dict[str, object]:
        self.status_reads += 1
        return {
            "read_only": True,
            "camera_count": 1,
            "camera_online": 1,
            "cameras": [
                {
                    "key": "camera-wrist-key",
                    "role": "wrist",
                    "serial": "not-used-as-a-route",
                    "connected": True,
                    "model": "fake",
                    "fps": 30.0,
                    "resolution": [640, 480],
                    "error": None,
                }
            ],
            "spacemouse": {
                "connected": True,
                "device": "fake SpaceMouse",
                "motion": [0.1, 0.0, 0.0, 0.0, 0.0, -0.2],
                "buttons": {"home": True},
                "active": True,
                "last_activity_ms": 2,
                "error": None,
            },
        }

    def camera_jpeg(self, key: str) -> bytes | None:
        self.frame_reads.append(key)
        if key != "camera-wrist-key":
            raise KeyError(key)
        return b"\xff\xd8fake-jpeg\xff\xd9"


def test_runtime_lifecycle_is_idempotent() -> None:
    provider = FakeStatusProvider()
    runtime = DashboardRuntime(provider)
    runtime.start()
    runtime.start()
    runtime.stop()
    runtime.stop()
    assert (provider.starts, provider.stops) == (1, 1)


def test_live_provider_status_and_role_key_camera_endpoint() -> None:
    provider = FakeStatusProvider()
    runtime = DashboardRuntime(provider)
    runtime.start()
    server = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(runtime))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        connection.request("GET", "/api/status")
        response = connection.getresponse()
        status = json.loads(response.read())
        assert response.status == 200
        assert status["spacemouse"]["motion"][0] == 0.1

        connection.request("GET", "/api/cameras/camera-wrist-key/frame.jpg")
        response = connection.getresponse()
        assert response.status == 200
        assert response.getheader("Content-Type") == "image/jpeg"
        assert response.read() == b"\xff\xd8fake-jpeg\xff\xd9"
        assert provider.frame_reads == ["camera-wrist-key"]

        connection.request("GET", "/api/cameras/not-a-serial/frame.jpg")
        response = connection.getresponse()
        assert response.status == 404
        response.read()
    finally:
        server.shutdown()
        server.server_close()
        runtime.stop()
        thread.join(timeout=2)
