"""Tests for the read-only collection dashboard."""

import json
import threading
from http.client import HTTPConnection
from http.server import ThreadingHTTPServer

from slai_mi.ui.collection_frontend import build_handler, offline_status


def test_offline_status_uses_configured_camera_names() -> None:
    status = offline_status(
        {
            "cameras": {"enabled": False, "devices": [{"name": "wrist", "serial": None}]},
            "spacemouse": {"enabled": False},
        }
    )
    assert status["read_only"] is True
    assert status["camera_online"] == 0
    assert status["cameras"][0]["key"] == "wrist"
    assert status["cameras"][0]["connected"] is False


def test_dashboard_serves_status_and_static_page() -> None:
    status = offline_status({"cameras": {"devices": []}, "spacemouse": {}})
    server = ThreadingHTTPServer(("127.0.0.1", 0), build_handler(status))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        connection.request("GET", "/api/status")
        response = connection.getresponse()
        assert response.status == 200
        assert json.loads(response.read())["read_only"] is True

        connection.request("GET", "/")
        response = connection.getresponse()
        assert response.status == 200
        assert b"SLaI" in response.read()
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)
