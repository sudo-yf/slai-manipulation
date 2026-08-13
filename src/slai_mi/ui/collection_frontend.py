"""Read-only HTTP dashboard for collection device status."""

from __future__ import annotations

import argparse
import copy
import json
import mimetypes
import threading
from collections.abc import Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from urllib.parse import unquote, urlparse

import yaml

STATIC_ROOT = Path(__file__).with_name("static")
PROJECT_ROOT = Path(__file__).resolve().parents[3]


@runtime_checkable
class StatusProvider(Protocol):
    """Read-only data source consumed by the dashboard.

    Implementations may monitor devices, but this interface deliberately exposes
    no commands that can move a robot or change device configuration.
    """

    def start(self) -> None: ...

    def stop(self) -> None: ...

    def status(self) -> dict[str, Any]: ...

    def camera_jpeg(self, key: str) -> bytes | None: ...


class OfflineStatusProvider:
    """In-memory provider used when live monitoring is not explicitly injected."""

    def __init__(self, status: dict[str, Any]) -> None:
        self._status = copy.deepcopy(status)

    def start(self) -> None:
        return

    def stop(self) -> None:
        return

    def status(self) -> dict[str, Any]:
        return copy.deepcopy(self._status)

    def camera_jpeg(self, key: str) -> bytes | None:
        return None


class DashboardRuntime:
    """Serialize provider access and make its lifecycle idempotent."""

    def __init__(self, provider: StatusProvider | dict[str, Any]) -> None:
        self._provider: StatusProvider = (
            OfflineStatusProvider(provider) if isinstance(provider, dict) else provider
        )
        self._lock = threading.RLock()
        self._started = False

    def start(self) -> None:
        with self._lock:
            if self._started:
                return
            self._provider.start()
            self._started = True

    def stop(self) -> None:
        with self._lock:
            if not self._started:
                return
            try:
                self._provider.stop()
            finally:
                self._started = False

    def status(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._provider.status())

    def camera_jpeg(self, key: str) -> bytes | None:
        with self._lock:
            frame = self._provider.camera_jpeg(key)
            return bytes(frame) if frame is not None else None


def load_hardware_config(path: str | Path) -> dict[str, Any]:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    try:
        payload = yaml.safe_load(candidate.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"Unable to load hardware config {candidate}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"Hardware config must contain a mapping: {candidate}")
    return payload


def offline_status(config: dict[str, Any]) -> dict[str, Any]:
    """Build status without importing or opening any device SDK."""
    camera_config = config.get("cameras", {})
    cameras = []
    for index, item in enumerate(camera_config.get("devices", [])):
        key = str(item.get("key") or item.get("name") or item.get("role") or f"camera_{index + 1}")
        role = str(item.get("role") or item.get("name") or key)
        serial = item.get("serial")
        cameras.append(
            {
                "key": key,
                "serial": str(serial) if serial else None,
                "role": role,
                "model": "Camera",
                "connected": False,
                "fps": 0.0,
                "resolution": [0, 0],
                "error": "设备监测未启用" if camera_config.get("enabled") else "配置中已禁用",
            }
        )
    return {
        "read_only": True,
        "camera_count": len(cameras),
        "camera_online": 0,
        "cameras": cameras,
        "spacemouse": {
            "connected": False,
            "device": "SpaceMouse",
            "motion": [0.0] * 6,
            "buttons": {},
            "active": False,
            "last_activity_ms": None,
            "error": "设备监测未启用"
            if config.get("spacemouse", {}).get("enabled")
            else "配置中已禁用",
        },
    }


def build_handler(
    provider: StatusProvider | DashboardRuntime | dict[str, Any],
) -> type[BaseHTTPRequestHandler]:
    runtime = provider if isinstance(provider, DashboardRuntime) else DashboardRuntime(provider)

    class DashboardHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            route = urlparse(self.path).path
            if route == "/api/status":
                try:
                    self._send_json(runtime.status())
                except Exception as exc:  # noqa: BLE001 - provider failures become HTTP status
                    self._send_json(
                        {"error": f"status unavailable: {type(exc).__name__}"},
                        HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                return
            prefix, suffix = "/api/cameras/", "/frame.jpg"
            if route.startswith(prefix) and route.endswith(suffix):
                key = unquote(route[len(prefix) : -len(suffix)])
                if not key or "/" in key or key in {".", ".."}:
                    self._send_json({"error": "invalid camera key"}, HTTPStatus.BAD_REQUEST)
                    return
                try:
                    frame = runtime.camera_jpeg(key)
                except KeyError:
                    self._send_json({"error": "unknown camera"}, HTTPStatus.NOT_FOUND)
                    return
                except Exception as exc:  # noqa: BLE001 - isolate provider failure from server
                    self._send_json(
                        {"error": f"frame unavailable: {type(exc).__name__}"},
                        HTTPStatus.SERVICE_UNAVAILABLE,
                    )
                    return
                if frame is None:
                    self._send_json({"error": "frame unavailable"}, HTTPStatus.SERVICE_UNAVAILABLE)
                    return
                self._send_bytes(frame, "image/jpeg")
                return
            requested = "index.html" if route == "/" else unquote(route.lstrip("/"))
            target = (STATIC_ROOT / requested).resolve()
            if STATIC_ROOT.resolve() not in target.parents or not target.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            body = target.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_json(self, value: Any, status: HTTPStatus = HTTPStatus.OK) -> None:
            body = json.dumps(value, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _send_bytes(self, body: bytes, content_type: str) -> None:
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store, max-age=0")
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: object) -> None:
            return

    return DashboardHandler


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-config", default="configs/hardware.yaml")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not 0 <= args.port <= 65535:
        raise SystemExit("--port must be between 0 and 65535")
    status = offline_status(load_hardware_config(args.hardware_config))
    runtime = DashboardRuntime(OfflineStatusProvider(status))
    runtime.start()
    server = ThreadingHTTPServer((args.host, args.port), build_handler(runtime))
    server.daemon_threads = True
    address, port = server.server_address[:2]
    print(f"Collection dashboard: http://{address}:{port} (read-only, devices offline)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        runtime.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
