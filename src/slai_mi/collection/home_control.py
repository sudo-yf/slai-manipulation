"""Local one-shot home commands shared by collection and hardware runtimes."""

from __future__ import annotations

import socket
from pathlib import Path
from typing import Self

HOME_PACKET = b"slai-manipulation-home-v1"
MAX_PACKET_BYTES = 256


class HomeCommandReceiver:
    """Receive non-blocking home requests on a process-specific Unix socket."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self._socket: socket.socket | None = None

    def start(self) -> None:
        if self._socket is not None:
            raise RuntimeError("home command receiver is already running")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            raise FileExistsError(f"home command socket already exists: {self.path}")
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
        try:
            sock.bind(str(self.path))
            sock.setblocking(False)
        except Exception:
            sock.close()
            raise
        self._socket = sock

    def poll(self) -> bool:
        if self._socket is None:
            raise RuntimeError("home command receiver is not running")
        requested = False
        while True:
            try:
                packet = self._socket.recv(MAX_PACKET_BYTES)
            except BlockingIOError:
                break
            requested |= packet == HOME_PACKET
        return requested

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None
        self.path.unlink(missing_ok=True)

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def request_home(paths: tuple[Path | str, ...]) -> None:
    """Deliver one request to every controller and report all failed targets."""
    failed: list[str] = []
    with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sender:
        for path in paths:
            try:
                sender.sendto(HOME_PACKET, str(path))
            except OSError as exc:
                failed.append(f"{path}: {exc}")
    if failed:
        raise RuntimeError("failed to request coordinated home: " + "; ".join(failed))
