"""TCP client for newline-delimited iPhone pose packets."""

from __future__ import annotations

import socket
import time
from types import TracebackType

from .protocol import IPhonePose, parse_pose_line

MAX_PACKET_BYTES = 16_384


class IPhonePoseClient:
    def __init__(
        self,
        host: str,
        port: int = 5005,
        *,
        connect_timeout_s: float = 20.0,
        read_timeout_s: float = 2.0,
        socket_factory=socket.create_connection,
    ):
        if not host or not 1 <= port <= 65535 or min(connect_timeout_s, read_timeout_s) <= 0:
            raise ValueError("invalid iPhone pose connection settings")
        self.host, self.port = host, port
        self.connect_timeout_s, self.read_timeout_s = connect_timeout_s, read_timeout_s
        self._socket_factory = socket_factory
        self._socket = None
        self._reader = None

    def connect(self) -> None:
        if self._socket is not None:
            raise RuntimeError("iPhone pose client is already connected")
        deadline, last_error = time.monotonic() + self.connect_timeout_s, None
        while time.monotonic() < deadline:
            try:
                connection = self._socket_factory(
                    (self.host, self.port), timeout=min(1.0, self.read_timeout_s)
                )
                connection.settimeout(self.read_timeout_s)
                self._socket, self._reader = connection, connection.makefile("rb")
                return
            except OSError as exc:
                last_error = exc
                time.sleep(0.2)
        raise ConnectionError(f"could not connect to iPhone pose stream: {last_error}")

    def receive(self) -> IPhonePose:
        if self._reader is None:
            raise RuntimeError("iPhone pose client is not connected")
        line = self._reader.readline(MAX_PACKET_BYTES + 1)
        if not line:
            raise ConnectionError("iPhone pose stream closed")
        if len(line) > MAX_PACKET_BYTES or not line.endswith(b"\n"):
            raise ValueError("iPhone pose packet exceeded the size limit")
        return parse_pose_line(line)

    def close(self) -> None:
        if self._reader is not None:
            self._reader.close()
        if self._socket is not None:
            self._socket.close()
        self._reader = self._socket = None

    def __enter__(self):
        self.connect()
        return self

    def __exit__(
        self,
        _type: type[BaseException] | None,
        _exc: BaseException | None,
        _traceback: TracebackType | None,
    ) -> None:
        self.close()
