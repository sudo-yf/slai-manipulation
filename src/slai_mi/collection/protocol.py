"""Loss-intolerant Unix socket protocol between Isaac and LeRobot v2.1."""

from __future__ import annotations

import json
import socket
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Self

import numpy as np

from .schema import ACTION_DIM, IMAGE_HEIGHT, IMAGE_WIDTH, Frame

_HEADER = struct.Struct("!II")
_RGB_BYTES = IMAGE_HEIGHT * IMAGE_WIDTH * 3
_VECTOR_BYTES = ACTION_DIM * np.dtype("<f4").itemsize
FRAME_BYTES = 2 * _RGB_BYTES + 2 * _VECTOR_BYTES


class ProtocolError(RuntimeError):
    pass


def encode_frame(frame: Frame) -> bytes:
    value = frame.validated()
    return b"".join(
        (
            value.far_rgb.tobytes(),
            value.near_rgb.tobytes(),
            value.joint_position.astype("<f4", copy=False).tobytes(),
            value.actions.astype("<f4", copy=False).tobytes(),
        )
    )


def decode_frame(payload: bytes) -> Frame:
    if len(payload) != FRAME_BYTES:
        raise ProtocolError(f"frame payload is {len(payload)} bytes, expected {FRAME_BYTES}")
    cursor = 0
    shape = (IMAGE_HEIGHT, IMAGE_WIDTH, 3)
    far = np.frombuffer(payload, dtype=np.uint8, count=_RGB_BYTES, offset=cursor).reshape(shape)
    cursor += _RGB_BYTES
    near = np.frombuffer(payload, dtype=np.uint8, count=_RGB_BYTES, offset=cursor).reshape(shape)
    cursor += _RGB_BYTES
    state = np.frombuffer(payload, dtype="<f4", count=ACTION_DIM, offset=cursor)
    cursor += _VECTOR_BYTES
    action = np.frombuffer(payload, dtype="<f4", count=ACTION_DIM, offset=cursor)
    return Frame(far.copy(), near.copy(), state.copy(), action.copy()).validated()


def send_message(sock: socket.socket, header: dict, payload: bytes = b"") -> None:
    metadata = json.dumps(header, separators=(",", ":")).encode("utf-8")
    sock.sendall(_HEADER.pack(len(metadata), len(payload)) + metadata + payload)


def recv_message(sock: socket.socket) -> tuple[dict, bytes]:
    packed = _recv_exact(sock, _HEADER.size)
    header_size, payload_size = _HEADER.unpack(packed)
    if header_size > 65536 or payload_size > FRAME_BYTES:
        raise ProtocolError("message exceeds protocol limits")
    try:
        header = json.loads(_recv_exact(sock, header_size).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ProtocolError("invalid JSON header") from exc
    if not isinstance(header, dict):
        raise ProtocolError("header must be a JSON object")
    return header, _recv_exact(sock, payload_size)


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < size:
        chunk = sock.recv(size - len(chunks))
        if not chunk:
            raise ProtocolError("writer connection closed")
        chunks.extend(chunk)
    return bytes(chunks)


@dataclass
class WriterClient:
    socket_path: Path
    timeout_s: float = 5.0

    def __post_init__(self) -> None:
        self._socket: socket.socket | None = None
        self._sequence = 0

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout_s)
        sock.connect(str(self.socket_path))
        self._socket = sock
        self._request("hello", {"protocol": 1})

    def append(self, frame: Frame) -> None:
        self._request("frame", {"sequence": self._sequence}, encode_frame(frame))
        self._sequence += 1

    def save_episode(self) -> None:
        self._request("save", {"frames": self._sequence})
        self._sequence = 0

    def discard_episode(self, reason: str) -> None:
        self._request("discard", {"frames": self._sequence, "reason": reason})
        self._sequence = 0

    def close(self) -> None:
        if self._socket is not None:
            try:
                self._request("close", {})
            finally:
                self._socket.close()
                self._socket = None

    def _request(self, command: str, fields: dict, payload: bytes = b"") -> dict:
        if self._socket is None:
            raise ProtocolError("writer client is not connected")
        send_message(self._socket, {"command": command, **fields}, payload)
        reply, reply_payload = recv_message(self._socket)
        if reply_payload or not reply.get("ok", False):
            raise ProtocolError(str(reply.get("error", "invalid writer acknowledgement")))
        return reply

    def __enter__(self) -> Self:
        self.connect()
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()
