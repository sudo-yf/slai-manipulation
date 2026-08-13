"""Pose Hub transport adapters for the local teleoperation process."""

from __future__ import annotations

import asyncio
import json
import math
import queue
import socket
import threading
import time
from collections.abc import Sequence
from typing import Any, Self
from urllib.parse import quote, urlsplit, urlunsplit


class RobotStateHandoff:
    """Publish measured robot joints to the local Pose Hub bridge."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5006) -> None:
        if not host or not 1 <= port <= 65535:
            raise ValueError("invalid robot-state handoff address")
        self.address = (host, port)
        self._socket: socket.socket | None = None
        self._lock = threading.Lock()

    def publish(
        self,
        joint_names: Sequence[str],
        joint_positions_rad: Sequence[float],
        *,
        timestamp_s: float | None = None,
    ) -> None:
        names = list(joint_names)
        positions = [float(value) for value in joint_positions_rad]
        if not names or len(names) != len(positions):
            raise ValueError("joint names and positions must have the same non-zero length")
        if not all(isinstance(name, str) and name for name in names):
            raise ValueError("joint names must be non-empty strings")
        if not all(math.isfinite(value) for value in positions):
            raise ValueError("joint positions must be finite radians")
        payload = {
            "type": "robot_state",
            "timestamp_s": time.time() if timestamp_s is None else float(timestamp_s),
            "joint_names": names,
            "joint_positions_rad": positions,
        }
        packet = (json.dumps(payload, separators=(",", ":"), allow_nan=False) + "\n").encode()
        with self._lock:
            for attempt in range(2):
                try:
                    if self._socket is None:
                        self._socket = socket.create_connection(self.address, timeout=1.0)
                    self._socket.sendall(packet)
                    return
                except OSError:
                    self.close()
                    if attempt:
                        raise

    def close(self) -> None:
        if self._socket is not None:
            self._socket.close()
            self._socket = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


class PoseHubBridge:
    """Forward fresh cloud poses locally and return measured robot state."""

    def __init__(
        self,
        url: str,
        session_id: str,
        token: str,
        pose_port: int = 5005,
        robot_state_port: int = 5006,
        max_age: float = 0.25,
    ) -> None:
        if not url or not session_id or not token:
            raise ValueError("Pose Hub URL, session and token are required")
        if not 1 <= pose_port <= 65535 or not 1 <= robot_state_port <= 65535:
            raise ValueError("Pose Hub local ports are invalid")
        if max_age <= 0:
            raise ValueError("Pose Hub maximum pose age must be positive")
        self.url = url.rstrip("/")
        self.session_id = session_id
        self.token = token
        self.pose_port = pose_port
        self.robot_state_port = robot_state_port
        self.max_age = max_age
        self.clients: set[socket.socket] = set()
        self.clients_lock = threading.Lock()
        self.robot_states: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=1)

    def websocket_url(self) -> str:
        parts = urlsplit(self.url)
        scheme = (
            "wss" if parts.scheme == "https" else "ws" if parts.scheme == "http" else parts.scheme
        )
        path = "/".join(
            part for part in (parts.path.strip("/"), "ws", "bridge", quote(self.session_id)) if part
        )
        return urlunsplit((scheme, parts.netloc, f"/{path}", "", ""))

    def accept_pose_clients(self, listener: socket.socket) -> None:
        while True:
            client, address = listener.accept()
            client.setblocking(False)
            with self.clients_lock:
                self.clients.add(client)
            print(f"local pose receiver connected: {address[0]}:{address[1]}", flush=True)

    def accept_robot_states(self, listener: socket.socket) -> None:
        while True:
            client, address = listener.accept()
            print(
                f"local robot state publisher connected: {address[0]}:{address[1]}",
                flush=True,
            )
            threading.Thread(target=self.read_robot_states, args=(client,), daemon=True).start()

    def read_robot_states(self, client: socket.socket) -> None:
        buffer = b""
        try:
            with client:
                while chunk := client.recv(65_536):
                    buffer += chunk
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if not line:
                            continue
                        try:
                            state = json.loads(line)
                        except (UnicodeDecodeError, json.JSONDecodeError):
                            continue
                        if state.get("type") != "robot_state":
                            state["type"] = "robot_state"
                        self.put_latest_robot_state(state)
        except OSError:
            return

    def put_latest_robot_state(self, state: dict[str, Any]) -> None:
        try:
            self.robot_states.put_nowait(state)
        except queue.Full:
            try:
                self.robot_states.get_nowait()
            except queue.Empty:
                pass
            self.robot_states.put_nowait(state)

    def broadcast(self, pose: dict[str, Any]) -> None:
        encoded = (json.dumps(pose, separators=(",", ":")) + "\n").encode()
        with self.clients_lock:
            for client in tuple(self.clients):
                try:
                    client.sendall(encoded)
                except (BlockingIOError, OSError):
                    self.clients.discard(client)
                    client.close()

    def start_local_servers(self) -> None:
        pose_listener = self._listener(self.pose_port)
        state_listener = self._listener(self.robot_state_port)
        threading.Thread(
            target=self.accept_pose_clients,
            args=(pose_listener,),
            daemon=True,
        ).start()
        threading.Thread(
            target=self.accept_robot_states,
            args=(state_listener,),
            daemon=True,
        ).start()
        print(f"serving poses on 127.0.0.1:{self.pose_port}", flush=True)
        print(
            f"accepting robot states on 127.0.0.1:{self.robot_state_port}",
            flush=True,
        )

    @staticmethod
    def _listener(port: int) -> socket.socket:
        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind(("127.0.0.1", port))
        listener.listen()
        return listener

    async def send_robot_states(self, websocket: Any) -> None:
        while True:
            state = await asyncio.to_thread(self.robot_states.get)
            await websocket.send(json.dumps(state, separators=(",", ":")))

    async def connected(self, websockets: Any) -> None:
        async with websockets.connect(
            self.websocket_url(),
            max_queue=1,
            ping_interval=15,
            ping_timeout=10,
            compression=None,
        ) as websocket:
            await websocket.send(
                json.dumps({"type": "auth", "token": self.token}, separators=(",", ":"))
            )
            authentication = json.loads(await asyncio.wait_for(websocket.recv(), timeout=8))
            if authentication.get("type") != "authenticated":
                raise RuntimeError("Pose Hub bridge authentication failed")
            print("Pose Hub WebSocket connected", flush=True)
            state_sender = asyncio.create_task(self.send_robot_states(websocket))
            try:
                async for raw in websocket:
                    message = json.loads(raw)
                    if message.get("type") != "pose":
                        continue
                    pose = message.get("pose", {})
                    age = time.time() - float(message.get("received_at_unix_s", 0))
                    if age <= self.max_age and pose.get("tracking") == "normal":
                        self.broadcast(pose)
            finally:
                state_sender.cancel()

    async def run(self) -> None:
        try:
            import websockets
        except ImportError as exc:
            raise RuntimeError("Pose Hub bridge requires the websockets package") from exc
        self.start_local_servers()
        while True:
            try:
                await self.connected(websockets)
            except (
                OSError,
                RuntimeError,
                ValueError,
                json.JSONDecodeError,
                websockets.WebSocketException,
            ) as error:
                print(f"Pose Hub disconnected: {error}; reconnecting", flush=True)
                await asyncio.sleep(1)
