import asyncio
import hashlib
import hmac
import json
import math
import os
import secrets
import sqlite3
import time
from collections import defaultdict
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .ik import RobotIK

PROJECT_ROOT = Path(__file__).resolve().parents[4]
STATE_ROOT = (
    Path(os.getenv("XDG_STATE_HOME", Path.home() / ".local" / "state")) / "slai-mi" / "pose-hub"
)
DATABASE_PATH = Path(os.getenv("POSE_HUB_DATABASE", STATE_ROOT / "pose-hub.sqlite3"))
ADMIN_TOKEN = os.getenv("POSE_HUB_ADMIN_TOKEN", "")
DEFAULT_VIEWER_SESSION = os.getenv("POSE_HUB_DEFAULT_VIEWER_SESSION", "")
DEFAULT_VIEWER_TOKEN = os.getenv("POSE_HUB_DEFAULT_VIEWER_TOKEN", "")
MAX_PACKET_BYTES = 16_384
PERSIST_INTERVAL_SECONDS = 0.1
ROBOT_STATE_MAX_AGE_SECONDS = 0.5
STATIC_ROOT = Path(__file__).with_name("static")
ROBOT_ASSET_PATH = PROJECT_ROOT / "assets" / "robots" / "ur5_wrist_wujihand"
ROBOT_URDF_PATH = ROBOT_ASSET_PATH / "ur5_wrist_wuji_right.urdf"


@dataclass
class SessionCredentials:
    session_id: str
    ingest_token: str
    viewer_token: str
    bridge_token: str


class CreateSessionRequest(BaseModel):
    label: str = Field(default="iPhone", max_length=80)


class SolveIKRequest(BaseModel):
    relative_transform: list[float] = Field(min_length=16, max_length=16)
    position_scale: float = Field(default=0.5, ge=0.05, le=2.0)


class PoseStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.latest: dict[str, dict[str, Any]] = {}
        self.clients: dict[str, set[WebSocket]] = defaultdict(set)
        self.bridge_clients: dict[str, set[WebSocket]] = defaultdict(set)
        self.robot_states: dict[str, dict[str, Any]] = {}
        self.last_persisted: dict[str, float] = {}
        self.lock = asyncio.Lock()

    def connection(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path)
        connection.row_factory = sqlite3.Row
        return connection

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    label TEXT NOT NULL,
                    ingest_hash TEXT NOT NULL,
                    viewer_hash TEXT NOT NULL,
                    bridge_hash TEXT NOT NULL,
                    created_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pose_samples (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    received_at REAL NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS pose_samples_session_received
                    ON pose_samples(session_id, received_at DESC);
                """
            )

    @staticmethod
    def token_hash(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    def create_session(self, label: str) -> SessionCredentials:
        credentials = SessionCredentials(
            session_id=f"p_{secrets.token_urlsafe(9)}",
            ingest_token=secrets.token_urlsafe(32),
            viewer_token=secrets.token_urlsafe(24),
            bridge_token=secrets.token_urlsafe(32),
        )
        with self.connection() as db:
            db.execute(
                "INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
                (
                    credentials.session_id,
                    label,
                    self.token_hash(credentials.ingest_token),
                    self.token_hash(credentials.viewer_token),
                    self.token_hash(credentials.bridge_token),
                    time.time(),
                ),
            )
        return credentials

    def valid_token(self, session_id: str, role: str, token: str) -> bool:
        column = f"{role}_hash"
        if role not in {"ingest", "viewer", "bridge"}:
            return False
        with self.connection() as db:
            row = db.execute(
                f"SELECT {column} FROM sessions WHERE session_id = ?", (session_id,)
            ).fetchone()
        return row is not None and hmac.compare_digest(row[column], self.token_hash(token))

    async def save_pose(self, session_id: str, pose: dict[str, Any]) -> dict[str, Any]:
        received_at = time.time()
        envelope = {"session_id": session_id, "received_at_unix_s": received_at, "pose": pose}
        async with self.lock:
            self.latest[session_id] = envelope
            if received_at - self.last_persisted.get(session_id, 0) >= PERSIST_INTERVAL_SECONDS:
                self.last_persisted[session_id] = received_at
                with self.connection() as db:
                    db.execute(
                        "INSERT INTO pose_samples(session_id, received_at, payload) VALUES (?, ?, ?)",
                        (session_id, received_at, json.dumps(envelope, separators=(",", ":"))),
                    )
                    db.execute(
                        "DELETE FROM pose_samples WHERE received_at < ?", (received_at - 86_400,)
                    )
        return envelope

    def get_latest(self, session_id: str) -> dict[str, Any] | None:
        cached = self.latest.get(session_id)
        if cached is not None:
            return cached
        with self.connection() as db:
            row = db.execute(
                "SELECT payload FROM pose_samples WHERE session_id = ? ORDER BY received_at DESC LIMIT 1",
                (session_id,),
            ).fetchone()
        return json.loads(row["payload"]) if row else None

    def save_robot_state(self, session_id: str, state: dict[str, Any]) -> dict[str, Any]:
        envelope = {
            "session_id": session_id,
            "received_at_unix_s": time.time(),
            **state,
        }
        self.robot_states[session_id] = envelope
        return envelope

    def get_robot_state(
        self, session_id: str, max_age_s: float = ROBOT_STATE_MAX_AGE_SECONDS
    ) -> dict[str, Any] | None:
        state = self.robot_states.get(session_id)
        if state is None or time.time() - state["received_at_unix_s"] > max_age_s:
            return None
        return state


store = PoseStore(DATABASE_PATH)
robot_ik: RobotIK | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    global robot_ik
    if not ADMIN_TOKEN:
        raise RuntimeError("POSE_HUB_ADMIN_TOKEN must be set")
    store.initialize()
    robot_ik = await asyncio.to_thread(RobotIK, ROBOT_URDF_PATH)
    yield


app = FastAPI(title="6D Pose Hub", lifespan=lifespan)
app.mount("/robot-assets", StaticFiles(directory=ROBOT_ASSET_PATH), name="robot-assets")


def require_admin(request: Request) -> None:
    value = request.headers.get("Authorization", "")
    if not value.startswith("Bearer ") or not hmac.compare_digest(value[7:], ADMIN_TOKEN):
        raise HTTPException(status_code=401, detail="admin authentication required")


def bearer_token(request: Request) -> str:
    value = request.headers.get("Authorization", "")
    if not value.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="bearer token required")
    return value[7:]


def parse_pose(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        raise TypeError("pose must be a JSON object")
    matrix = raw.get("world_from_camera")
    if not isinstance(matrix, list) or len(matrix) != 16:
        raise ValueError("world_from_camera must be a 16-value row-major matrix")
    if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in matrix):
        raise ValueError("world_from_camera contains non-finite values")
    tracking = raw.get("tracking")
    if not isinstance(tracking, str) or len(tracking) > 64:
        raise ValueError("tracking must be a short string")
    sequence = raw.get("sequence")
    if not isinstance(sequence, int) or sequence < 0:
        raise ValueError("sequence must be a non-negative integer")
    teleop_enabled = raw.get("teleop_enabled")
    if teleop_enabled is not None and not isinstance(teleop_enabled, bool):
        raise ValueError("teleop_enabled must be a boolean")
    teleop_epoch = raw.get("teleop_epoch")
    if teleop_epoch is not None and (not isinstance(teleop_epoch, int) or teleop_epoch < 0):
        raise ValueError("teleop_epoch must be a non-negative integer")
    return raw


def parse_robot_state(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict) or raw.get("type") != "robot_state":
        raise ValueError("expected a robot_state message")
    names = raw.get("joint_names")
    positions = raw.get("joint_positions_rad")
    if (
        not isinstance(names, list)
        or not names
        or len(names) > 128
        or not all(isinstance(name, str) for name in names)
    ):
        raise ValueError("joint_names must be a non-empty string array")
    if (
        not isinstance(positions, list)
        or len(positions) != len(names)
        or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in positions)
    ):
        raise ValueError("joint_positions_rad must match joint_names and contain finite numbers")
    return {
        "joint_names": names,
        "joint_positions_rad": positions,
        "source_timestamp_s": raw.get("timestamp_s"),
    }


async def broadcast_pose(session_id: str, envelope: dict[str, Any]) -> None:
    message = {"type": "pose", **envelope}
    for clients in (store.clients[session_id], store.bridge_clients[session_id]):
        for client in tuple(clients):
            try:
                await asyncio.wait_for(client.send_json(message), timeout=0.02)
            except Exception:  # noqa: BLE001 - isolate failed WebSocket clients
                clients.discard(client)


async def websocket_auth(websocket: WebSocket, session_id: str, role: str) -> bool:
    await websocket.accept()
    try:
        message = await asyncio.wait_for(websocket_message(websocket), timeout=8)
        payload = json.loads(message)
        token = payload.get("token", "") if payload.get("type") == "auth" else ""
    except (
        TimeoutError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
        WebSocketDisconnect,
    ):
        await websocket.close(code=4401)
        return False
    if not isinstance(token, str) or not store.valid_token(session_id, role, token):
        await websocket.close(code=4401)
        return False
    await websocket.send_json({"type": "authenticated", "session_id": session_id})
    return True


async def websocket_message(websocket: WebSocket) -> str:
    message = await websocket.receive()
    if message["type"] == "websocket.disconnect":
        raise WebSocketDisconnect(message.get("code", 1000))
    if text := message.get("text"):
        return text
    if data := message.get("bytes"):
        return data.decode("utf-8")
    raise ValueError("websocket message has no payload")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", include_in_schema=False)
def root_viewer() -> RedirectResponse:
    if not DEFAULT_VIEWER_SESSION or not DEFAULT_VIEWER_TOKEN:
        raise HTTPException(status_code=404, detail="no default viewer configured")
    return RedirectResponse(url=f"/view/{DEFAULT_VIEWER_SESSION}#{DEFAULT_VIEWER_TOKEN}")


@app.post("/api/v1/sessions", dependencies=[Depends(require_admin)])
def create_session(request: CreateSessionRequest) -> dict[str, str]:
    credentials = store.create_session(request.label)
    return {
        "session_id": credentials.session_id,
        "ingest_token": credentials.ingest_token,
        "viewer_url": f"/view/{credentials.session_id}#{credentials.viewer_token}",
        "bridge_token": credentials.bridge_token,
    }


@app.get("/api/v1/sessions/{session_id}/latest")
def latest_pose(session_id: str, request: Request) -> dict[str, Any]:
    if not store.valid_token(session_id, "bridge", bearer_token(request)):
        raise HTTPException(status_code=401, detail="bridge authentication failed")
    envelope = store.get_latest(session_id)
    if envelope is None:
        raise HTTPException(status_code=404, detail="no pose received for session")
    envelope["age_s"] = max(0.0, time.time() - envelope["received_at_unix_s"])
    return envelope


@app.post("/api/v1/sessions/{session_id}/ik")
async def solve_ik(session_id: str, request: Request, body: SolveIKRequest) -> dict[str, object]:
    if not store.valid_token(session_id, "viewer", bearer_token(request)):
        raise HTTPException(status_code=401, detail="viewer authentication failed")
    if robot_ik is None:
        raise HTTPException(status_code=503, detail="IK solver is unavailable")
    try:
        return await asyncio.to_thread(
            robot_ik.solve_relative,
            session_id,
            body.relative_transform,
            body.position_scale,
        )
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


@app.get("/view/{session_id}")
def viewer(session_id: str) -> FileResponse:
    return FileResponse(STATIC_ROOT / "viewer.html")


@app.get("/assets/pose-calibration.mjs", include_in_schema=False)
def pose_calibration_asset() -> FileResponse:
    return FileResponse(
        STATIC_ROOT / "pose-calibration.mjs",
        media_type="text/javascript",
    )


@app.get("/assets/pose-filter.mjs", include_in_schema=False)
def pose_filter_asset() -> FileResponse:
    return FileResponse(
        STATIC_ROOT / "pose-filter.mjs",
        media_type="text/javascript",
    )


@app.websocket("/ws/ingest/{session_id}")
async def ingest(websocket: WebSocket, session_id: str) -> None:
    if not await websocket_auth(websocket, session_id, "ingest"):
        return
    try:
        while True:
            message = await websocket_message(websocket)
            if len(message) > MAX_PACKET_BYTES:
                await websocket.send_json({"type": "error", "detail": "packet too large"})
                continue
            try:
                envelope = await store.save_pose(session_id, parse_pose(json.loads(message)))
            except (json.JSONDecodeError, TypeError, ValueError) as error:
                await websocket.send_json({"type": "error", "detail": str(error)})
                continue
            await broadcast_pose(session_id, envelope)
    except WebSocketDisconnect:
        return


@app.websocket("/ws/view/{session_id}")
async def view_stream(websocket: WebSocket, session_id: str) -> None:
    if not await websocket_auth(websocket, session_id, "viewer"):
        return
    store.clients[session_id].add(websocket)
    latest = store.get_latest(session_id)
    if latest is not None:
        await websocket.send_json({"type": "pose", **latest})
    try:
        while True:
            await websocket_message(websocket)
    except WebSocketDisconnect:
        store.clients[session_id].discard(websocket)


@app.websocket("/ws/bridge/{session_id}")
async def bridge_stream(websocket: WebSocket, session_id: str) -> None:
    if not await websocket_auth(websocket, session_id, "bridge"):
        return
    store.bridge_clients[session_id].add(websocket)
    latest = store.get_latest(session_id)
    if latest is not None:
        await websocket.send_json({"type": "pose", **latest})
    try:
        while True:
            try:
                state = parse_robot_state(json.loads(await websocket_message(websocket)))
            except (json.JSONDecodeError, ValueError) as error:
                await websocket.send_json({"type": "error", "detail": str(error)})
                continue
            store.save_robot_state(session_id, state)
            await websocket.send_json({"type": "robot_state_accepted"})
    except WebSocketDisconnect:
        store.bridge_clients[session_id].discard(websocket)


@app.websocket("/ws/ik/{session_id}")
async def ik_stream(websocket: WebSocket, session_id: str) -> None:
    if not await websocket_auth(websocket, session_id, "viewer"):
        return
    try:
        while True:
            payload = json.loads(await websocket_message(websocket))
            message_type = payload.get("type")
            if message_type not in {
                "bind",
                "solve",
                "robot_pose",
                "calibrate_alignment",
                "set_alignment",
                "get_alignment",
            }:
                continue
            if robot_ik is None:
                await websocket.send_json(
                    {"type": "ik_error", "detail": "IK solver is unavailable"}
                )
                continue
            if message_type == "robot_pose":
                state = store.get_robot_state(session_id)
                if state is None:
                    await websocket.send_json(
                        {"type": "ik_error", "detail": "fresh robot joint state is unavailable"}
                    )
                    continue
                try:
                    pose = await asyncio.to_thread(
                        robot_ik.robot_pose,
                        state["joint_names"],
                        state["joint_positions_rad"],
                    )
                except ValueError as error:
                    await websocket.send_json({"type": "ik_error", "detail": str(error)})
                    continue
                await websocket.send_json(
                    {
                        "type": "robot_pose",
                        "request_id": payload.get("request_id"),
                        "robot_state_age_s": max(0.0, time.time() - state["received_at_unix_s"]),
                        **pose,
                    }
                )
                continue
            if message_type == "calibrate_alignment":
                try:
                    alignment = await asyncio.to_thread(
                        robot_ik.calibrate_alignment,
                        session_id,
                        payload.get("origin"),
                        payload.get("right_point"),
                        payload.get("forward_point"),
                    )
                except (TypeError, ValueError) as error:
                    await websocket.send_json({"type": "ik_error", "detail": str(error)})
                    continue
                await websocket.send_json({"type": "alignment", **alignment})
                continue
            if message_type == "set_alignment":
                try:
                    alignment = await asyncio.to_thread(
                        robot_ik.set_alignment,
                        session_id,
                        payload.get("robot_from_operator"),
                    )
                except (TypeError, ValueError) as error:
                    await websocket.send_json({"type": "ik_error", "detail": str(error)})
                    continue
                await websocket.send_json({"type": "alignment", **alignment})
                continue
            if message_type == "get_alignment":
                alignment = await asyncio.to_thread(robot_ik.get_alignment, session_id)
                await websocket.send_json({"type": "alignment", **alignment})
                continue
            if message_type == "bind":
                state = store.get_robot_state(session_id)
                try:
                    if state is None:
                        reference = await asyncio.to_thread(robot_ik.bind_reference, session_id)
                    else:
                        reference = await asyncio.to_thread(
                            robot_ik.bind_reference,
                            session_id,
                            state["joint_names"],
                            state["joint_positions_rad"],
                        )
                except ValueError as error:
                    await websocket.send_json({"type": "ik_error", "detail": str(error)})
                    continue
                await websocket.send_json({"type": "ik_bound", **reference})
                continue
            try:
                solution = await asyncio.to_thread(
                    robot_ik.solve_relative,
                    session_id,
                    payload.get("relative_transform"),
                    payload.get("position_scale", 0.5),
                )
            except (TypeError, ValueError) as error:
                await websocket.send_json({"type": "ik_error", "detail": str(error)})
                continue
            await websocket.send_json(
                {
                    "type": "ik",
                    "request_id": payload.get("request_id"),
                    **solution,
                }
            )
    except (json.JSONDecodeError, WebSocketDisconnect):
        return
