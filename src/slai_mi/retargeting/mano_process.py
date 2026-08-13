"""Length-prefixed subprocess client for an external MANO fitting worker."""

from __future__ import annotations

import pickle
import struct
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

MAX_PACKET_BYTES = 128 * 1024 * 1024


def _read_exact(stream, size: int) -> bytes:
    chunks = []
    while size:
        chunk = stream.read(size)
        if not chunk:
            raise RuntimeError("MANO worker closed its output")
        chunks.append(chunk)
        size -= len(chunk)
    return b"".join(chunks)


def read_packet(stream):
    size = struct.unpack(">I", _read_exact(stream, 4))[0]
    if size > MAX_PACKET_BYTES:
        raise RuntimeError("MANO worker packet exceeds safety limit")
    return pickle.loads(_read_exact(stream, size))


def write_packet(stream, value: object) -> None:
    payload = pickle.dumps(value, protocol=4)
    if len(payload) > MAX_PACKET_BYTES:
        raise ValueError("MANO request packet exceeds safety limit")
    stream.write(struct.pack(">I", len(payload)) + payload)
    stream.flush()


@dataclass(frozen=True)
class ManoFit:
    vertices: np.ndarray
    joints: np.ndarray
    pose_axis_angle: np.ndarray
    betas: np.ndarray
    translation: np.ndarray
    mean_error_mm: float
    maximum_error_mm: float
    fit_ms: float


class ManoFitterProcess:
    """Launch a user-supplied worker; no environment-specific defaults are embedded."""

    def __init__(self, *, python: Path, worker: Path, mano_root: Path, device: str = "cuda:0", cwd: Path | None = None):
        self.python, self.worker, self.mano_root = map(Path, (python, worker, mano_root))
        self.device, self.cwd = device, cwd
        self.process = None
        self.faces = None

    def start(self) -> None:
        if self.process is not None:
            raise RuntimeError("MANO worker is already running")
        missing = [str(path) for path in (self.python, self.worker, self.mano_root) if not Path(path).exists()]
        if missing:
            raise FileNotFoundError(f"missing MANO dependencies: {missing}")
        self.process = subprocess.Popen([str(self.python), str(self.worker), "--mano-root", str(self.mano_root), "--device", self.device], stdin=subprocess.PIPE, stdout=subprocess.PIPE, cwd=self.cwd)
        ready = read_packet(self.process.stdout)
        if not isinstance(ready, dict) or ready.get("type") != "ready":
            self.stop()
            raise RuntimeError(f"MANO worker failed to initialize: {ready}")
        self.faces = np.asarray(ready["faces"], dtype=np.int32)

    def fit(self, keypoints: np.ndarray, weights: np.ndarray) -> ManoFit:
        if self.process is None or self.process.poll() is not None:
            raise RuntimeError("MANO worker is not running")
        points, confidence = np.asarray(keypoints, dtype=np.float32), np.asarray(weights, dtype=np.float32)
        if points.shape != (21, 3) or confidence.shape != (21,) or not np.isfinite(points).all() or not np.isfinite(confidence).all():
            raise ValueError("MANO observations must have shapes (21,3) and (21,)")
        write_packet(self.process.stdin, {"type": "fit", "keypoints": points, "weights": confidence})
        response = read_packet(self.process.stdout)
        if response.get("type") != "fit":
            raise RuntimeError(response.get("message", "invalid MANO response"))
        return ManoFit(*(np.asarray(response[name], dtype=np.float32) for name in ("vertices", "joints", "pose_axis_angle", "betas", "translation")), *(float(response[name]) for name in ("mean_error_mm", "maximum_error_mm", "fit_ms")))

    def stop(self) -> None:
        process, self.process = self.process, None
        if process is None:
            return
        if process.poll() is None:
            try:
                write_packet(process.stdin, {"type": "stop"})
                process.wait(timeout=3)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=2)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
