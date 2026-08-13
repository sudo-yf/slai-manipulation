"""Persisted dual-camera extrinsics and robust point-pair fitting."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .geometry import estimate_rigid_transform, validate_transform


@dataclass(frozen=True)
class DualCameraExtrinsics:
    primary_serial: str
    secondary_serial: str
    primary_from_secondary: np.ndarray
    rms_error_m: float
    inlier_count: int
    sample_count: int

    def __post_init__(self) -> None:
        if not self.primary_serial or not self.secondary_serial or self.primary_serial == self.secondary_serial:
            raise ValueError("camera serials must be distinct")
        object.__setattr__(self, "primary_from_secondary", validate_transform(self.primary_from_secondary))
        if self.rms_error_m < 0 or self.inlier_count < 3 or self.sample_count < self.inlier_count:
            raise ValueError("invalid calibration metrics")

    def save(self, path: Path) -> None:
        matrix = self.primary_from_secondary
        payload = {"format_version": 1, "primary_serial": self.primary_serial, "secondary_serial": self.secondary_serial, "translation_m": matrix[:3, 3].tolist(), "rotation_matrix": matrix[:3, :3].tolist(), "rms_error_m": self.rms_error_m, "inlier_count": self.inlier_count, "sample_count": self.sample_count}
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_dual_camera_extrinsics(path: Path) -> DualCameraExtrinsics:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if data.get("format_version") != 1:
            raise ValueError("format_version must be 1")
        transform = np.eye(4)
        transform[:3, 3], transform[:3, :3] = data["translation_m"], data["rotation_matrix"]
        return DualCameraExtrinsics(data["primary_serial"], data["secondary_serial"], transform, float(data["rms_error_m"]), int(data["inlier_count"]), int(data["sample_count"]))
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"invalid dual-camera extrinsics {path}: {exc}") from exc


def robust_extrinsic_fit(secondary_points: np.ndarray, primary_points: np.ndarray, *, retain_fraction: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    source, target = np.asarray(secondary_points, dtype=float), np.asarray(primary_points, dtype=float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 12 or not 0.5 <= retain_fraction <= 1:
        raise ValueError("calibration requires at least 12 paired points")
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    mask = finite.copy()
    for _ in range(4):
        transform, _ = estimate_rigid_transform(source[mask], target[mask])
        residuals = np.linalg.norm(source @ transform[:3, :3].T + transform[:3, 3] - target, axis=1)
        mask = finite & (residuals <= max(0.008, np.quantile(residuals[finite], retain_fraction)))
    transform, _ = estimate_rigid_transform(source[mask], target[mask])
    residuals = np.linalg.norm(source @ transform[:3, :3].T + transform[:3, 3] - target, axis=1)
    residuals[~finite] = np.nan
    return transform, residuals
