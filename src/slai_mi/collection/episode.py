"""Episode success/failure state and deterministic augmentation splits."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class EpisodeDecision(str, Enum):
    RECORD = "record"
    SUCCESS = "success"
    DISCARD_TIMEOUT = "discard_timeout"
    DISCARD_INVALID = "discard_invalid"
    DISCARD_DROPPED = "discard_dropped"
    DISCARD_OUT_OF_BOUNDS = "discard_out_of_bounds"


@dataclass(frozen=True)
class SuccessObservation:
    object_front_clearance_m: float
    grasp_distance_m: float
    object_height_above_drawer_floor_m: float
    linear_speed_m_s: float
    angular_speed_rad_s: float
    in_workspace: bool = True
    dropped: bool = False

    def finite(self) -> bool:
        values = (
            self.object_front_clearance_m,
            self.grasp_distance_m,
            self.object_height_above_drawer_floor_m,
            self.linear_speed_m_s,
            self.angular_speed_rad_s,
        )
        return bool(np.isfinite(values).all())


class EpisodeMonitor:
    """Require ten consecutive 15 Hz success frames and reject invalid episodes."""

    def __init__(self, *, hold_frames: int = 10, max_frames: int = 180) -> None:
        if hold_frames <= 0 or max_frames < hold_frames:
            raise ValueError("invalid episode frame limits")
        self.hold_frames = hold_frames
        self.max_frames = max_frames
        self.frame_count = 0
        self.success_count = 0

    def reset(self) -> None:
        self.frame_count = 0
        self.success_count = 0

    def update(self, observation: SuccessObservation) -> EpisodeDecision:
        self.frame_count += 1
        if not observation.finite():
            return EpisodeDecision.DISCARD_INVALID
        if observation.dropped:
            return EpisodeDecision.DISCARD_DROPPED
        if not observation.in_workspace:
            return EpisodeDecision.DISCARD_OUT_OF_BOUNDS

        successful = (
            observation.object_front_clearance_m > 0.0
            and observation.grasp_distance_m < 0.04
            and observation.object_height_above_drawer_floor_m > 0.05
            and observation.linear_speed_m_s < 0.25
            and observation.angular_speed_rad_s < 1.5
        )
        self.success_count = self.success_count + 1 if successful else 0
        if self.success_count >= self.hold_frames:
            return EpisodeDecision.SUCCESS
        if self.frame_count >= self.max_frames:
            return EpisodeDecision.DISCARD_TIMEOUT
        return EpisodeDecision.RECORD


class DatasetSplit(str, Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    ZERO_SHOT_TEST = "zero_shot_test"


@dataclass(frozen=True)
class Randomization:
    seed: int
    split: DatasetSplit
    object_xy: tuple[float, float]
    object_yaw: float
    object_scale: float
    mass_scale: float
    friction_scale: float
    drawer_open_m: float
    cabinet_xy: tuple[float, float]
    cabinet_yaw_rad: float
    camera_position_jitter_m: tuple[float, float, float]
    camera_angle_jitter_rad: tuple[float, float, float]
    fov_scale: float
    light_scale: float


def split_for_index(index: int, total: int = 5000) -> DatasetSplit:
    if total < 3 or not 0 <= index < total:
        raise ValueError("index must be inside a dataset of at least three episodes")
    train_end = round(total * 0.8)
    validation_end = round(total * 0.9)
    if index < train_end:
        return DatasetSplit.TRAIN
    if index < validation_end:
        return DatasetSplit.VALIDATION
    return DatasetSplit.ZERO_SHOT_TEST


def sample_randomization(seed: int, split: DatasetSplit) -> Randomization:
    """Sample train/validation interiors or held-out edge/camera extremes."""
    rng = np.random.default_rng(seed)
    if split is DatasetSplit.ZERO_SHOT_TEST:
        corner = rng.choice((-1.0, 1.0), size=2)
        object_xy = tuple((corner * rng.uniform((0.075, 0.045), (0.095, 0.065))).tolist())
        camera_pos = tuple((rng.choice((-1.0, 1.0), size=3) * 0.02).tolist())
        camera_angle = tuple((rng.choice((-1.0, 1.0), size=3) * np.deg2rad(3.0)).tolist())
        fov_scale = float(rng.choice((0.97, 1.03)))
    else:
        object_xy = tuple(rng.uniform((-0.070, -0.040), (0.070, 0.040)).tolist())
        camera_pos = tuple(rng.uniform(-0.02, 0.02, size=3).tolist())
        camera_angle = tuple(rng.uniform(-np.deg2rad(3), np.deg2rad(3), size=3).tolist())
        fov_scale = float(rng.uniform(0.97, 1.03))
    return Randomization(
        seed=seed,
        split=split,
        object_xy=object_xy,
        object_yaw=float(rng.uniform(-np.pi, np.pi)),
        object_scale=float(rng.uniform(0.90, 1.10)),
        mass_scale=float(rng.uniform(0.80, 1.20)),
        friction_scale=float(rng.uniform(0.80, 1.20)),
        drawer_open_m=float(rng.uniform(0.20, 0.28)),
        cabinet_xy=tuple(rng.uniform(-0.02, 0.02, size=2).tolist()),
        cabinet_yaw_rad=float(rng.uniform(-np.deg2rad(2), np.deg2rad(2))),
        camera_position_jitter_m=camera_pos,
        camera_angle_jitter_rad=camera_angle,
        fov_scale=fov_scale,
        light_scale=float(rng.uniform(0.7, 1.3)),
    )
