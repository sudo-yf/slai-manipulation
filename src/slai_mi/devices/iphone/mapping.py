"""Map relative ARKit motion to bounded robot TCP targets."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial.transform import Rotation

from slai_mi.retargeting.geometry import validate_transform

DEFAULT_BASE_FROM_IPHONE_NEUTRAL = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


@dataclass
class RelativeIPhoneToTcpMapper:
    translation_scale: float = 0.5
    workspace_radius_m: np.ndarray = field(default_factory=lambda: np.full(3, 0.25))
    maximum_rotation_rad: float = math.radians(60)
    base_from_iphone_world_rotation: np.ndarray = field(
        default_factory=lambda: DEFAULT_BASE_FROM_IPHONE_NEUTRAL.copy()
    )

    def __post_init__(self) -> None:
        self.workspace_radius_m = np.asarray(self.workspace_radius_m, dtype=float)
        rotation = np.eye(4)
        rotation[:3, :3] = self.base_from_iphone_world_rotation
        self.base_from_iphone_world_rotation = validate_transform(rotation)[:3, :3]
        if (
            self.translation_scale <= 0
            or self.workspace_radius_m.shape != (3,)
            or np.any(self.workspace_radius_m <= 0)
            or not 0 < self.maximum_rotation_rad <= math.pi
        ):
            raise ValueError("invalid iPhone mapper limits")
        self._iphone_zero: np.ndarray | None = None
        self._tcp_zero: np.ndarray | None = None
        self.saturated_axes = (False, False, False)
        self.rotation_saturated = False

    @property
    def ready(self) -> bool:
        return self._iphone_zero is not None

    def reset(self, iphone_world_from_camera: np.ndarray, base_from_tcp: np.ndarray) -> None:
        self._iphone_zero = validate_transform(iphone_world_from_camera)
        self._tcp_zero = validate_transform(base_from_tcp)

    def target(self, iphone_world_from_camera: np.ndarray) -> np.ndarray:
        if self._iphone_zero is None or self._tcp_zero is None:
            raise RuntimeError("mapper has no neutral pose")
        current = validate_transform(iphone_world_from_camera)
        delta = (
            self.translation_scale
            * self.base_from_iphone_world_rotation
            @ (current[:3, 3] - self._iphone_zero[:3, 3])
        )
        self.saturated_axes = tuple(bool(x) for x in (np.abs(delta) > self.workspace_radius_m))
        relative = current[:3, :3] @ self._iphone_zero[:3, :3].T
        relative = (
            self.base_from_iphone_world_rotation @ relative @ self.base_from_iphone_world_rotation.T
        )
        rotvec = Rotation.from_matrix(relative).as_rotvec()
        angle = np.linalg.norm(rotvec)
        self.rotation_saturated = bool(angle > self.maximum_rotation_rad)
        if self.rotation_saturated:
            rotvec *= self.maximum_rotation_rad / angle
        result = np.eye(4)
        result[:3, 3] = self._tcp_zero[:3, 3] + np.clip(
            delta, -self.workspace_radius_m, self.workspace_radius_m
        )
        result[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix() @ self._tcp_zero[:3, :3]
        return validate_transform(result)


def rate_limit_tcp_target(
    current: np.ndarray,
    desired: np.ndarray,
    dt_s: float,
    *,
    maximum_translation_speed_m_s: float,
    maximum_rotation_speed_rad_s: float,
) -> np.ndarray:
    current, desired = validate_transform(current), validate_transform(desired)
    if dt_s <= 0 or maximum_translation_speed_m_s <= 0 or maximum_rotation_speed_rad_s <= 0:
        raise ValueError("rate limits and time step must be positive")
    translation = desired[:3, 3] - current[:3, 3]
    distance, limit = np.linalg.norm(translation), maximum_translation_speed_m_s * dt_s
    if distance > limit:
        translation *= limit / distance
    rotvec = Rotation.from_matrix(desired[:3, :3] @ current[:3, :3].T).as_rotvec()
    angle, rotation_limit = np.linalg.norm(rotvec), maximum_rotation_speed_rad_s * dt_s
    if angle > rotation_limit:
        rotvec *= rotation_limit / angle
    result = np.eye(4)
    result[:3, 3] = current[:3, 3] + translation
    result[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix() @ current[:3, :3]
    return validate_transform(result)
