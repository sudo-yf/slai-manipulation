"""Offline UR5 inverse kinematics backed by the repository URDF and Pinocchio."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .geometry import vector6

UR5_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)
DEFAULT_SEED = np.array([0.0, -math.pi / 2, math.pi / 2, -math.pi / 2, -math.pi / 2, 0.0])


@dataclass(frozen=True)
class IKResult:
    joint_positions_rad: np.ndarray
    converged: bool
    position_error_m: float
    rotation_error_rad: float
    evaluations: int

    def payload(self) -> dict[str, object]:
        return {
            "joint_names": list(UR5_JOINT_NAMES),
            "joint_positions_rad": self.joint_positions_rad.tolist(),
            "converged": self.converged,
            "position_error_m": self.position_error_m,
            "rotation_error_rad": self.rotation_error_rad,
            "evaluations": self.evaluations,
        }


class UR5InverseKinematics:
    """Solve one six-joint UR5 chain without opening an RTDE connection."""

    def __init__(
        self,
        urdf_path: Path,
        *,
        base_frame: str = "base",
        target_frame: str = "tool0",
    ) -> None:
        try:
            import pinocchio as pin
        except ImportError as exc:  # pragma: no cover - depends on the hardware environment
            raise RuntimeError(
                "Pinocchio is required for offline IK; run this command in the Wuji environment"
            ) from exc
        if not urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        self.pin: Any = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path.resolve()))
        self.data = self.model.createData()
        self.base_frame_id = self._frame_id(base_frame)
        self.target_frame_id = self._frame_id(target_frame)
        self.q_template = pin.neutral(self.model)
        self.q_indexes = np.array(
            [self.model.joints[self.model.getJointId(name)].idx_q for name in UR5_JOINT_NAMES]
        )
        self.lower = self.model.lowerPositionLimit[self.q_indexes]
        self.upper = self.model.upperPositionLimit[self.q_indexes]

    def _frame_id(self, name: str) -> int:
        frame_id = int(self.model.getFrameId(name))
        if frame_id >= self.model.nframes:
            raise ValueError(f"URDF does not contain frame {name!r}")
        return frame_id

    def _base_from_target(self, joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        q = self.q_template.copy()
        q[self.q_indexes] = joints
        self.pin.framesForwardKinematics(self.model, self.data, q)
        world_from_base = self.data.oMf[self.base_frame_id]
        world_from_target = self.data.oMf[self.target_frame_id]
        base_from_target = world_from_base.inverse() * world_from_target
        return np.asarray(base_from_target.translation), np.asarray(base_from_target.rotation)

    def solve(
        self,
        target_ur_pose: np.ndarray,
        *,
        seed: np.ndarray | None = None,
        max_evaluations: int = 300,
        position_tolerance_m: float = 0.002,
        rotation_tolerance_rad: float = math.radians(1.0),
    ) -> IKResult:
        target = vector6(target_ur_pose, "IK target pose")
        if max_evaluations <= 0:
            raise ValueError("maximum IK evaluations must be positive")
        if position_tolerance_m <= 0.0 or rotation_tolerance_rad <= 0.0:
            raise ValueError("IK tolerances must be positive")
        target_translation = target[:3]
        target_rotation = Rotation.from_rotvec(target[3:]).as_matrix()

        def errors(joints: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            translation, rotation = self._base_from_target(joints)
            translation_error = translation - target_translation
            rotation_error = Rotation.from_matrix(target_rotation.T @ rotation).as_rotvec()
            return translation_error, rotation_error

        def residual(joints: np.ndarray) -> np.ndarray:
            translation_error, rotation_error = errors(joints)
            return np.concatenate(
                (
                    translation_error / position_tolerance_m,
                    rotation_error / rotation_tolerance_rad,
                )
            )

        initial = DEFAULT_SEED if seed is None else vector6(seed, "IK seed")
        initial = np.clip(initial, self.lower, self.upper)
        candidates = [initial]
        if seed is None:
            candidates.extend(
                np.clip(candidate, self.lower, self.upper)
                for candidate in (
                    DEFAULT_SEED + np.array([math.pi, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    DEFAULT_SEED + np.array([-math.pi, 0.0, 0.0, 0.0, 0.0, 0.0]),
                    np.array([0.0, -math.pi / 2, -math.pi / 2, math.pi / 2, -math.pi / 2, 0.0]),
                )
            )

        best = None
        for candidate in candidates:
            result = least_squares(
                residual,
                candidate,
                bounds=(self.lower, self.upper),
                max_nfev=max_evaluations,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10,
            )
            score = float(np.linalg.norm(residual(result.x)))
            if best is None or score < best[0]:
                best = (score, result)
        assert best is not None
        result = best[1]
        translation_error, rotation_error = errors(result.x)
        position_error = float(np.linalg.norm(translation_error))
        angle_error = float(np.linalg.norm(rotation_error))
        return IKResult(
            joint_positions_rad=np.asarray(result.x),
            converged=bool(
                result.success
                and position_error <= position_tolerance_m
                and angle_error <= rotation_tolerance_rad
            ),
            position_error_m=position_error,
            rotation_error_rad=angle_error,
            evaluations=int(result.nfev),
        )
