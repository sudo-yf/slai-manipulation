from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

ACTIVE_JOINT_NAMES = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
    "wrist_fe_joint",
    "wrist_ru_joint",
)
TASK_HOME_SEED = np.array([-0.173, -1.339, 1.113, -1.981, -1.468, 1.432, 0.0, 0.0])
MAX_IK_ITERATIONS = 20
IK_DAMPING = 1e-4
IK_STEP_GAIN = 0.75
DEFAULT_POSITION_SCALE = 0.5
MAX_LINEAR_SPEED_M_S = 1.0
MAX_ANGULAR_SPEED_RAD_S = 4.0
MIN_ALIGNMENT_MOVE_M = 0.06
MIN_ALIGNMENT_ANGLE_DEG = 35.0
MAX_ALIGNMENT_ANGLE_DEG = 145.0
LOCKED_HAND_JOINTS = {
    "right_finger1_joint1": 0.38846272859564046,
    "right_finger1_joint2": 0.09036175217747225,
    "right_finger1_joint3": -0.018066956363950067,
    "right_finger1_joint4": -0.024368862203700836,
    "right_finger2_joint1": 0.2183019960217428,
    "right_finger2_joint2": -0.20780269147867644,
    "right_finger2_joint3": -0.029969872288003695,
    "right_finger2_joint4": -0.021127784752592152,
    "right_finger3_joint1": 0.22308222713978212,
    "right_finger3_joint2": -0.20578086847211993,
    "right_finger3_joint3": -0.02869209526858126,
    "right_finger3_joint4": -0.02812458838366304,
    "right_finger4_joint1": 0.22015227140257224,
    "right_finger4_joint2": -0.20825366816101828,
    "right_finger4_joint3": -0.02664804468471802,
    "right_finger4_joint4": -0.027316279331113004,
    "right_finger5_joint1": 0.22222357615026492,
    "right_finger5_joint2": -0.20411234603354328,
    "right_finger5_joint3": -0.02340399750990704,
    "right_finger5_joint4": -0.01623599150177595,
}


@dataclass
class IKSessionState:
    template: np.ndarray
    reference_translation: np.ndarray
    reference_rotation: np.ndarray
    last_solution: np.ndarray
    last_target_translation: np.ndarray
    last_target_rotation: np.ndarray
    last_solve_at: float
    reference_source: str
    robot_from_operator: np.ndarray
    alignment_source: str


class RobotIK:
    def __init__(self, urdf_path: Path) -> None:
        try:
            import pinocchio as pin
        except ImportError as exc:
            raise RuntimeError("Pinocchio is unavailable") from exc
        if not urdf_path.is_file():
            raise FileNotFoundError(f"URDF not found: {urdf_path}")
        self.pin: Any = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path.resolve()))
        self.data = self.model.createData()
        self.base_frame_id = self._frame_id("base_link")
        self.target_frame_id = self._frame_id("right_palm_link")
        self.active_indexes = np.array(
            [self.model.joints[self.model.getJointId(name)].idx_q for name in ACTIVE_JOINT_NAMES]
        )
        self.active_velocity_indexes = np.array(
            [self.model.joints[self.model.getJointId(name)].idx_v for name in ACTIVE_JOINT_NAMES]
        )
        self.lower = self.model.lowerPositionLimit[self.active_indexes]
        self.upper = self.model.upperPositionLimit[self.active_indexes]
        self.q_template = pin.neutral(self.model)
        for name, value in LOCKED_HAND_JOINTS.items():
            self.q_template[self.model.joints[self.model.getJointId(name)].idx_q] = value
        self.movable_joint_names = tuple(
            self.model.names[joint_id]
            for joint_id in range(1, self.model.njoints)
            if self.model.joints[joint_id].nq == 1
        )
        if len(self.movable_joint_names) != 28:
            raise ValueError(f"expected 28 movable joints, found {len(self.movable_joint_names)}")
        self.default_solution = np.clip(TASK_HOME_SEED, self.lower, self.upper)
        self.sessions: dict[str, IKSessionState] = {}
        self.lock = threading.Lock()

    def _frame_id(self, name: str) -> int:
        frame_id = int(self.model.getFrameId(name))
        if frame_id >= self.model.nframes:
            raise ValueError(f"URDF does not contain frame {name!r}")
        return frame_id

    def _full_configuration(
        self, active_joints: np.ndarray, template: np.ndarray | None = None
    ) -> np.ndarray:
        q = self.q_template.copy() if template is None else template.copy()
        q[self.active_indexes] = active_joints
        return q

    def _base_from_target(
        self,
        active_joints: np.ndarray,
        template: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        q = self._full_configuration(active_joints, template)
        self.pin.framesForwardKinematics(self.model, self.data, q)
        world_from_base = self.data.oMf[self.base_frame_id]
        world_from_target = self.data.oMf[self.target_frame_id]
        base_from_target = world_from_base.inverse() * world_from_target
        return np.asarray(base_from_target.translation), np.asarray(base_from_target.rotation)

    def _placement(self, active_joints: np.ndarray, template: np.ndarray) -> tuple[np.ndarray, Any]:
        q = self._full_configuration(active_joints, template)
        self.pin.framesForwardKinematics(self.model, self.data, q)
        world_from_base = self.data.oMf[self.base_frame_id]
        world_from_target = self.data.oMf[self.target_frame_id]
        return q, world_from_base.inverse() * world_from_target

    @staticmethod
    def _relative_transform(values: list[float]) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.shape != (16,) or not np.isfinite(matrix).all():
            raise ValueError("relative transform must contain 16 finite values")
        transform = matrix.reshape(4, 4)
        if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-5):
            raise ValueError("relative transform has an invalid last row")
        rotation = transform[:3, :3]
        if (
            not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3)
            or np.linalg.det(rotation) < 0.99
        ):
            raise ValueError("relative transform rotation is not rigid")
        return transform

    @staticmethod
    def _rotation_matrix(values: list[float]) -> np.ndarray:
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.shape == (9,):
            matrix = matrix.reshape(3, 3)
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("robot alignment must contain 9 finite values")
        if not np.allclose(matrix.T @ matrix, np.eye(3), atol=2e-3) or np.linalg.det(matrix) < 0.99:
            raise ValueError("robot alignment is not a right-handed rotation")
        return matrix

    @staticmethod
    def _point(values: list[float], label: str) -> np.ndarray:
        point = np.asarray(values, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError(f"{label} must contain 3 finite values")
        return point

    def _default_state(self, source: str = "slai-4090-task-home-fallback") -> IKSessionState:
        template = self.q_template.copy()
        reference = self.default_solution.copy()
        translation, rotation = self._base_from_target(reference, template)
        return IKSessionState(
            template=template,
            reference_translation=translation.copy(),
            reference_rotation=rotation.copy(),
            last_solution=reference,
            last_target_translation=translation.copy(),
            last_target_rotation=rotation.copy(),
            last_solve_at=0.0,
            reference_source=source,
            robot_from_operator=np.eye(3),
            alignment_source="identity-uncalibrated",
        )

    def _configuration_from_joint_state(
        self,
        joint_names: list[str],
        joint_positions_rad: list[float],
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(joint_names) != len(joint_positions_rad) or not joint_names:
            raise ValueError("robot joint names and positions must have the same non-zero length")
        if len(set(joint_names)) != len(joint_names):
            raise ValueError("robot joint names must be unique")
        positions = np.asarray(joint_positions_rad, dtype=np.float64)
        if not np.isfinite(positions).all():
            raise ValueError("robot joint positions must be finite")
        template = self.q_template.copy()
        known = set(self.movable_joint_names)
        for name, value in zip(joint_names, positions):
            if name not in known:
                raise ValueError(f"unknown robot joint {name!r}")
            template[self.model.joints[self.model.getJointId(name)].idx_q] = value
        missing_active = [name for name in ACTIVE_JOINT_NAMES if name not in joint_names]
        if missing_active:
            raise ValueError(f"robot state is missing active joints: {', '.join(missing_active)}")
        active = np.clip(template[self.active_indexes], self.lower, self.upper)
        template[self.active_indexes] = active
        return template, active

    def bind_reference(
        self,
        session_id: str,
        joint_names: list[str] | None = None,
        joint_positions_rad: list[float] | None = None,
    ) -> dict[str, object]:
        with self.lock:
            previous = self.sessions.get(session_id)
            alignment = previous.robot_from_operator.copy() if previous is not None else np.eye(3)
            alignment_source = (
                previous.alignment_source if previous is not None else "identity-uncalibrated"
            )
            if joint_names is not None or joint_positions_rad is not None:
                if joint_names is None or joint_positions_rad is None:
                    raise ValueError("robot joint names and positions must be supplied together")
                template, reference = self._configuration_from_joint_state(
                    joint_names, joint_positions_rad
                )
                source = "robot-current-joints"
            elif previous is not None:
                template = previous.template.copy()
                reference = previous.last_solution.copy()
                source = "solver-last-solution"
            else:
                state = self._default_state()
                template = state.template
                reference = state.last_solution
                source = state.reference_source

            translation, rotation = self._base_from_target(reference, template)
            self.sessions[session_id] = IKSessionState(
                template=template,
                reference_translation=translation.copy(),
                reference_rotation=rotation.copy(),
                last_solution=reference.copy(),
                last_target_translation=translation.copy(),
                last_target_rotation=rotation.copy(),
                last_solve_at=time.perf_counter(),
                reference_source=source,
                robot_from_operator=alignment,
                alignment_source=alignment_source,
            )
            return {
                "reference": source,
                "reference_translation_m": translation.tolist(),
                "joint_names": list(self.movable_joint_names),
                "joint_positions_rad": self._joint_positions(reference, template),
                "robot_from_operator": alignment.reshape(-1).tolist(),
                "alignment_source": alignment_source,
            }

    def robot_pose(
        self, joint_names: list[str], joint_positions_rad: list[float]
    ) -> dict[str, object]:
        with self.lock:
            template, active = self._configuration_from_joint_state(
                joint_names, joint_positions_rad
            )
            translation, rotation = self._base_from_target(active, template)
        transform = np.eye(4)
        transform[:3, :3] = rotation
        transform[:3, 3] = translation
        return {
            "translation_m": translation.tolist(),
            "rotation": rotation.reshape(-1).tolist(),
            "base_from_palm": transform.reshape(-1).tolist(),
        }

    def set_alignment(
        self,
        session_id: str,
        robot_from_operator: list[float],
        source: str = "restored-calibration",
    ) -> dict[str, object]:
        alignment = self._rotation_matrix(robot_from_operator)
        with self.lock:
            state = self.sessions.get(session_id)
            if state is None:
                state = self._default_state()
                self.sessions[session_id] = state
            state.robot_from_operator = alignment.copy()
            state.alignment_source = source
        return self.get_alignment(session_id)

    def get_alignment(self, session_id: str) -> dict[str, object]:
        with self.lock:
            state = self.sessions.get(session_id)
            if state is None:
                state = self._default_state()
                self.sessions[session_id] = state
            return {
                "robot_from_operator": state.robot_from_operator.reshape(-1).tolist(),
                "alignment_source": state.alignment_source,
                "calibrated": state.alignment_source != "identity-uncalibrated",
            }

    def calibrate_alignment(
        self,
        session_id: str,
        origin: list[float],
        right_point: list[float],
        forward_point: list[float],
    ) -> dict[str, object]:
        origin_value = self._point(origin, "robot origin")
        right = self._point(right_point, "robot right point") - origin_value
        forward = self._point(forward_point, "robot forward point") - origin_value
        right_distance = float(np.linalg.norm(right))
        forward_distance = float(np.linalg.norm(forward))
        if min(right_distance, forward_distance) < MIN_ALIGNMENT_MOVE_M:
            raise ValueError(
                f"move the robot palm at least {MIN_ALIGNMENT_MOVE_M * 100:.0f} cm for each axis"
            )
        x_axis = right / right_distance
        forward_unit = forward / forward_distance
        angle_deg = math.degrees(math.acos(float(np.clip(x_axis @ forward_unit, -1.0, 1.0))))
        if not MIN_ALIGNMENT_ANGLE_DEG <= angle_deg <= MAX_ALIGNMENT_ANGLE_DEG:
            raise ValueError("robot right and forward samples must point in different directions")
        y_axis_raw = forward - x_axis * float(x_axis @ forward)
        y_axis = y_axis_raw / np.linalg.norm(y_axis_raw)
        z_axis = np.cross(x_axis, y_axis)
        z_axis /= np.linalg.norm(z_axis)
        alignment = np.column_stack((x_axis, y_axis, z_axis))
        result = self.set_alignment(
            session_id, alignment.reshape(-1).tolist(), "measured-palm-motion"
        )
        return {
            **result,
            "right_distance_m": right_distance,
            "forward_distance_m": forward_distance,
            "angle_deg": angle_deg,
        }

    def _joint_positions(self, active_joints: np.ndarray, template: np.ndarray) -> list[float]:
        full = self._full_configuration(active_joints, template)
        return [
            float(full[self.model.joints[self.model.getJointId(name)].idx_q])
            for name in self.movable_joint_names
        ]

    def solve_relative(
        self,
        session_id: str,
        values: list[float],
        position_scale: float = DEFAULT_POSITION_SCALE,
    ) -> dict[str, object]:
        started = time.perf_counter()
        relative = self._relative_transform(values)
        if not math.isfinite(position_scale) or not 0.05 <= position_scale <= 2.0:
            raise ValueError("position scale must be between 0.05 and 2.0")

        with self.lock:
            state = self.sessions.get(session_id)
            if state is None:
                state = self._default_state()
                self.sessions[session_id] = state
            aligned_translation = state.robot_from_operator @ relative[:3, 3]
            aligned_rotation = (
                state.robot_from_operator @ relative[:3, :3] @ state.robot_from_operator.T
            )
            desired_translation = state.reference_translation + np.clip(
                aligned_translation * position_scale,
                -0.25,
                0.25,
            )
            desired_rotation = aligned_rotation @ state.reference_rotation
            now = time.perf_counter()
            if state.last_solve_at:
                dt = min(0.1, max(1.0 / 120.0, now - state.last_solve_at))
                translation_step = desired_translation - state.last_target_translation
                translation_distance = float(np.linalg.norm(translation_step))
                max_translation_step = MAX_LINEAR_SPEED_M_S * dt
                if translation_distance > max_translation_step:
                    translation_step *= max_translation_step / translation_distance
                target_translation = state.last_target_translation + translation_step

                rotation_step = self.pin.log3(state.last_target_rotation.T @ desired_rotation)
                rotation_distance = float(np.linalg.norm(rotation_step))
                max_rotation_step = MAX_ANGULAR_SPEED_RAD_S * dt
                if rotation_distance > max_rotation_step:
                    rotation_step *= max_rotation_step / rotation_distance
                target_rotation = state.last_target_rotation @ self.pin.exp3(rotation_step)
            else:
                target_translation = desired_translation
                target_rotation = desired_rotation
            target = self.pin.SE3(target_rotation, target_translation)

            reference = state.last_solution.copy()
            solved = reference.copy()
            evaluations = 0
            for evaluations in range(1, MAX_IK_ITERATIONS + 1):
                q, current = self._placement(solved, state.template)
                position_error = float(np.linalg.norm(current.translation - target_translation))
                angle_error = float(
                    np.linalg.norm(self.pin.log3(target_rotation.T @ current.rotation))
                )
                if position_error <= 0.004 and angle_error <= math.radians(2.0):
                    break
                error = self.pin.log6(current.inverse() * target).vector
                jacobian = self.pin.computeFrameJacobian(
                    self.model,
                    self.data,
                    q,
                    self.target_frame_id,
                    self.pin.ReferenceFrame.LOCAL,
                )[:, self.active_velocity_indexes]
                step = jacobian.T @ np.linalg.solve(
                    jacobian @ jacobian.T + IK_DAMPING * np.eye(6),
                    error,
                )
                solved = np.clip(solved + IK_STEP_GAIN * step, self.lower, self.upper)

            _, final = self._placement(solved, state.template)
            position_error = float(np.linalg.norm(final.translation - target_translation))
            angle_error = float(np.linalg.norm(self.pin.log3(target_rotation.T @ final.rotation)))
            converged = bool(position_error <= 0.004 and angle_error <= math.radians(2.0))
            if converged:
                state.last_solution = solved.copy()
                state.last_target_translation = target_translation.copy()
                state.last_target_rotation = target_rotation.copy()
                state.last_solve_at = now
            joint_positions = self._joint_positions(solved, state.template)

        return {
            "joint_names": list(self.movable_joint_names),
            "joint_positions_rad": joint_positions,
            "active_joint_names": list(ACTIVE_JOINT_NAMES),
            "converged": converged,
            "position_error_m": position_error,
            "rotation_error_rad": angle_error,
            "evaluations": evaluations,
            "solve_ms": (time.perf_counter() - started) * 1000.0,
            "reference": state.reference_source,
            "reference_translation_m": state.reference_translation.tolist(),
            "target_translation_m": target_translation.tolist(),
            "position_scale": position_scale,
            "robot_from_operator": state.robot_from_operator.reshape(-1).tolist(),
            "alignment_source": state.alignment_source,
        }
