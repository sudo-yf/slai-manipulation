from __future__ import annotations

import unittest
from pathlib import Path

import numpy as np

from slai_mi.ui.pose_hub.ik import ACTIVE_JOINT_NAMES, TASK_HOME_SEED, RobotIK


class RobotIKTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.solver = RobotIK(
            Path(__file__).resolve().parents[1]
            / "assets"
            / "robots"
            / "ur5_wrist_wujihand"
            / "ur5_wrist_wuji_right.urdf"
        )

    def test_identity_matches_home_pose(self) -> None:
        result = self.solver.solve_relative("identity", np.eye(4).reshape(-1).tolist())
        self.assertTrue(result["converged"])
        self.assertEqual(len(result["joint_names"]), 28)
        joints = dict(zip(result["joint_names"], result["joint_positions_rad"]))
        for name, expected in zip(ACTIVE_JOINT_NAMES, TASK_HOME_SEED):
            self.assertAlmostEqual(joints[name], expected, places=6)
        self.assertEqual(result["reference"], "slai-4090-task-home-fallback")
        self.assertLess(result["position_error_m"], 1e-6)

    def test_small_translation_converges(self) -> None:
        target = np.eye(4)
        target[0, 3] = 0.02
        result = self.solver.solve_relative("translation", target.reshape(-1).tolist())
        self.assertTrue(result["converged"])
        self.assertLess(result["position_error_m"], 0.004)

    def test_rejects_non_rigid_transform(self) -> None:
        target = np.eye(4)
        target[0, 0] = 2.0
        with self.assertRaisesRegex(ValueError, "not rigid"):
            self.solver.solve_relative("invalid", target.reshape(-1).tolist())

    def test_bind_uses_current_robot_joints(self) -> None:
        default = self.solver.bind_reference("bind-default")
        positions = list(default["joint_positions_rad"])
        shoulder_index = default["joint_names"].index("shoulder_pan_joint")
        positions[shoulder_index] += 0.1
        bound = self.solver.bind_reference("bound", default["joint_names"], positions)
        result = self.solver.solve_relative("bound", np.eye(4).reshape(-1).tolist())

        self.assertEqual(bound["reference"], "robot-current-joints")
        self.assertEqual(result["reference"], "robot-current-joints")
        self.assertTrue(result["converged"])
        joints = dict(zip(result["joint_names"], result["joint_positions_rad"]))
        self.assertAlmostEqual(joints["shoulder_pan_joint"], positions[shoulder_index], places=6)

    def test_bind_rejects_incomplete_active_state(self) -> None:
        with self.assertRaisesRegex(ValueError, "missing active joints"):
            self.solver.bind_reference("incomplete", ["shoulder_pan_joint"], [0.0])

    def test_robot_alignment_maps_operator_axes_to_base(self) -> None:
        result = self.solver.calibrate_alignment(
            "alignment",
            origin=[0.0, 0.0, 0.0],
            right_point=[0.0, 0.12, 0.0],
            forward_point=[-0.12, 0.0, 0.0],
        )
        alignment = np.asarray(result["robot_from_operator"]).reshape(3, 3)

        np.testing.assert_allclose(alignment @ [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(alignment @ [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0], atol=1e-9)
        np.testing.assert_allclose(alignment @ [0.0, 0.0, 1.0], [0.0, 0.0, 1.0], atol=1e-9)
        self.assertAlmostEqual(result["angle_deg"], 90.0, places=6)

        relative = np.eye(4)
        relative[0, 3] = 0.02
        solved = self.solver.solve_relative("alignment", relative.reshape(-1).tolist())
        reference = np.asarray(solved["reference_translation_m"])
        target = np.asarray(solved["target_translation_m"])
        np.testing.assert_allclose(target - reference, [0.0, 0.01, 0.0], atol=1e-9)

    def test_binding_preserves_robot_alignment(self) -> None:
        self.solver.calibrate_alignment(
            "preserved",
            origin=[0.0, 0.0, 0.0],
            right_point=[0.0, 0.12, 0.0],
            forward_point=[-0.12, 0.0, 0.0],
        )
        before = self.solver.get_alignment("preserved")
        self.solver.bind_reference("preserved")
        after = self.solver.get_alignment("preserved")

        np.testing.assert_allclose(
            after["robot_from_operator"], before["robot_from_operator"], atol=1e-9
        )

    def test_robot_pose_uses_measured_joint_state(self) -> None:
        fixture = self.solver.bind_reference("robot-pose-fixture")
        pose = self.solver.robot_pose(fixture["joint_names"], fixture["joint_positions_rad"])

        np.testing.assert_allclose(
            pose["translation_m"], fixture["reference_translation_m"], atol=1e-9
        )
        self.assertEqual(len(pose["base_from_palm"]), 16)


if __name__ == "__main__":
    unittest.main()
