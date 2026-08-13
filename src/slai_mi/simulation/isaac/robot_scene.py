"""Robot-only Isaac scene implementing the backend-neutral simulation contract."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

# AppLauncher must already be alive before this module is imported.
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul, subtract_frame_transforms

from slai_mi.simulation.runtime import SimulationCommand

from .robot_cfg import UR5_WUJI_RIGHT_CFG


class RobotOnlySceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg(size=(4.0, 4.0))
    )
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2500.0, color=(0.8, 0.8, 0.8)),
    )
    robot = UR5_WUJI_RIGHT_CFG


class IsaacRobotScene:
    def __init__(self, *, device: str = "cuda:0") -> None:
        self.sim = SimulationContext(sim_utils.SimulationCfg(dt=1.0 / 120.0, device=device))
        self.sim.set_camera_view(eye=(2.2, 2.2, 1.8), target=(0.0, 0.0, 0.65))
        self.scene = InteractiveScene(RobotOnlySceneCfg(num_envs=1, env_spacing=2.0))
        self.sim.reset()
        self.robot = self.scene["robot"]
        self.arm = SceneEntityCfg(
            "robot",
            joint_names=[
                "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint", "wrist_[123]_joint"
            ],
            body_names=["right_palm_link"],
        )
        self.arm.resolve(self.scene)
        self.ee_body_id = self.arm.body_ids[0]
        self.ee_jacobian_id = self.ee_body_id - 1 if self.robot.is_fixed_base else self.ee_body_id
        self.controller = DifferentialIKController(
            DifferentialIKControllerCfg(
                command_type="pose", use_relative_mode=False, ik_method="dls"
            ),
            num_envs=1,
            device=self.sim.device,
        )
        self.target_pos = torch.zeros((1, 3), device=self.sim.device)
        self.target_quat = torch.zeros((1, 4), device=self.sim.device)
        self._steps = 0

    def reset(self, *, seed: int) -> None:
        torch.manual_seed(seed)
        joint_pos = self.robot.data.default_joint_pos.clone()
        joint_vel = self.robot.data.default_joint_vel.clone()
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel)
        self.robot.reset()
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(self.sim.get_physics_dt())
        ee_pose = self.robot.data.body_pose_w[:, self.ee_body_id]
        root_pose = self.robot.data.root_pose_w
        pos, quat = subtract_frame_transforms(
            root_pose[:, :3], root_pose[:, 3:7], ee_pose[:, :3], ee_pose[:, 3:7]
        )
        self.target_pos, self.target_quat = pos.clone(), quat.clone()
        self.controller.reset()
        self._steps = 0

    def step(self, command: SimulationCommand) -> Mapping[str, Any]:
        twist = torch.as_tensor(command.twist, dtype=torch.float32, device=self.sim.device)
        dt = self.sim.get_physics_dt()
        self.target_pos[0] += twist[:3] * dt
        rotation = twist[3:] * dt
        delta_quat = quat_from_euler_xyz(rotation[0:1], rotation[1:2], rotation[2:3])
        self.target_quat = quat_mul(self.target_quat, delta_quat)
        self.controller.set_command(torch.cat((self.target_pos, self.target_quat), dim=-1))
        jacobian = self.robot.root_physx_view.get_jacobians()[
            :, self.ee_jacobian_id, :, self.arm.joint_ids
        ]
        ee_pose = self.robot.data.body_pose_w[:, self.ee_body_id]
        root_pose = self.robot.data.root_pose_w
        ee_pos, ee_quat = subtract_frame_transforms(
            root_pose[:, :3], root_pose[:, 3:7], ee_pose[:, :3], ee_pose[:, 3:7]
        )
        joints = self.robot.data.joint_pos[:, self.arm.joint_ids]
        target = self.controller.compute(ee_pos, ee_quat, jacobian, joints)
        self.robot.set_joint_position_target(target, joint_ids=self.arm.joint_ids)
        self.scene.write_data_to_sim()
        self.sim.step()
        self.scene.update(dt)
        self._steps += 1
        return {
            "timestamp": self._steps * dt,
            "observation.state": self.robot.data.joint_pos[0].detach().cpu().numpy(),
            "action.twist": twist.detach().cpu().numpy(),
        }

    def episode_done(self) -> bool:
        return False

    def episode_success(self) -> bool:
        return False


def create_scene(
    *,
    simulation_app: Any,
    task_config: Mapping[str, Any],
    project_root: Path,
) -> IsaacRobotScene:
    del simulation_app, task_config, project_root
    return IsaacRobotScene()
