"""Isaac Lab articulation configuration for UR5 CB3 + 2-DOF wrist + Wuji right hand."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ROBOT_URDF = (
    PROJECT_ROOT
    / "assets"
    / "robots"
    / "ur5_wrist_wujihand"
    / "ur5_wrist_wuji_right.urdf"
)

WRIST_VISUAL_COLORS = {
    "wrist_base": (0.55, 0.58, 0.62),
    "fe_yoke": (0.48, 0.20, 0.55),
    "output_flange": (0.05, 0.36, 0.57),
}
RIGHT_HAND_LINKS = ("right_palm_link",) + tuple(
    f"right_finger{finger}_{segment}"
    for finger in range(1, 6)
    for segment in ("link1", "link2", "link3", "link4", "tip_link")
)


def apply_robot_visual_materials(robot_prim_path: str, with_wrist: bool) -> None:
    """Rebind colors that Isaac's STL importer replaces with white."""
    aluminum_material_path = f"{robot_prim_path}/Looks/ur5_brushed_aluminum"
    aluminum_material_cfg = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.48, 0.50, 0.49),
        roughness=0.30,
        metallic=0.72,
    )
    aluminum_material_cfg.func(aluminum_material_path, aluminum_material_cfg)
    for link_name in ("upper_arm_link", "forearm_link"):
        sim_utils.bind_visual_material(
            f"{robot_prim_path}/{link_name}",
            aluminum_material_path,
            stronger_than_descendants=True,
        )

    joint_material_path = f"{robot_prim_path}/Looks/ur5_joint_gray"
    joint_material_cfg = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.34, 0.35, 0.34),
        roughness=0.48,
        metallic=0.18,
    )
    joint_material_cfg.func(joint_material_path, joint_material_cfg)
    for link_name in (
        "base_link",
        "shoulder_link",
        "wrist_1_link",
        "wrist_2_link",
        "wrist_3_link",
    ):
        sim_utils.bind_visual_material(
            f"{robot_prim_path}/{link_name}",
            joint_material_path,
            stronger_than_descendants=True,
        )

    hand_material_path = f"{robot_prim_path}/Looks/right_hand_black"
    hand_material_cfg = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.02, 0.02, 0.02),
        roughness=0.55,
    )
    hand_material_cfg.func(hand_material_path, hand_material_cfg)
    for link_name in ("wuji_50mm_offset", *RIGHT_HAND_LINKS):
        sim_utils.bind_visual_material(
            f"{robot_prim_path}/{link_name}",
            hand_material_path,
            stronger_than_descendants=True,
        )

    if not with_wrist:
        return

    for link_name, color in WRIST_VISUAL_COLORS.items():
        material_path = f"{robot_prim_path}/Looks/{link_name}_material"
        material_cfg = sim_utils.PreviewSurfaceCfg(diffuse_color=color, roughness=0.6)
        material_cfg.func(material_path, material_cfg)
        sim_utils.bind_visual_material(
            f"{robot_prim_path}/{link_name}",
            material_path,
            stronger_than_descendants=True,
        )


UR5_WUJI_RIGHT_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/Robot",
    spawn=sim_utils.UrdfFileCfg(
        asset_path=str(ROBOT_URDF),
        fix_base=True,
        merge_fixed_joints=False,
        make_instanceable=False,
        self_collision=False,
        collision_from_visuals=False,
        collider_type="convex_decomposition",
        replace_cylinders_with_capsules=False,
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=2,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0.0,
                damping=0.0,
            ),
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),
        joint_pos={
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.5708,
            "elbow_joint": 1.5708,
            "wrist_1_joint": -1.5708,
            "wrist_2_joint": -1.5708,
            "wrist_3_joint": 0.0,
            "wrist_fe_joint": 0.0,
            "wrist_ru_joint": 0.0,
            "right_finger1_joint1": 0.390,
            "right_finger1_joint2": 0.097,
            "right_finger1_joint[34]": -0.021,
            "right_finger[2-5]_joint1": 0.220,
            "right_finger[2-5]_joint2": -0.207,
            "right_finger[2-5]_joint[34]": -0.025,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        "ur5_arm": ImplicitActuatorCfg(
            joint_names_expr=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_[123]_joint",
            ],
            effort_limit_sim=150.0,
            velocity_limit_sim=3.15,
            stiffness=400.0,
            damping=40.0,
        ),
        "robot_wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_(fe|ru)_joint"],
            effort_limit_sim=2.0,
            velocity_limit_sim=2.0,
            stiffness=25.0,
            damping=1.0,
        ),
        "wuji_right_hand": ImplicitActuatorCfg(
            joint_names_expr=["right_finger[1-5]_joint[1-4]"],
            effort_limit_sim=0.5,
            velocity_limit_sim=8.0,
            stiffness=5.0,
            damping=0.2,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
