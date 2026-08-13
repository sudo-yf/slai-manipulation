"""Shared Isaac Lab scene configuration for drawer retrieval collection."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from .robot_cfg import UR5_WUJI_RIGHT_CFG

DRAWER_OPEN_POSITION = 0.40
DRAWER_FLOOR_Z = 0.67
DRAWER_FRONT_X = 0.70
TARGET_RADIUS = 0.026
TARGET_HEIGHT = 0.07
PEDESTAL_HEIGHT = 0.50
FAR_CAMERA_EYE = (-0.15, 1.65, 1.30)
FAR_CAMERA_TARGET = (0.64, 0.0, 0.72)
NEAR_CAMERA_EYE = (0.70, -0.50, 1.18)
NEAR_CAMERA_TARGET = (0.93, 0.0, 0.74)

FLOOR_MATERIAL = sim_utils.RigidBodyMaterialCfg(
    static_friction=0.85,
    dynamic_friction=0.70,
    restitution=0.02,
)
TARGET_VISUAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.92, 0.20, 0.08), roughness=0.38, metallic=0.08
)
PEDESTAL_PLATE_VISUAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.07, 0.09, 0.11), roughness=0.42, metallic=0.72
)
PEDESTAL_COLUMN_VISUAL = sim_utils.PreviewSurfaceCfg(
    diffuse_color=(0.24, 0.30, 0.35), roughness=0.52, metallic=0.48
)

TASK_ROBOT_CFG = UR5_WUJI_RIGHT_CFG.copy()
TASK_ROBOT_CFG.init_state.pos = (0.0, 0.0, PEDESTAL_HEIGHT)
TASK_ROBOT_CFG.init_state.joint_pos.update(
    {
        "shoulder_pan_joint": -0.173,
        "shoulder_lift_joint": -1.339,
        "elbow_joint": 1.113,
        "wrist_1_joint": -1.981,
        "wrist_2_joint": -1.468,
        "wrist_3_joint": 1.432,
        "wrist_fe_joint": 0.0,
        "wrist_ru_joint": 0.0,
    }
)


@configclass
class DrawerCollectionSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/Ground",
        spawn=sim_utils.GroundPlaneCfg(size=(4.0, 4.0), physics_material=FLOOR_MATERIAL),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(intensity=2200.0, color=(0.82, 0.84, 0.88)),
    )
    key_light = AssetBaseCfg(
        prim_path="/World/KeyLight",
        spawn=sim_utils.DistantLightCfg(intensity=1200.0, color=(1.0, 0.94, 0.84), angle=0.45),
        init_state=AssetBaseCfg.InitialStateCfg(rot=(0.8660, 0.1913, -0.3314, -0.3314)),
    )
    pedestal_base = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PedestalBase",
        spawn=sim_utils.CuboidCfg(
            size=(0.42, 0.42, 0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=PEDESTAL_PLATE_VISUAL,
            physics_material=FLOOR_MATERIAL,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.025)),
    )
    pedestal_column = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PedestalColumn",
        spawn=sim_utils.CylinderCfg(
            radius=0.12,
            height=0.40,
            axis="Z",
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=PEDESTAL_COLUMN_VISUAL,
            physics_material=FLOOR_MATERIAL,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.25)),
    )
    pedestal_top = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/PedestalTop",
        spawn=sim_utils.CylinderCfg(
            radius=0.17,
            height=0.05,
            axis="Z",
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=PEDESTAL_PLATE_VISUAL,
            physics_material=FLOOR_MATERIAL,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.475)),
    )
    robot = TASK_ROBOT_CFG
    cabinet = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Cabinet",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Sektion_Cabinet/sektion_cabinet_instanceable.usd"
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(1.25, 0.0, 0.40),
            rot=(0.0, 0.0, 0.0, 1.0),
            joint_pos={
                "door_left_joint": -1.57,
                "door_right_joint": 1.57,
                "drawer_bottom_joint": DRAWER_OPEN_POSITION,
                "drawer_top_joint": DRAWER_OPEN_POSITION,
            },
        ),
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=["drawer_top_joint", "drawer_bottom_joint"],
                effort_limit_sim=500.0,
                stiffness=1000.0,
                damping=100.0,
            ),
            "doors": ImplicitActuatorCfg(
                joint_names_expr=["door_left_joint", "door_right_joint"],
                effort_limit_sim=87.0,
                stiffness=20.0,
                damping=2.5,
            ),
        },
    )
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/TargetObject",
        spawn=sim_utils.CylinderCfg(
            radius=TARGET_RADIUS,
            height=TARGET_HEIGHT,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=12,
                solver_velocity_iteration_count=4,
                max_depenetration_velocity=0.5,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
            collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True, contact_offset=0.002, rest_offset=0.0
            ),
            visual_material=TARGET_VISUAL,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.90, dynamic_friction=0.75, restitution=0.01
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.00, 0.0, 0.72)),
    )
    far_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/FarCamera",
        update_period=1.0 / 15.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.5,
            focus_distance=1.5,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 3.0),
        ),
    )
    near_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/NearCamera",
        update_period=1.0 / 15.0,
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=15.5,
            focus_distance=0.7,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 2.0),
        ),
    )
