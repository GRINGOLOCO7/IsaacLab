import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os
from math import pi

BOTZO_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:\\Users\\grego\\Desktop\\GRINGO\\botzo\\botzo\\CAD_files\\URDF\\BOTZO_URDF_description\\urdf\\BOTZO_URDF\\BOTZO_URDF.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=12,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.1),
        joint_pos={
            ".*_HAA": 0.0,
            ".*_HFE": 0.8,
            ".*_KFE": 0.0,
            ".*_FOOT": 0.0,
        },
        joint_vel={".*": 0.0},
        # ROTETE ROBOT UPSIDE DOWN
        #rot=(0.0, 1.0, 0.0, 0.0),
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_HAA", ".*_HFE", ".*_KFE"],
            effort_limit_sim=1e6,     # increased from 3e5
            velocity_limit=1000.0,
            stiffness=5e6,            # 5x stronger
            damping=100.0,            # much higher damping
        ),
        # "legs": DCMotorCfg(
        #     joint_names_expr=[".*_HAA", ".*_HFE", ".*_KFE"],
        #     effort_limit=33.5,
        #     saturation_effort=33.5,
        #     velocity_limit=21.0,
        #     stiffness=25.0,
        #     damping=0.5,
        #     friction=0.0,
        # ),
        "foot": ImplicitActuatorCfg(
            joint_names_expr=[".*_FOOT"],
            effort_limit_sim=10.0,
            velocity_limit_sim=10.0,
            stiffness=50.0,
            damping=1.0,
        ),
    },

)