import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
import os
from math import pi

BOTZO_ACTUATOR_CFG = ImplicitActuatorCfg(
    joint_names_expr=[
        ".*HAA",  # hip abduction/adduction
        ".*HFE",  # hip flexion/extension
        ".*KFE",  # knee flexion/extension
    ],
    effort_limit=80.0,      # ← allows enough torque to hold weight (try 400–800)
    velocity_limit=7.0,     # ← safe high limit
    stiffness=40.0,        # ← stronger position control (acts like motor Kp)
    damping=5.0,            # ← reasonable damping (acts like Kd)
)

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
        pos=(0.0, 0.0, 0.5),
        joint_pos={
            ".*HAA": 0.0,
            ".*HFE": 0.8,
            ".*KFE": 0.0,
            ".*FOOT": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*HAA", ".*HFE", ".*KFE"],
            effort_limit=60.0,
            saturation_effort=80.0,
            velocity_limit=10.0,
            stiffness={".*": 10.0},
            damping={".*": 0.2},
            friction={".*": 0.05},
        ),
    },
)