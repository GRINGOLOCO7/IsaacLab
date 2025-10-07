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
            solver_position_iteration_count=4, #20,
            solver_velocity_iteration_count=0, #8,
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
        # "legs": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_HAA", ".*_HFE", ".*_KFE"],
        #     effort_limit_sim=1e6,     # increased from 3e5
        #     velocity_limit=1000.0,
        #     stiffness=5e6,            # 5x stronger
        #     damping=100.0,            # much higher damping
        # ),
        # "foot": ImplicitActuatorCfg(
        #     joint_names_expr=[".*_FOOT"],
        #     effort_limit_sim=10.0,
        #     velocity_limit_sim=10.0,
        #     stiffness=50.0,
        #     damping=1.0,
        # ),

        # "legs": DCMotorCfg(
        #     joint_names_expr=[".*_HAA", ".*_HFE", ".*_KFE"],
        #     effort_limit=33.5,
        #     saturation_effort=33.5,
        #     velocity_limit=21.0,
        #     stiffness=25.0,
        #     damping=0.5,
        #     friction=0.0,
        # ),

        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_HAA", ".*_HFE", ".*_KFE"],
            effort_limit_sim=5000.0,
            velocity_limit=100.0,
            stiffness=15000.0,
            damping=1000.0,
        ),
        "foot": ImplicitActuatorCfg(
            joint_names_expr=[".*_FOOT"],
            effort_limit_sim=500.0,
            velocity_limit_sim=50.0,
            stiffness=5000.0,
            damping=100.0,
        ),
    },

)

'''
| Goal                          | What to change                    | Direction                       |
| ----------------------------- | --------------------------------- | ------------------------------- |
| Robot can’t stand / collapses | ↑ stiffness or ↑ effort_limit_sim | increase by ×2                  |
| Vibrates or jitters           | ↓ stiffness or ↑ damping          | halve stiffness, double damping |
| Movements feel “sluggish”     | ↓ damping                         | lower by 20–50%                 |
| Movements explode             | ↓ stiffness and ↓ dt              | start 1e3–1e4 range             |
'''