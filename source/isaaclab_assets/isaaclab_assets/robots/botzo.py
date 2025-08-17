import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

stiffness = 2000.0     # N⋅m/rad - Very stiff for standing
damping = 200.0        # N⋅m⋅s/rad - High damping for stability
effort_limit_sim = 4.0 # N⋅m - Allow higher than rated for simulation
velocity_limit_sim = 10.0 # rad/s - Slightly higher for responsiveness

stiffness *= 40
damping *= 15
effort_limit_sim *= 25
velocity_limit_sim *= 10

BOTZO_CONFIG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:\\Users\\grego\\Desktop\\GRINGO\\botzo\\botzo\\simulation\\reinforcement_learning\\botzo_USD\\botzo_USD.usd",
        scale=(0.3, 0.3, 0.3),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "BL_shoulder_joint": 0.0,
            "BL_femur_joint": 0.0,
            "BL_knee_joint": 0.0,

            "BR_shoulder_joint": 0.0,
            "BR_femur_joint": 0.0,
            "Revolute_43": 0.0,

            "FL_shoulder_joint": 0.0,
            "FL_femur_joint": 0.0,
            "FL_knee_joint": 0.0,

            "FR_shoulder_joint": 0.0,
            "FR_femur_joint": 0.0,
            "FR_knee_joint": 0.0,
        },
        pos=(0.0, 0.0, 0.25),
        # rotate 90 degrees around the x-axis
        rot=(-0.7071, 0.0, 0.0, 0.7071),  # Quaternion for 90 degrees rotation around x-axis
        # rotate upside down
        #rot=(0.0, 1.0, 0.0, 0.0),
    ),
    actuators={
        "BL_shoulder_act": ImplicitActuatorCfg(joint_names_expr=["BL_shoulder_joint"],effort_limit_sim=effort_limit_sim, velocity_limit_sim=velocity_limit_sim, stiffness=stiffness, damping=damping),
        "BL_femur_act": ImplicitActuatorCfg(joint_names_expr=["BL_femur_joint"], effort_limit_sim=effort_limit_sim, velocity_limit_sim=velocity_limit_sim, stiffness=stiffness, damping=damping),
        "BL_knee_act": ImplicitActuatorCfg(joint_names_expr=["BL_knee_joint"], effort_limit_sim=effort_limit_sim, velocity_limit_sim=velocity_limit_sim, stiffness=stiffness, damping=damping),
        "BR_shoulder_act": ImplicitActuatorCfg(joint_names_expr=["BR_shoulder_joint"], effort_limit_sim=effort_limit_sim, velocity_limit_sim=velocity_limit_sim, stiffness=stiffness, damping=damping),
        "BR_femur_act": ImplicitActuatorCfg(joint_names_expr=["BR_femur_joint"], effort_limit_sim=effort_limit_sim, velocity_limit_sim=velocity_limit_sim, stiffness=stiffness, damping=damping),
        "Revolute_43_act": ImplicitActuatorCfg(joint_names_expr=["Revolute_43"], effort_limit_sim=effort_limit_sim, velocity_limit_sim=velocity_limit_sim, stiffness=stiffness, damping=damping),
        "FL_shoulder_act": ImplicitActuatorCfg(joint_names_expr=["FL_shoulder_joint"], effort_limit_sim=effort_limit_sim, velocity_limit_sim=velocity_limit_sim, stiffness=stiffness, damping=damping),
        "FL_femur_act": ImplicitActuatorCfg(joint_names_expr=["FL_femur_joint"], effort_limit_sim=effort_limit_sim, velocity_limit_sim=velocity_limit_sim, stiffness=stiffness, damping=damping),
        "FL_knee_act": ImplicitActuatorCfg(joint_names_expr=["FL_knee_joint"], effort_limit_sim=effort_limit_sim, velocity_limit_sim=velocity_limit_sim, stiffness=stiffness, damping=damping),
        "FR_shoulder_act": ImplicitActuatorCfg(joint_names_expr=["FR_shoulder_joint"], effort_limit_sim=effort_limit_sim, velocity_limit_sim=velocity_limit_sim, stiffness=stiffness, damping=damping),
        "FR_femur_act": ImplicitActuatorCfg(joint_names_expr=["FR_femur_joint"], effort_limit_sim=effort_limit_sim, velocity_limit_sim=velocity_limit_sim, stiffness=stiffness, damping=damping),
        "FR_knee_act": ImplicitActuatorCfg(joint_names_expr=["FR_knee_joint"], effort_limit_sim=effort_limit_sim, velocity_limit_sim=velocity_limit_sim, stiffness=stiffness, damping=damping),
    },
)
