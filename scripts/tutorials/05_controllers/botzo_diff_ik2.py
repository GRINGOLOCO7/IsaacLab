import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates using differential IK controller with Botzo quadruped robot.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import subtract_frame_transforms, quat_rotate

##
# Pre-defined configs
##
from isaaclab_assets.robots.botzo import BOTZO_CONFIG

class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Configuration for a cart-pole scene."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg()
    )

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    Botzo = BOTZO_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Botzo")

def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    """Runs the simulation loop."""
    # Extract scene entities
    robot = scene["Botzo"]
    
    # Define foot offset from ankle (from URDF)
    # FR_foot offset: xyz="-0.224988 -0.015448 -0.157285" "0.0066 0.067876 -0.059031"
    #foot_offset = torch.tensor([-0.224988, -0.015448, -0.157285], device=sim.device)
    foot_offset = torch.tensor([0.0066, 0.067876, -0.059031], device=sim.device)

    # Create controller for 3-DOF leg (shoulder, femur, knee)
    diff_ik_cfg = DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls")
    diff_ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=scene.num_envs, device=sim.device)

    # Markers for visualization
    frame_marker_cfg = FRAME_MARKER_CFG.copy()
    frame_marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
    ee_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_current"))
    goal_marker = VisualizationMarkers(frame_marker_cfg.replace(prim_path="/Visuals/ee_goal"))

    # Define goals for the foot (relative to robot base)
    # These positions now account for the foot, not just the ankle
    ee_goals = [
        [0.0, 0.3, 0.0, 0.0, 0.0, 0.0, 1.0],   # Adjusted Z for foot contact
        [0.1, 0.3, 0.0, 0.0, 0.0, 0.0, 1.0],   
        [0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 1.0],  
    ]
    # add offset 0.2315880018714994 0.08332368868620879 0.08627405461043021 to ee_goals
    #ee_goals = [[goal[0] + 0.2315880018714994, goal[1] + 0.08332368868620879, goal[2] + 0.08627405461043021] + goal[3:] for goal in ee_goals]

    ee_goals = torch.tensor(ee_goals, device=sim.device)
    
    # Track the given command
    current_goal_idx = 0
    # Create buffers to store actions
    ik_commands = torch.zeros(scene.num_envs, diff_ik_controller.action_dim, device=robot.device)
    ik_commands[:] = ee_goals[current_goal_idx]

    robot_entity_cfg_shoulder_pos = SceneEntityCfg(
        "Botzo", 
        body_names=["FR_shoulder_servo_arm_v11"] 
    )
    robot_entity_cfg_shoulder_pos.resolve(scene)
    # Then get the position
    FR_shoulder_pos = robot.data.body_pos_w[:, robot_entity_cfg_shoulder_pos.body_ids[0]]
    FR_shoulder_x = FR_shoulder_pos[:, 0]
    FR_shoulder_y = FR_shoulder_pos[:, 1] 
    FR_shoulder_z = FR_shoulder_pos[:, 2]
    print(f"[INFO]: FR_shoulder position - X: {FR_shoulder_x}, Y: {FR_shoulder_y}, Z: {FR_shoulder_z}")

    # Define robot-specific parameters for Botzo - FR leg
    robot_entity_cfg = SceneEntityCfg(
        "Botzo", 
        joint_names=["FR_shoulder_joint", "FR_femur_joint", "FR_knee_joint"],  # FR leg joints
        body_names=["FR_leg_ankle_v11"]  # End-effector (ankle) - we'll add foot offset
    )
    
    # Resolving the scene entities
    robot_entity_cfg.resolve(scene)
    
    # Obtain the frame index of the end-effector
    if robot.is_fixed_base:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0] - 1
    else:
        ee_jacobi_idx = robot_entity_cfg.body_ids[0]

    # Define simulation stepping
    sim_dt = sim.get_physics_dt()
    count = 0
    
    # Simulation loop
    while simulation_app.is_running():
        # get position of FR_shoulder_joint body_link_state_w 
        FR_shoulder_pos = robot.data.body_pos_w[:, robot_entity_cfg_shoulder_pos.body_ids[0]]
        FR_shoulder_x = FR_shoulder_pos[:, 0]
        FR_shoulder_y = FR_shoulder_pos[:, 1] 
        FR_shoulder_z = FR_shoulder_pos[:, 2]
        print(f"[INFO]: FR_shoulder position - X: {FR_shoulder_x}, Y: {FR_shoulder_y}, Z: {FR_shoulder_z}")
        # reset
        if count % 300 == 0:
            # reset time
            count = 0
            # reset joint state
            joint_pos = robot.data.default_joint_pos.clone()
            joint_vel = robot.data.default_joint_vel.clone()
            robot.write_joint_state_to_sim(joint_pos, joint_vel)
            robot.reset()
            # reset actions
            # transform ee_goal relative to current shoulder position
            shoulder_offset = torch.tensor([FR_shoulder_x, FR_shoulder_y, FR_shoulder_z, 0, 0, 0, 0], device=sim.device)
            ik_commands[:] = shoulder_offset + ee_goals[current_goal_idx]
            joint_pos_des = joint_pos[:, robot_entity_cfg.joint_ids].clone()
            # reset controller
            diff_ik_controller.reset()
            diff_ik_controller.set_command(ik_commands)
            # change goal
            current_goal_idx = (current_goal_idx + 1) % len(ee_goals)
            print(f"[INFO]: Switching to goal {current_goal_idx}: {ee_goals[current_goal_idx]}")
        else:
            # obtain quantities from simulation
            try:
                jacobian = robot.root_physx_view.get_jacobians()[:, ee_jacobi_idx, :, robot_entity_cfg.joint_ids]
                ankle_pose_w = robot.data.body_pose_w[:, robot_entity_cfg.body_ids[0]]
                root_pose_w = robot.data.root_pose_w
                joint_pos = robot.data.joint_pos[:, robot_entity_cfg.joint_ids]
                
                # Calculate foot position from ankle position + offset
                # Transform foot offset from ankle frame to world frame
                foot_offset_w = quat_rotate(ankle_pose_w[:, 3:7], foot_offset.unsqueeze(0).repeat(scene.num_envs, 1))
                foot_pos_w = ankle_pose_w[:, 0:3] + foot_offset_w
                foot_quat_w = ankle_pose_w[:, 3:7]  # Assume same orientation as ankle
                
                # Compute foot frame in root frame
                foot_pos_b, foot_quat_b = subtract_frame_transforms(
                    root_pose_w[:, 0:3], root_pose_w[:, 3:7], 
                    foot_pos_w, foot_quat_w
                )
                
                # Compute the joint commands using foot position
                joint_pos_des = diff_ik_controller.compute(
                    foot_pos_b, 
                    foot_quat_b,
                    jacobian, 
                    joint_pos
                )
            except Exception as e:
                print(f"[WARNING]: IK computation failed: {e}")
                # Fall back to default positions
                joint_pos_des = robot.data.default_joint_pos[:, robot_entity_cfg.joint_ids].clone()

        # Apply actions to the specific leg joints
        robot.set_joint_position_target(joint_pos_des, joint_ids=robot_entity_cfg.joint_ids)
        scene.write_data_to_sim()
        
        # perform step
        sim.step()
        # update sim-time
        count += 1
        # update buffers
        scene.update(sim_dt)

        # Update visualization markers - show actual foot position
        try:
            ankle_pose_w = robot.data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7]
            # Calculate foot position for visualization
            foot_offset_w = quat_rotate(ankle_pose_w[:, 3:7], foot_offset.unsqueeze(0).repeat(scene.num_envs, 1))
            foot_pos_w = ankle_pose_w[:, 0:3] + foot_offset_w
            
            # Update marker positions (show foot position, not ankle)
            ee_marker.visualize(foot_pos_w, ankle_pose_w[:, 3:7])
            goal_marker.visualize(ik_commands[:, 0:3] + scene.env_origins, ik_commands[:, 3:7])
        except Exception as e:
            print(f"[WARNING]: Visualization failed: {e}")


def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view([2.5, 2.5, 2.5], [0.0, 0.0, 0.0])
    # Design scene
    scene_cfg = NewRobotsSceneCfg(num_envs=args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()