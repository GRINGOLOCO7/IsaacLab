import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(
    description="This script demonstrates adding a custom robot to an Isaac Lab environment."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


from isaaclab_assets.robots.botzo import BOTZO_CONFIG # source\isaaclab_assets\isaaclab_assets\robots\botzo.py
from botzo_IK_solver import *
import math


class NewRobotsSceneCfg(InteractiveSceneCfg):
    """Designs the scene."""

    # Ground-plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    dome_light = AssetBaseCfg(
        prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75))
    )

    # robot
    Botzo = BOTZO_CONFIG.replace(prim_path="{ENV_REGEX_NS}/Botzo")


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    idx = 0

    while simulation_app.is_running():
        # reset
        if count % 500 == 0:
            # reset counters
            count = 0
            # reset the scene entities to their initial positions offset by the environment origins
            robot_botzo_state = scene["Botzo"].data.default_root_state.clone()
            robot_botzo_state[:, :3] += scene.env_origins

            # copy the default root state to the sim for the jetbot's orientation and velocity
            scene["Botzo"].write_root_pose_to_sim(robot_botzo_state[:, :7])
            scene["Botzo"].write_root_velocity_to_sim(robot_botzo_state[:, 7:])

            # copy the default joint states to the sim
            joint_pos, joint_vel = (
                scene["Botzo"].data.default_joint_pos.clone(),
                scene["Botzo"].data.default_joint_vel.clone(),
            )
            scene["Botzo"].write_joint_state_to_sim(joint_pos, joint_vel)
            # clear internal buffers
            scene.reset()
            print("[INFO]: Resetting Botzo state...")



        # move botzo
        '''
        tensor([[
                0  -  BL_shoulder, 
                1  -  BR_shoulder, 
                2  -  FL_shoulder, 
                3  -  FR_shoulder, 

                4  -  BL_femur, 
                5  -  BR_femur, 
                6  -  FL_femur, 
                7  -  FR_femur, 

                8  -  BL_knee, 
                9  -  BR_knee, 
                10 -  FL_knee, 
                11 -  FR_knee
                ]], device='cuda:0')
        '''
        botzo_action = scene["Botzo"].data.default_joint_pos
        # print the difference btw current pos and target pos
        #print(f"{'='*10}\nCurrent: {scene['Botzo'].data.joint_pos},\nTarget: {botzo_action}\n{'='*10}\n\n\n")
        tolerance = math.radians(5.0)
        if not torch.all(torch.abs(scene["Botzo"].data.joint_pos - botzo_action) < tolerance):
            #print("position not reached")
            pass
        else:
            #print("position reached, pass to new target")
            # calculate angles
            FR_s_f_t = legIK(forward_targets_FR_BL[idx][0], forward_targets_FR_BL[idx][1], forward_targets_FR_BL[idx][2])
            FR_angle_shoulder, FR_angle_femur, FR_angle_knee = FR_s_f_t
            target_FR_angle_shoulder = math.radians(real_sim_angle(FR_angle_shoulder,joint_ids["FR"]["shoulder"]))
            target_FR_angle_femur = math.radians(real_sim_angle(FR_angle_femur, joint_ids["FR"]["femur"]))
            target_FR_angle_knee = math.radians(real_sim_angle(FR_angle_knee, joint_ids["FR"]["knee"]))
            FL_s_f_t = legIK(forward_targets_FL_BR[idx][0], forward_targets_FL_BR[idx][1], forward_targets_FL_BR[idx][2])
            FL_angle_shoulder, FL_angle_femur, FL_angle_knee = FL_s_f_t
            target_FL_angle_shoulder = math.radians(real_sim_angle(FL_angle_shoulder,joint_ids["FL"]["shoulder"]))
            target_FL_angle_femur = math.radians(real_sim_angle(FL_angle_femur, joint_ids["FL"]["femur"]))
            target_FL_angle_knee = math.radians(real_sim_angle(FL_angle_knee, joint_ids["FL"]["knee"]))
            BR_s_f_t = legIK(forward_targets_FL_BR[idx][0], forward_targets_FL_BR[idx][1], forward_targets_FL_BR[idx][2])
            BR_angle_shoulder, BR_angle_femur, BR_angle_knee = BR_s_f_t
            target_BR_angle_shoulder = math.radians(real_sim_angle(BR_angle_shoulder,joint_ids["BR"]["shoulder"]))
            target_BR_angle_femur = math.radians(real_sim_angle(BR_angle_femur, joint_ids["BR"]["femur"]))
            target_BR_angle_knee = math.radians(real_sim_angle(BR_angle_knee, joint_ids["BR"]["knee"]))
            BL_s_f_t = legIK(forward_targets_FR_BL[idx][0], forward_targets_FR_BL[idx][1], forward_targets_FR_BL[idx][2])
            BL_angle_shoulder, BL_angle_femur, BL_angle_knee = BL_s_f_t
            target_BL_angle_shoulder = math.radians(real_sim_angle(BL_angle_shoulder,joint_ids["BL"]["shoulder"]))
            target_BL_angle_femur = math.radians(real_sim_angle(BL_angle_femur, joint_ids["BL"]["femur"]))
            target_BL_angle_knee = math.radians(real_sim_angle(BL_angle_knee, joint_ids["BL"]["knee"]))
            
            if idx >= len(forward_targets_FR_BL)-1:
                idx = 0  # Reset index if it exceeds the number of targets
            else:
                idx += 1
        # set the angles to the botzo action
        botzo_action[:, 0] = target_BL_angle_shoulder
        botzo_action[:, 1] = target_BR_angle_shoulder
        botzo_action[:, 2] = target_FL_angle_shoulder
        botzo_action[:, 3] = target_FR_angle_shoulder
        botzo_action[:, 4] = target_BL_angle_femur
        botzo_action[:, 5] = target_BR_angle_femur
        botzo_action[:, 6] = target_FL_angle_femur
        botzo_action[:, 7] = target_FR_angle_femur
        botzo_action[:, 8] = target_BL_angle_knee
        botzo_action[:, 9] = target_BR_angle_knee
        botzo_action[:, 10] = target_FL_angle_knee
        botzo_action[:, 11] = target_FR_angle_knee
        # write the action to the sim
        scene["Botzo"].set_joint_position_target(botzo_action)



        scene.write_data_to_sim()
        sim.step()
        sim_time += sim_dt
        count += 1
        scene.update(sim_dt)


def main():
    """Main function."""
    # Initialize the simulation context
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)

    sim.set_camera_view([3.5, 0.0, 3.2], [0.0, 0.0, 0.5])
    # design scene
    scene_cfg = NewRobotsSceneCfg(args_cli.num_envs, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    main()
    simulation_app.close()