from isaaclab.utils import configclass

from .rough_env_cfg import CustomQuadRoughEnvCfg

# RUN WITH: 
#   1. conda activate env_isaaclab 
#   2. cd C:\Users\grego\Desktop\GRINGO\IsaacLab\IsaacLab\scripts\reinforcement_learning\rsl_rl\
#   3. python train.py --task=Isaac-Velocity-Flat-Custom-Quad-v0 --num_envs=<n>

@configclass
class CustomQuadFlatEnvCfg(CustomQuadRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        self.rewards.flat_orientation_l2.weight = -2.5
        self.rewards.feet_air_time.weight = 0.25

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


class CustomQuadFlatEnvCfg_PLAY(CustomQuadFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing event
        self.events.base_external_force_torque = None
        self.events.push_robot = None
