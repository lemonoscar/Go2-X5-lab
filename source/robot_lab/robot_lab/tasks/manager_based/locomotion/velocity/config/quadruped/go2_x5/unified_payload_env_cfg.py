"""Stage 2: original-frequency flat work replay plus rigid-load terrain adaptation."""
from isaaclab.utils import configclass
from isaaclab.managers import EventTermCfg as EventTerm, TerminationTermCfg as DoneTerm, CurriculumTermCfg as CurrTerm
import isaaclab.terrains as terrains
from .unified_carry_env_cfg import UnifiedCarryFlatEnvCfg, UnifiedCarryPPORunnerCfg
from robot_lab.tasks.manager_based.locomotion.velocity.mdp.unified_payload import (
    PayloadVelocityCommandCfg, PayloadArmCommandCfg, randomize_tcp_payload, patch_complete, payload_terrain_levels,
)


@configclass
class UnifiedPayloadTerrainEnvCfg(UnifiedCarryFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        # Inherit original 0.7--2 Hz / 20% swing, not the rejected smooth variant.
        self.commands.base_velocity = PayloadVelocityCommandCfg(
            asset_name='robot', joint_names=self.joint_names, resampling_time_range=(65.,65.),
            rel_standing_envs=0.0, rel_heading_envs=0., heading_command=False, debug_vis=False,
            ranges=PayloadVelocityCommandCfg.Ranges(lin_vel_x=(-.5,.75),lin_vel_y=(-.2,.2),ang_vel_z=(-.3,.3)))
        self.commands.arm_joint_pos = PayloadArmCommandCfg(debug_vis=False)
        self.scene.terrain.terrain_generator = terrains.TerrainGeneratorCfg(
            seed=42, curriculum=True, size=(16.,16.), border_width=12., num_rows=4, num_cols=12,
            horizontal_scale=.1, vertical_scale=.005, slope_threshold=.75, use_cache=False,
            sub_terrains={
                'flat': terrains.MeshPlaneTerrainCfg(proportion=6/12),
                'rough': terrains.HfRandomUniformTerrainCfg(proportion=2/12,noise_range=(.01,.06),noise_step=.01,border_width=1.),
                'blocks': terrains.MeshRandomGridTerrainCfg(proportion=1/12,grid_width=.45,grid_height_range=(.02,.10),platform_width=3.),
                'slope': terrains.HfPyramidSlopedTerrainCfg(proportion=1/12,slope_range=(0.,.25),platform_width=3.,border_width=1.),
                'stairs_up': terrains.MeshInvertedPyramidStairsTerrainCfg(proportion=1/12,step_height_range=(.03,.10),step_width=.3,platform_width=3.,border_width=1.),
                'stairs_down': terrains.MeshPyramidStairsTerrainCfg(proportion=1/12,step_height_range=(.03,.10),step_width=.3,platform_width=3.,border_width=1.),
            })
        self.scene.terrain.max_init_terrain_level = 0
        self.events.tcp_payload = EventTerm(func=randomize_tcp_payload, mode='reset')
        self.terminations.patch_complete = DoneTerm(func=patch_complete, time_out=True)
        self.curriculum.terrain_levels = CurrTerm(func=payload_terrain_levels)


@configclass
class UnifiedPayloadTerrainPPORunnerCfg(UnifiedCarryPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = 'go2_x5_unified_payload_terrain'
        self.max_iterations = 2000
