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


@configclass
class UnifiedTerrainFocusEnvCfg(UnifiedPayloadTerrainEnvCfg):
    """80% terrain locomotion and 20% full-range stationary arm stabilization."""
    def __post_init__(self):
        super().__post_init__()
        from robot_lab.tasks.manager_based.locomotion.velocity.mdp.unified_payload import (
            TerrainFocusVelocityCommandCfg, FullRangeArmCommandCfg,
        )
        self.scene.num_envs = 4096
        self.commands.base_velocity = TerrainFocusVelocityCommandCfg(
            asset_name='robot', joint_names=self.joint_names, resampling_time_range=(65.,65.),
            rel_standing_envs=0., rel_heading_envs=0., heading_command=False, debug_vis=False,
            ranges=TerrainFocusVelocityCommandCfg.Ranges(lin_vel_x=(-.5,.75),lin_vel_y=(-.2,.2),ang_vel_z=(-.3,.3)))
        self.commands.arm_joint_pos = FullRangeArmCommandCfg(debug_vis=False)
        terrain = self.scene.terrain.terrain_generator
        terrain.num_cols = 10
        terrain.sub_terrains = {
            'flat': terrains.MeshPlaneTerrainCfg(proportion=.2),
            'rough': terrains.HfRandomUniformTerrainCfg(proportion=.2,noise_range=(.01,.06),noise_step=.01,border_width=1.),
            'blocks': terrains.MeshRandomGridTerrainCfg(proportion=.2,grid_width=.45,grid_height_range=(.02,.10),platform_width=3.),
            'stairs_up': terrains.MeshInvertedPyramidStairsTerrainCfg(proportion=.2,step_height_range=(.03,.10),step_width=.4,platform_width=2.4,border_width=4.6),
            'stairs_down': terrains.MeshPyramidStairsTerrainCfg(proportion=.2,step_height_range=(.03,.10),step_width=.4,platform_width=2.4,border_width=4.6),
        }
        # Six surrounding rings plus center transition = seven risers; height curriculum remains.
        self.scene.terrain.max_init_terrain_level = 0


@configclass
class UnifiedTerrainFocusPPORunnerCfg(UnifiedPayloadTerrainPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = 'go2_x5_unified_terrain_focus'


@configclass
class UnifiedTaskArmEnvCfg(UnifiedTerrainFocusEnvCfg):
    """Initial empty/slow task-arm repair; terrain replay retains 0--450 g."""
    startup_hold_seconds: float = 1.0
    def __post_init__(self):
        super().__post_init__()
        from robot_lab.tasks.manager_based.locomotion.velocity.mdp.task_arm import TaskArmCommandCfg, task_payload, task_clearance_violation, spawn_task_robot
        self.commands.arm_joint_pos=TaskArmCommandCfg(debug_vis=False)
        self.scene.robot.spawn.articulation_props.enabled_self_collisions=True
        self.scene.robot.spawn.func=spawn_task_robot
        self.terminations.task_clearance=DoneTerm(func=task_clearance_violation)
        from isaaclab.managers import SceneEntityCfg
        from isaaclab.envs.mdp import illegal_contact
        self.terminations.arm_contact=DoneTerm(func=illegal_contact, params={
            'sensor_cfg':SceneEntityCfg('contact_forces',body_names=['arm_link.*']), 'threshold':1.0})
        terrain=self.scene.terrain.terrain_generator
        terrain.num_cols=20
        for name,cfg in terrain.sub_terrains.items():cfg.proportion=.6 if name=='flat' else .1
        self.events.tcp_payload=EventTerm(func=task_payload,mode='reset')
        # Explicit provisional URDF constraint, not a claim of hardware ratings.
        for name,a in self.scene.robot.actuators.items():
            if name.startswith('arm_'):
                proximal=name in ('arm_joint12','arm_joint3')
                a.effort_limit=a.effort_limit_sim=a.saturation_effort=15. if proximal else 3.
                a.velocity_limit=a.velocity_limit_sim=3.
                a.stiffness=250. if proximal else 80.
                a.damping=8. if proximal else 3.
        self.rewards.standing_position.weight=-2.
        self.rewards.standing_velocity.weight=-1.
        self.rewards.standing_posture.weight=-1.


@configclass
class UnifiedTaskArmPPORunnerCfg(UnifiedTerrainFocusPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name='go2_x5_unified_task_arm'
