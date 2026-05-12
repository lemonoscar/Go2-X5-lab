# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
import isaaclab.envs.mdp as core_mdp
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from .umi_commands import UMI_ACTION_DIM, UMI_COMMAND_DIM, UMI_UNIFIED_JOINT_NAMES

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


UMI_POLICY_OBS_DIM: int = 67
UMI_CRITIC_OBS_DIM: int = 110
UMI_MINIMAL_PRIVILEGED_SETUP_DIM: int = 40

UMI_POLICY_OBS_COMPONENTS: tuple[tuple[str, int], ...] = (
    ("root_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos_rel", 18),
    ("joint_vel", 18),
    ("locomotion6d_command", UMI_COMMAND_DIM),
    ("last_action", UMI_ACTION_DIM),
)

UMI_CRITIC_OBS_COMPONENTS: tuple[tuple[str, int], ...] = (
    ("root_lin_vel", 3),
    ("root_ang_vel", 3),
    ("projected_gravity", 3),
    ("joint_pos_rel", 18),
    ("joint_vel", 18),
    ("kp", 18),
    ("kd", 18),
    ("base_mass", 1),
    ("base_com_offset", 3),
    ("locomotion6d_command", UMI_COMMAND_DIM),
    ("last_action", UMI_ACTION_DIM),
)


def _resolve_ids(indices: slice | list[int] | torch.Tensor, size: int, device: str) -> torch.Tensor:
    if isinstance(indices, slice):
        return torch.arange(size, device=device)[indices]
    if isinstance(indices, torch.Tensor):
        return indices.to(device=device, dtype=torch.long)
    return torch.tensor(indices, device=device, dtype=torch.long)


def _resolve_body_ids(asset: Articulation, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return _resolve_ids(asset_cfg.body_ids, len(asset.body_names), asset.device)


def _resolve_joint_ids(asset: Articulation, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    return _resolve_ids(asset_cfg.joint_ids, len(asset.joint_names), asset.device)


def _actuator_property(env: "ManagerBasedEnv", key: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    full = torch.zeros((env.num_envs, len(asset.joint_names)), device=asset.device)
    for actuator in asset.actuators.values():
        joint_ids = _resolve_ids(actuator.joint_indices, len(asset.joint_names), asset.device)
        full[:, joint_ids] = getattr(actuator, key)
    return full[:, _resolve_joint_ids(asset, asset_cfg)]


def locomotion6d_commands(env: "ManagerBasedEnv", command_name: str) -> torch.Tensor:
    """Return the 7-D locomotion command scaled to match the UMI observation convention."""

    command_term = env.command_manager.get_term(command_name)
    command = command_term.command.clone()
    lin_vel_scale = getattr(command_term.cfg, "lin_vel_obs_scale", 2.0)
    ang_vel_scale = getattr(command_term.cfg, "ang_vel_obs_scale", 0.25)
    command[:, :2] *= lin_vel_scale
    command[:, 2] *= ang_vel_scale
    return command


def actuator_stiffness(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=UMI_UNIFIED_JOINT_NAMES, preserve_order=True),
) -> torch.Tensor:
    return _actuator_property(env, "stiffness", asset_cfg)


def actuator_damping(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=UMI_UNIFIED_JOINT_NAMES, preserve_order=True),
) -> torch.Tensor:
    return _actuator_property(env, "damping", asset_cfg)


def joint_friction_coeff(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=UMI_UNIFIED_JOINT_NAMES, preserve_order=True),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.data.joint_friction_coeff[:, _resolve_joint_ids(asset, asset_cfg)]


def body_mass(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["base"], preserve_order=True),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses().to(asset.device)
    return masses[:, _resolve_body_ids(asset, asset_cfg)]


def body_com_offset(
    env: "ManagerBasedEnv",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["base"], preserve_order=True),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    coms = asset.root_physx_view.get_coms().to(asset.device)
    return coms[:, _resolve_body_ids(asset, asset_cfg), :3].reshape(env.num_envs, -1)


@configclass
class UmiPolicyObservationsCfg(ObsGroup):
    """67-D policy observation group aligned with UMI locomotion6d."""

    root_ang_vel = ObsTerm(
        func=core_mdp.base_ang_vel,
        noise=Unoise(n_min=-0.2, n_max=0.2),
        clip=(-100.0, 100.0),
        scale=0.25,
    )
    projected_gravity = ObsTerm(
        func=core_mdp.projected_gravity,
        noise=Unoise(n_min=-0.05, n_max=0.05),
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    joint_pos = ObsTerm(
        func=core_mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=UMI_UNIFIED_JOINT_NAMES, preserve_order=True)},
        noise=Unoise(n_min=-0.01, n_max=0.01),
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    joint_vel = ObsTerm(
        func=core_mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=UMI_UNIFIED_JOINT_NAMES, preserve_order=True)},
        noise=Unoise(n_min=-1.5, n_max=1.5),
        clip=(-100.0, 100.0),
        scale=0.05,
    )
    locomotion6d_command = ObsTerm(
        func=locomotion6d_commands,
        params={"command_name": "locomotion6d"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    actions = ObsTerm(
        func=core_mdp.last_action,
        clip=(-100.0, 100.0),
        scale=1.0,
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class UmiCriticObservationsCfg(ObsGroup):
    """110-D critic observation group with minimal privileged setup."""

    root_lin_vel = ObsTerm(
        func=core_mdp.base_lin_vel,
        clip=(-100.0, 100.0),
        scale=2.0,
    )
    root_ang_vel = ObsTerm(
        func=core_mdp.base_ang_vel,
        clip=(-100.0, 100.0),
        scale=0.25,
    )
    projected_gravity = ObsTerm(
        func=core_mdp.projected_gravity,
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    joint_pos = ObsTerm(
        func=core_mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=UMI_UNIFIED_JOINT_NAMES, preserve_order=True)},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    joint_vel = ObsTerm(
        func=core_mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=UMI_UNIFIED_JOINT_NAMES, preserve_order=True)},
        clip=(-100.0, 100.0),
        scale=0.05,
    )
    kp = ObsTerm(
        func=actuator_stiffness,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=UMI_UNIFIED_JOINT_NAMES, preserve_order=True)},
        clip=(-100.0, 100.0),
        scale=0.1,
    )
    kd = ObsTerm(
        func=actuator_damping,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=UMI_UNIFIED_JOINT_NAMES, preserve_order=True)},
        clip=(-100.0, 100.0),
        scale=10.0,
    )
    base_mass = ObsTerm(
        func=body_mass,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"], preserve_order=True)},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    base_com_offset = ObsTerm(
        func=body_com_offset,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["base"], preserve_order=True)},
        clip=(-100.0, 100.0),
        scale=10.0,
    )
    locomotion6d_command = ObsTerm(
        func=locomotion6d_commands,
        params={"command_name": "locomotion6d"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    actions = ObsTerm(
        func=core_mdp.last_action,
        clip=(-100.0, 100.0),
        scale=1.0,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class UmiLocomotion6dObservationsCfg:
    """Observation container for UMI locomotion6d envs."""

    policy: UmiPolicyObservationsCfg = UmiPolicyObservationsCfg()
    critic: UmiCriticObservationsCfg = UmiCriticObservationsCfg()
