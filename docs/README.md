# Go2-X5-lab Documentation

This directory is split by document purpose. Use this page as the entry point before editing or running the project.

Chinese overview: [Readme_CN.md](Readme_CN.md).

## Directory Roles

| Directory | Role | Git policy |
| --- | --- | --- |
| `development/` | Codex-facing repository map, current RL policy contract, VLA-RL integration contract, workflow, and `PLANS.md`. | Ignored |
| `user/` | Developer-facing setup, configuration, framework, training, replay, keyboard, camera, and VLA-SFT usage docs. | Pushable |
| `train/` | Local records for training rewards, reward weights, PPO settings, and stage notes. | Ignored |
| `history/` | Expired development notes and one-time check reports. First line of each file must be an expiry notice. | Ignored |
| `appends/images/` | Stable images used by root `README.md` and docs. | Pushable for stable image assets |
| `appends/media/` | Local screenshots and runtime visual check outputs. | Ignored |

## Current Documents

### Local Development

These files are local-only and ignored by git:

- `development/01_repository_map.md`: main repository directories and ownership boundaries.
- `development/02_current_rl_policy.md`: current DogOnly PPO policy, observations, actions, training task IDs, and config paths.
- `development/03_vla_rl_integration.md`: VLA-RL integration surface, 10-D action contract, and task scaffolds.
- `development/04_workflow.md`: plan-before-code workflow.
- `development/PLANS.md`: plan file requiring human approval before code changes.

### User

- [user/environment.md](user/environment.md): environment initialization and configuration.
- [user/network_and_framework.md](user/network_and_framework.md): Isaac Lab / RSL-RL / task framework explanation.
- [user/training_and_replay.md](user/training_and_replay.md): training, resume, checkpoint migration, and replay commands.
- [user/keyboard_control.md](user/keyboard_control.md): keyboard control usage.
- [user/camera_visualization.md](user/camera_visualization.md): camera visualization and testing.
- [user/vla_sft.md](user/vla_sft.md): VLA-SFT scene and visualization docs.

### Local Training Records

- `train/dogonly_ppo_training_standard.md`: current DogOnly PPO training standard.
- `train/dogonly_ppo_p1_foundation_flat.md`: P1 reward weights and foundation flat training record.
- `train/dogonly_ppo_p2_rough_transfer_arm_warmup.md`: P2a/P2b rough transfer and arm warmup record.

### History

The `history/` directory stores old repair logs, completed migration plans, asset manifests, non-current training route folders, VLA-SFT design drafts, and visual check reports. These are not current instructions.

## Placement Rules

- Put active architecture and development conventions in `development/`.
- Put commands a developer runs in `user/`.
- Put reward weights, PPO settings, checkpoint notes, and training stage evidence in `train/`.
- Move obsolete plans or one-time reports to `history/` and add the expiry line before the old title.
