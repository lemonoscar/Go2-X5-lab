"""Migrate a trusted 26000 checkpoint without changing initial actor outputs."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = ROOT / 'source/robot_lab/robot_lab/tasks/manager_based/locomotion/velocity/mdp/unified_contract.py'
spec = importlib.util.spec_from_file_location('unified_contract', CONTRACT)
contract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(contract)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('destination', type=Path)
    args = p.parse_args()
    if args.destination.exists():
        raise FileExistsError(args.destination)
    old = torch.load(args.source, weights_only=True, map_location='cpu')
    expanded = contract.expand_state_dict(old['model_state_dict'])
    # Validate full deterministic actor, including the checkpoint's hidden layers.
    def actor(state, x):
        for index in (0, 2, 4, 6):
            x = torch.nn.functional.linear(x, state[f'actor.{index}.weight'], state[f'actor.{index}.bias'])
            if index != 6:
                x = torch.nn.functional.elu(x)
        return x
    torch.manual_seed(42)
    x = torch.randn(256, contract.OBS_DIM)
    before = actor(old['model_state_dict'], x[:, :260])
    after = actor(expanded, x)
    torch.testing.assert_close(before, after, atol=2e-5, rtol=2e-5)
    manifest = dict(
        source_sha256=hashlib.sha256(args.source.read_bytes()).hexdigest(),
        source_iteration=old['iter'], obs_dim=contract.OBS_DIM, action_dim=12,
        prefix_dim=260, history_frames=4, history_frame_dim=54, history_order='oldest_first_previous_frames',
        history_reset='repeat_first_measured_frame', policy_dt=0.02,
        extra_order=['phase_sin', 'phase_cos', 'standing', 'anchor_x_body', 'anchor_y_body',
                     'yaw_error_sin', 'yaw_error_cos_minus_one', 'history_4x54'],
        history_frame_order=['base_ang_vel_3', 'projected_gravity_3', 'joint_pos_rel_18',
                             'joint_vel_18', 'last_leg_action_12'],
        optimizer='fresh', stage='1_flat_small_arm_disturbance_no_payload',
        max_initial_action_difference=float((before-after).abs().max()),
    )
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    # RSL load(..., load_optimizer=False) does not need optimizer_state_dict.
    torch.save({'model_state_dict': expanded, 'iter': 0, 'infos': {'unified_contract': manifest}}, args.destination)
    args.destination.with_suffix('.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
