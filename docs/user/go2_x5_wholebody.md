# Go2-X5 WholeBody Inference

## What It Is

The task `RobotLab-Isaac-Go2-X5-WholeBody-v0` exposes one 10-D action:

```text
[vx, vy, wz, tcp_x, tcp_y, tcp_z, tcp_roll, tcp_pitch, tcp_yaw, gripper]
```

The first three entries command base velocity. The next six specify an absolute
`arm_eef_link` Cartesian pose. The final entry is continuous gripper opening
(`0=closed`, `1=open`). The task runs the frozen RoboDuet Dog/Arm `019999`
models and Pink IK internally at 50 Hz.

This is an inference task. It does not provide a 10-D PPO trainer.

## Install The Optional IK Stack

Use the project Isaac Lab environment and install the feature extra once:

```bash
python -m pip install -e "source/robot_lab[wholebody-ik]"
```

The extra is not required for existing DogOnly tasks or for importing the
whole-body PyTorch model definitions.

## Import The Trusted 019999 Run

From the repository root:

```bash
python scripts/checkpoints/import_roboduet_go2_x5_wholebody.py \
  /path/to/dummy-w11894zo_seed5806
```

The importer requires these three source files:

```text
parameters.pkl
checkpoints_dog/ac_weights_019999.pt
checkpoints_arm/ac_weights_019999.pt
```

It verifies the frozen network contract, copies the full raw state dicts into
the ignored `models/go2_x5_wholebody/019999` directory, and writes the safe
hash-bound YAML manifest. Runtime never loads `parameters.pkl`.

To keep the model elsewhere, set:

```bash
export GO2_X5_WHOLEBODY_MODEL_DIR=/absolute/path/to/019999
```

Verify one environment and five controller ticks before the long replay:

```bash
python test/environment/smoke_test_go2_x5_wholebody.py --headless
```

Startup fails closed if the checkpoint/URDF hashes, named-joint order, live
position/effort/velocity limits, PD/friction profile, total mass, or whole-body
CoM differ from the imported manifest.

## Run The Fixed 60-Second Contract

```bash
python scripts/wholebody/replay_go2_x5_wholebody.py \
  --task RobotLab-Isaac-Go2-X5-WholeBody-v0 \
  --headless \
  --output-dir outputs/go2_x5_wholebody/019999_contract
```

The 3000-step replay covers stand, positive/negative `vx/vy/wz`, TCP position
and orientation steps, gripper open/close, finite clipping, NaN rejection,
an extreme IK target, and recovery. Outputs are:

```text
samples.jsonl
summary.json
REPORT.md
```

`contract_pass` means the full simulated 60 seconds completed without an
unexpected reset or non-finite joint target. `stability_pass`, fall/contact,
IK-hold, stall counts, and controller mean/p95/max timing are reported separately.

## Command Rules

- Velocity limits: `vx [-1.5,1.5]`, `vy [-0.3,0.3]`, `wz [-1.5,1.5]`.
- TCP is absolute, base-yaw aligned and ground referenced; it is not a delta.
- Pose orientation uses roll/pitch/yaw with `Rz @ Ry @ Rx` composition.
- Finite out-of-range input is clipped and reported.
- NaN/Inf rejects the whole 10-D packet and holds the previous valid command.
- Upper commands may arrive below 50 Hz; the task uses zero-order hold.
- Pink IK currently requires `num_envs=1`; larger values fail at startup.
- Falls, non-foot contact, IK hold, and stall are diagnostics, not auto-reset triggers.

At reset the task immediately uses zero base velocity, a closed gripper, zero
model histories, and the canonical TCP equivalent of RoboDuet legacy command
`[0.5, 0.2, 0, 0.1, 0.5, 0]`. There is no action ramp or warmup.
