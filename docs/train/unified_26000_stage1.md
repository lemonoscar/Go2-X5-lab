# Stage 1 reward changes

Source: model_26000 SHA256 b46813c2daf17354c6344766816528cb04200f0fb996b53d5f3eb12302b49d51.
Seed 42, fixed LR 5e-5, entropy 0.001, 32 steps per rollout, save every 100 updates.
Initial budget 1500 updates, one GPU. Small arm excitation only, no payload claims.

- XY velocity tracking 5.0; yaw tracking 2.5.
- Soft four-beat swing-force/stance-speed score 0.3; RL, FL, RR, FR phase order.
- Standing XY squared drift / 5cm weight -0.6.
- Standing XY squared velocity / 0.1m/s weight -0.4.
- Standing yaw and out-of-band roll/pitch squared cost weight -0.4.
- Remove original trot reward, stand-still joint penalty and no-command contact reward.
- Reduce nominal joint posture weight to -0.05 and level-body weight to -0.1.
- Remaining Rough regularizers retained. Source default posture restored.

Reward values are training coefficients, not acceptance thresholds. Full user contract
and unresolved gates are recorded in docs/development/PLANS.md and docs/user/unified_26000.md.
