"""Torch-only training/deployment contract for the 26000 continuation.

Keep this module independent of Isaac so the exact same history and command
processing can be tested and used in a deployment adapter.
"""
import torch

BASE_DIM = 260
FRAME_DIM = 54
HISTORY_LENGTH = 4
EXTRA_DIM = 7 + HISTORY_LENGTH * FRAME_DIM
OBS_DIM = BASE_DIM + EXTRA_DIM


def gait_frequency(command, minimum=0.7, maximum=2.0):
    """Reference cycle Hz, zero at stand; includes pure yaw commands."""
    demand = (command[..., :2].norm(dim=-1) / 0.75 + command[..., 2].abs() / 0.6).clamp(0, 1)
    return torch.where((command != 0).any(-1), minimum + (maximum - minimum) * demand, 0.0)


def gait_swing_mask(phase, fraction=0.20):
    if not 0 < fraction <= 0.25:
        raise ValueError('Four-beat swing fraction must be in (0, 0.25]')
    return (phase[..., None] - phase.new_tensor([0, 0.25, 0.5, 0.75])) % 1 < fraction


def filter_command(command):
    if command.shape[-1] != 3 or not torch.isfinite(command).all():
        raise ValueError("Expected finite [...,3] velocity commands")
    dead = command.new_tensor([0.1, 0.05, 0.05])
    c = torch.where(command.abs() <= dead, 0.0, command)
    limits = c.new_tensor([0.75, 0.2, 0.3]).expand_as(c).clone()
    limits[..., 0] = torch.where(c[..., 0] < 0, 0.5, 0.75)
    norm = torch.linalg.vector_norm(c / limits, dim=-1, keepdim=True)
    c = c / norm.clamp_min(1.0)
    return torch.where(c.abs() <= dead, 0.0, c)


class ShortHistory:
    """Previous four frames, oldest first; fill all frames at episode reset."""

    def __init__(self, count, device):
        self.frames = torch.zeros(count, HISTORY_LENGTH, FRAME_DIM, device=device)
        self.valid = torch.zeros(count, dtype=torch.bool, device=device)
        self.output = self.frames.flatten(1).clone()
        self.last_step = None

    def reset(self, ids):
        self.valid[ids] = False

    def update(self, frame, step):
        # reset() can be followed by a second observation read in the same step.
        missing = ~self.valid
        self.frames[missing] = frame[missing, None, :]
        self.output[missing] = self.frames[missing].flatten(1)
        self.valid[:] = True
        if self.last_step != step:
            self.output.copy_(self.frames.flatten(1))
            self.frames[:, :-1] = self.frames[:, 1:].clone()
            self.frames[:, -1] = frame
            self.last_step = step
        return self.output


def expand_state_dict(state):
    """Zero added input columns; preserve every pretrained tensor otherwise."""
    if tuple(state['actor.0.weight'].shape) != (512, BASE_DIM):
        raise ValueError('Expected the original 260-input actor')
    if tuple(state['actor.6.weight'].shape) != (12, 128):
        raise ValueError('Expected 12 leg actions')
    result = {k: v.clone() for k, v in state.items()}
    for key in ('actor.0.weight', 'critic.0.weight'):
        old = state[key]
        if old.shape != (512, BASE_DIM):
            raise ValueError(f'Unexpected tensor {key}: {old.shape}')
        new = old.new_zeros(512, OBS_DIM)
        new[:, :BASE_DIM] = old
        result[key] = new
    return result
