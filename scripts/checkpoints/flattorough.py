#!/usr/bin/env python3
# Copyright (c) 2026
# SPDX-License-Identifier: Apache-2.0

"""
Pad (or truncate) RSL-RL ActorCritic input layers to match a new observation size.

Typical usage for flat -> rough transfer where only critic obs dims changed:
  python scripts/checkpoints/flattorough.py \
    --src logs/rsl_rl/go2_x5_rough/transfer_from_flat/model_33000.pt \
    --critic-in 259

If actor obs dims also changed (e.g., enabling policy height_scan):
  python scripts/checkpoints/flattorough.py \
    --src logs/rsl_rl/go2_x5_rough/transfer_from_flat/model_33000.pt \
    --critic-in 259 \
    --actor-in 259
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Dict

import torch


def _pad_or_truncate(w: torch.Tensor, new_in: int) -> torch.Tensor:
    """Pad (zeros) or truncate input dimension (dim=1) to new_in."""
    if w.ndim != 2:
        raise ValueError(f"Expected 2D weight tensor, got shape={tuple(w.shape)}")
    if w.shape[1] == new_in:
        return w
    new_w = torch.zeros((w.shape[0], new_in), dtype=w.dtype)
    copy = min(w.shape[1], new_in)
    new_w[:, :copy] = w[:, :copy]
    return new_w


def _maybe_update_layer(
    state_dict: Dict[str, torch.Tensor],
    weight_key: str,
    bias_key: str,
    new_in: int | None,
    label: str,
) -> bool:
    """Return True if updated."""
    if weight_key not in state_dict:
        print(f"[WARN] {label} weight not found: {weight_key}")
        return False
    if new_in is None:
        w = state_dict[weight_key]
        print(f"[INFO] {label} keep: {weight_key} shape={tuple(w.shape)} (no target dim provided)")
        return False
    w = state_dict[weight_key]
    old_in = w.shape[1]
    if old_in == new_in:
        print(f"[INFO] {label} already matches: {weight_key} shape={tuple(w.shape)}")
        return False
    state_dict[weight_key] = _pad_or_truncate(w, new_in)
    if bias_key in state_dict:
        state_dict[bias_key] = state_dict[bias_key]  # keep bias unchanged
    print(f"[INFO] {label} updated: {weight_key} {old_in} -> {new_in}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Pad ActorCritic input layers for transfer learning.")
    parser.add_argument("--src", required=True, help="Source checkpoint (.pt)")
    parser.add_argument(
        "--dst",
        default=None,
        help="Destination checkpoint (.pt). Defaults to <src>_padded.pt",
    )
    parser.add_argument("--critic-in", type=int, default=None, help="Target critic input dimension")
    parser.add_argument("--actor-in", type=int, default=None, help="Target actor input dimension")
    args = parser.parse_args()

    src = args.src
    if not os.path.isfile(src):
        print(f"[ERROR] Source checkpoint not found: {src}")
        return 2

    dst = args.dst
    if dst is None:
        base, ext = os.path.splitext(src)
        dst = f"{base}_padded{ext or '.pt'}"

    ckpt = torch.load(src, map_location="cpu")
    if "model_state_dict" not in ckpt:
        print("[ERROR] checkpoint missing key: model_state_dict")
        return 3

    sd = ckpt["model_state_dict"]

    updated = False
    updated |= _maybe_update_layer(sd, "critic.0.weight", "critic.0.bias", args.critic_in, "critic")
    updated |= _maybe_update_layer(sd, "actor.0.weight", "actor.0.bias", args.actor_in, "actor")

    if not updated:
        print("[WARN] No layers updated. Check target dims or key names.")

    ckpt["model_state_dict"] = sd
    torch.save(ckpt, dst)
    print(f"[INFO] Saved: {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
