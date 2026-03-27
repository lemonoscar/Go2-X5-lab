#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

if [[ -z "${ISAACLAB_ROOT:-}" ]]; then
  for candidate in \
    "${ROOT_DIR}/../IsaacLab" \
    "${ROOT_DIR}/../../IsaacLab" \
    "/home/lemon/Issac/IsaacLab"
  do
    if [[ -x "${candidate}/isaaclab.sh" ]]; then
      ISAACLAB_ROOT="${candidate}"
      break
    fi
  done
fi

if [[ -z "${ISAACLAB_ROOT:-}" || ! -x "${ISAACLAB_ROOT}/isaaclab.sh" ]]; then
  echo "Set ISAACLAB_ROOT to a valid IsaacLab checkout containing isaaclab.sh." >&2
  exit 1
fi

export PYTHONPATH="${ROOT_DIR}/source/robot_lab:${PYTHONPATH:-}"
if [[ -z "${GO2_X5_LOW_LEVEL_POLICY_PATH:-}" ]]; then
  for candidate in \
    "${ROOT_DIR}/logs/rsl_rl/go2_x5_foundation_flat/2026-03-12_12-11-27/exported/policy.pt" \
    "${ROOT_DIR}/logs/rsl_rl/go2_x5_flat/2026-02-06_00-39-51/exported/policy.pt"
  do
    if [[ -f "${candidate}" ]]; then
      export GO2_X5_LOW_LEVEL_POLICY_PATH="${candidate}"
      break
    fi
  done
fi

if [[ -z "${GO2_X5_LOW_LEVEL_POLICY_PATH:-}" || ! -f "${GO2_X5_LOW_LEVEL_POLICY_PATH}" ]]; then
  echo "Set GO2_X5_LOW_LEVEL_POLICY_PATH to a valid frozen locomotion policy." >&2
  exit 1
fi

exec "${ISAACLAB_ROOT}/isaaclab.sh" \
  -p "${ROOT_DIR}/scripts/tools/view_ground_pick.py" \
  --device cuda:0 \
  --num_envs 1 \
  --enable_cameras \
  "$@"
