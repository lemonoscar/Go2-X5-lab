#!/bin/bash
set -euo pipefail
# Test with arm camera looking FORWARD (recommended for grasping)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Find IsaacLab root
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

export PYTHONPATH="${ROOT_DIR}/source/robot_lab:${PYTHONPATH:-}"

# Use isaaclab.sh to run the script (not direct python!)
exec "${ISAACLAB_ROOT}/isaaclab.sh" \
  -p "${ROOT_DIR}/scripts/visualize_camera.py" \
  --dog_cam_pos 0.30 0.0 0.16 \
  --dog_cam_rot -0.3799 0.5963 0.5963 -0.3799 \
  --arm_cam_pos 0.08657 0.0 0.0 \
  --arm_cam_rot 0 0 0 1 \
  --save_images \
  "$@"
