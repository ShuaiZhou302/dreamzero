#!/usr/bin/env bash
set -euo pipefail

# Input: ALOHA-style HDF5 episodes
# OLD (flat layout, single global annotation.task from folder basename):
# INPUT_DIR="/data/user/wsong890/shuaizhou/dreamzero/cobot_data/cooking"
# NEW: nested task folders + per-episode task from HDF5 root attr `task_description`
INPUT_DIR="/data/user/wsong890/shuaizhou/dreamzero/cobot_data/30min_data"

# Output: LeRobot v2 dataset root
OUTPUT_DIR="/data/user/wsong890/shuaizhou/dreamzero/cobot_data/cobot_dataset"

# Optional: change this if your dataset FPS is different
FPS="30"

# Per-episode language instructions:
# - Leave TASK_TEXT unset/empty: cobot_lerobotv2_convert.py reads each episode's HDF5 root
#   attributes in order (task_description, language_instruction, instruction, task, task_name)
#   and writes that string to annotation.task for every frame of THAT episode only.
# - Set TASK_TEXT only if you want to FORCE the same string for all episodes (disables HDF5 attrs).
# OLD (optional global override — same text for every episode):
# TASK_TEXT="${TASK_TEXT:-}"
TASK_TEXT="${TASK_TEXT:-}"

# Optional: quick test with a small number of episodes; leave empty to convert all
# OLD default was 2 for smoke test:
# MAX_EPISODES="${MAX_EPISODES:-2}"
MAX_EPISODES="${MAX_EPISODES:-}"

echo "[1/2] Convert HDF5 -> LeRobot v2 layout"
TASK_ARG=()
if [[ -n "${TASK_TEXT}" ]]; then
  TASK_ARG=(--task-text "${TASK_TEXT}")
fi

if [[ -n "${MAX_EPISODES}" ]]; then
  python "/data/user/wsong890/shuaizhou/dreamzero/cobot_data/cobot_lerobotv2_convert.py" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --fps "${FPS}" \
    "${TASK_ARG[@]}" \
    --max-episodes "${MAX_EPISODES}" \
    --force
else
  python "/data/user/wsong890/shuaizhou/dreamzero/cobot_data/cobot_lerobotv2_convert.py" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --fps "${FPS}" \
    "${TASK_ARG[@]}" \
    --force
fi

echo "[2/2] Convert LeRobot v2 metadata -> DreamZero GEAR metadata"

python "/data/user/wsong890/shuaizhou/dreamzero/scripts/data/convert_lerobot_to_gear.py" \
  --dataset-path "${OUTPUT_DIR}" \
  --embodiment-tag agx_aloha \
  --state-keys '{"left_joint_pos":[0,6],"left_gripper_pos":[6,7],"right_joint_pos":[7,13],"right_gripper_pos":[13,14]}' \
  --action-keys '{"left_joint_pos":[0,6],"left_gripper_pos":[6,7],"right_joint_pos":[7,13],"right_gripper_pos":[13,14]}' \
  --relative-action-keys left_joint_pos left_gripper_pos right_joint_pos right_gripper_pos \
  --task-key annotation.task \
  --force

echo "Done. Converted dataset is at: ${OUTPUT_DIR}"
