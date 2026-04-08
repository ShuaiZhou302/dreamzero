#!/usr/bin/env bash
set -euo pipefail

# Input: ALOHA-style HDF5 episodes
INPUT_DIR="/data/user/wsong890/shuaizhou/dreamzero/cobot_data/cooking2"

# Output: LeRobot v2 dataset root
OUTPUT_DIR="/data/user/wsong890/shuaizhou/dreamzero/cobot_data/cobot_dataset"

# Optional: change this if your dataset FPS is different
FPS="30"

# Optional: for quick test, set MAX_EPISODES=2 (or any small number)
# Leave empty to convert all episodes in INPUT_DIR
MAX_EPISODES="${MAX_EPISODES:-2}"

echo "[1/2] Convert HDF5 -> LeRobot v2 layout"
if [[ -n "${MAX_EPISODES}" ]]; then
  python "/data/user/wsong890/shuaizhou/dreamzero/cobot_data/cobot_lerobotv2_convert.py" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --fps "${FPS}" \
    --max-episodes "${MAX_EPISODES}" \
    --force
else
  python "/data/user/wsong890/shuaizhou/dreamzero/cobot_data/cobot_lerobotv2_convert.py" \
    --input-dir "${INPUT_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --fps "${FPS}" \
    --force
fi

echo "[2/2] Convert LeRobot v2 metadata -> DreamZero GEAR metadata"
# python "/data/user/wsong890/shuaizhou/dreamzero/scripts/data/convert_lerobot_to_gear.py" \
#   --dataset-path "${OUTPUT_DIR}" \
#   --embodiment-tag xdof \
#   --state-keys '{"joint_pos":[0,13],"gripper_pos":[13,14]}' \
#   --action-keys '{"joint_pos":[0,13],"gripper_pos":[13,14]}' \
#   --relative-action-keys joint_pos gripper_pos \
#   --task-key annotation.task \
#   --force

# echo "Done. Converted dataset is at: ${OUTPUT_DIR}"
