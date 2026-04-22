source /opt/ros/noetic/setup.bash
export PYTHONPATH="$PWD:${PYTHONPATH}"
export PYTHONUNBUFFERED=1

SERVER_HOST="${SERVER_HOST:-127.0.0.1}"
SERVER_PORT="${SERVER_PORT:-8766}"

# Better grammar than "took the vegetables in the pan out"
PROMPT="${PROMPT:-Take the vegetables out of the pan.}"

python -u /home/agilex/cobot_magic/aloha-devel/dreamzero/deploy/dz_aloha_infer.py \
  --server-host "${SERVER_HOST}" \
  --server-port "${SERVER_PORT}" \
  --prompt "${PROMPT}"
