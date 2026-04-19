#!/bin/bash
# DreamZero AGX Aloha Training Script
#
# Usage:
#   # Set your dataset path and output directory, then run:
#   bash scripts/train/agx_aloha_training.sh
#
# Prerequisites:
#   - AGX ALOHA dataset in LeRobot format at AGX_ALOHA_DATA_ROOT (state 14, action 14, 3 views: cam_high, cam_left_wrist, cam_right_wrist)
#     meta/embodiment.json must have "embodiment_tag": "agx_aloha"
#     modality: state (left_joint_pos, left_gripper_pos, right_joint_pos, right_gripper_pos),
#               action (same keys), video (cam_high, cam_left_wrist, cam_right_wrist), annotation.task
#   - Wan2.1-I2V-14B-480P weights (auto-downloaded or pre-downloaded from HuggingFace)
#     Download: huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir ./checkpoints/Wan2.1-I2V-14B-480P
#   - umt5-xxl tokenizer (auto-downloaded or pre-downloaded from HuggingFace)
#     Download: huggingface-cli download google/umt5-xxl --local-dir ./checkpoints/umt5-xxl
#   - DreamZero-AgiBot pretrained checkpoint (for loading LoRA weights before fine-tuning)
#     git clone https://huggingface.co/GEAR-Dreams/DreamZero-AgiBot ./checkpoints/DreamZero-AgiBot

export HYDRA_FULL_ERROR=1

# Weights & Biases: 默认离线（无需 wandb login / API key），日志在 $OUTPUT_DIR/wandb/offline-run-*
# 需要云端面板时：export WANDB_MODE=online 并先执行 wandb login；离线 run 之后可用 wandb sync 上传
export WANDB_MODE="${WANDB_MODE:-offline}"

# ============ CHANGE THESE VARIABLES ============
# Dataset path (LeRobot v2 + GEAR meta: state 14, action 14, 3 views cam_high / cam_left_wrist / cam_right_wrist)
AGX_ALOHA_DATA_ROOT=${AGX_ALOHA_DATA_ROOT:-"./data/agx_aloha_lerobot"}

# Output directory for training checkpoints
OUTPUT_DIR=${OUTPUT_DIR:-"./checkpoints/dreamzero_agx_aloha_lora_dz_pretrained_100k_folding"}

# Number of GPUs to use (default: all visible GPUs, so 4-GPU machines use 4 without setting NUM_GPUS)
if [ -z "${NUM_GPUS}" ]; then
  NUM_GPUS=$(nvidia-smi -L 2>/dev/null | wc -l)
fi
NUM_GPUS=${NUM_GPUS:-8}

# 终端完整日志（stdout+stderr）：不设则不写文件。设为 1 则写入 $OUTPUT_DIR/console.log；设为路径则写入该文件
TRAIN_CONSOLE_LOG=${TRAIN_CONSOLE_LOG:-}

# Model weight paths (download from HuggingFace if not already present)
WAN_CKPT_DIR=${WAN_CKPT_DIR:-"./checkpoints/Wan2.1-I2V-14B-480P"}
TOKENIZER_DIR=${TOKENIZER_DIR:-"./checkpoints/umt5-xxl"}
# =============================================

# ============ AUTO-DOWNLOAD WEIGHTS ============
if [ ! -d "$WAN_CKPT_DIR" ] || [ -z "$(ls -A "$WAN_CKPT_DIR" 2>/dev/null)" ]; then
    echo "Wan2.1-I2V-14B-480P not found at $WAN_CKPT_DIR. Downloading from HuggingFace..."
    huggingface-cli download Wan-AI/Wan2.1-I2V-14B-480P --local-dir "$WAN_CKPT_DIR"
fi

if [ ! -d "$TOKENIZER_DIR" ] || [ -z "$(ls -A "$TOKENIZER_DIR" 2>/dev/null)" ]; then
    echo "umt5-xxl tokenizer not found at $TOKENIZER_DIR. Downloading from HuggingFace..."
    huggingface-cli download google/umt5-xxl --local-dir "$TOKENIZER_DIR"
fi
# ================================================

# Validate dataset exists
if [ ! -d "$AGX_ALOHA_DATA_ROOT" ]; then
    echo "ERROR: AGX ALOHA dataset not found at $AGX_ALOHA_DATA_ROOT"
    echo "Set AGX_ALOHA_DATA_ROOT to your LeRobot-format AGX ALOHA dataset (meta/embodiment.json with embodiment_tag: agx_aloha)"
    exit 1
fi

# 用当前环境的 python 启动，避免 conda 里 torchrun 的 shebang 仍指向已删除的旧 env（如 dreamzero）
TRAIN_LOG_PATH=""
if [ -n "$TRAIN_CONSOLE_LOG" ]; then
    case "$TRAIN_CONSOLE_LOG" in
        1|true|TRUE|yes|YES) TRAIN_LOG_PATH="$OUTPUT_DIR/console.log" ;;
        *) TRAIN_LOG_PATH="$TRAIN_CONSOLE_LOG" ;;
    esac
    mkdir -p "$(dirname "$TRAIN_LOG_PATH")"
    echo "Logging console output to: $TRAIN_LOG_PATH"
fi

_run_train() {
python -m torch.distributed.run --nproc_per_node $NUM_GPUS --standalone groot/vla/experiment/experiment.py \
    report_to=wandb \
    data=dreamzero/agx_aloha_relative \
    wandb_project=dreamzero \
    train_architecture=lora \
    num_frames=33 \
    action_horizon=24 \
    num_views=3 \
    model=dreamzero/vla \
    model/dreamzero/action_head=wan_flow_matching_action_tf \
    model/dreamzero/transform=dreamzero_cotrain \
    num_frame_per_block=2 \
    num_action_per_block=24 \
    num_state_per_block=1 \
    seed=42 \
    training_args.learning_rate=1e-5 \
    training_args.deepspeed="groot/vla/configs/deepspeed/zero2.json" \
    save_steps=10000 \
    training_args.warmup_ratio=0.05 \
    output_dir=$OUTPUT_DIR \
    per_device_train_batch_size=1 \
    gradient_accumulation_steps=1 \
    max_steps=100000 \
    weight_decay=1e-5 \
    save_total_limit=10 \
    upload_checkpoints=false \
    bf16=true \
    tf32=true \
    eval_bf16=true \
    dataloader_pin_memory=false \
    dataloader_num_workers=1 \
    image_resolution_width=320 \
    image_resolution_height=176 \
    save_lora_only=true \
    max_chunk_size=4 \
    frame_seqlen=880 \
    save_strategy=steps \
    agx_aloha_data_root=$AGX_ALOHA_DATA_ROOT \
    dit_version=$WAN_CKPT_DIR \
    text_encoder_pretrained_path=$WAN_CKPT_DIR/models_t5_umt5-xxl-enc-bf16.pth \
    image_encoder_pretrained_path=$WAN_CKPT_DIR/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    vae_pretrained_path=$WAN_CKPT_DIR/Wan2.1_VAE.pth \
    tokenizer_path=$TOKENIZER_DIR \
    pretrained_model_path=./checkpoints/DreamZero-AgiBot \
    ++action_head_cfg.config.skip_component_loading=true \
    ++action_head_cfg.config.defer_lora_injection=true
}

if [ -n "$TRAIN_LOG_PATH" ]; then
    set -o pipefail
    _run_train 2>&1 | tee -a "$TRAIN_LOG_PATH"
    exit "${PIPESTATUS[0]}"
else
    _run_train
fi