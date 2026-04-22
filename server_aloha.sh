cd /data/user/wsong890/shuaizhou/dreamzero
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/torch/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH

export HF_HOME=/data/user/wsong890/shuaizhou/dreamzero/checkpoints/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1


CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 \
  socket_test_optimized_AR_aloha.py \
  --model-path checkpoints/agx_aloha_h100_run1/checkpoint-50000 \
  --port 8766 \
  --enable-dit-cache