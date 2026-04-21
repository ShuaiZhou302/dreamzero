export WANDB_API_KEY=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
NEWID=$(python -c "import wandb; print(wandb.util.generate_id())")
echo "Using new W&B run id: ${NEWID}"
python -m wandb sync /data/user/wsong890/shuaizhou/dreamzero/checkpoints/agx_aloha_h100_run1/wandb/latest-run --id "${NEWID}"
# NEWID=$(conda run -n dz_shuai python -c "import wandb; print(wandb.util.generate_id())")
# echo "Using new W&B run id: ${NEWID}"
# conda run -n dz_shuai python -m wandb sync /data/user/wsong890/shuaizhou/dreamzero/checkpoints/agx_aloha_h100_run1/wandb/latest-run --id "${NEWID}"
unset WANDB_API_KEY


#fist time run
# export WANDB_API_KEY=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
# NEWID=$(python -c "import wandb; print(wandb.util.generate_id())")
# echo "$NEWID"
# python -m wandb sync /data/user/wsong890/shuaizhou/dreamzero/checkpoints/agx_aloha_h100_run1/wandb/latest-run --id "$NEWID"
# unset WANDB_API_KEY