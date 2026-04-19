task_name=stack_bowls
ckpt_dir=$HOME/train0820/test1
ckpt_name=policy_best.ckpt

python act/inference.py \
 --task_name $task_name \
 --ckpt_dir $ckpt_dir \
 --ckpt_name $ckpt_name