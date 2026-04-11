left_joint_pos: [0,6]
left_gripper_pos: [6,7]
right_joint_pos: [7,13]
right_gripper_pos: [13,14]

TODO:

new data that i collected has a new thing in its hdf5 call task decription, it could be used as annotation.task for each episode

potential problem:

1.The gripper value is not 0 to 1, but 0 to 0.1, where 0.1 means fully open, and 0 means closed.

2.in /data/user/wsong890/shuaizhou/dreamzero/groot/vla/configs/data/dreamzero/base_48_wan_fine_aug_relative.yaml，i set the eval_delta_indices: [-3, -2, -1, 0] for out video like agibot instead of [0] like yam, i don't know if this is gonna cause problem. 

3. in training 
we set: 
    per_device_train_batch_size=1 \
    gradient_accumulation_steps=4 \
originally it is 
    per_device_train_batch_size=4

we didn't uses it because it will CUDA OUT OF MEMORY!!!
