#!/bin/bash

if [ "$1" == "sim" ]; then
    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --env_id SimEmbodiedAnt \
        --learning_starts 2000 \
        --batch_size 256 \
        --task_type forward \
        --model_path ../../sim/assets/embodied_mujoco_ant.xml \
        --capture_video \
        --exp_name  sim_learning_to_walk
fi

# Learn on hardware.
if [ "$1" == "hw" ]; then
    python3 sac_cleanrl.py \
        --render_mode rgb_array \
        --dt 0.12 \
        --seed 1 \
        --env_id HwEmbodiedAnt \
        --hw_config ../../embodied_ant_env/ant12.json \
        --learning_starts 2000 \
        --task_type forward \
        --exp_name ant_forward \
        # --weights_path ...
        # --capture_video \
        # --eval True
fi