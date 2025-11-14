#
#!/bin/bash
domains=("movie" "music" "poi")
expname="Main"
for i in 1 2 3
do
    for domain in "${domains[@]}"
    do
    CUDA_VISIBLE_DEVICES=3 accelerate launch --main_process_port 3030 --gpu_ids 3 --num_processes 1 run.py  \
        --exp_name $expname \
        --project_name MODPL \
        --seed $i \
        --scenario recommendation \
        --log_dir logs \
        --loggers terminal,wandb,file \
        --datasets durecdial \
        --models icl_aif \
        --domain $domain \
        --is_so_game \
        --use_persona \
        --gen_models qwen \
        --model_type qwen \
        --metrics acc,prf1,sr,total_reward,avg_turn
    done
done