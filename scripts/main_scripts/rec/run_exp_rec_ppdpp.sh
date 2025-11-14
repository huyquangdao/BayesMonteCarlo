#
#!/bin/bash
domains=("movie" "music" "poi")
expname="Main"
for i in 1 2 3
do
    for domain in "${domains[@]}"
    do
    CUDA_VISIBLE_DEVICES=2 accelerate launch --main_process_port 2020 --gpu_ids 2 --num_processes 1 run.py  \
        --exp_name $expname \
        --project_name MODPL \
        --seed $i \
        --scenario recommendation \
        --log_dir logs \
        --loggers terminal,wandb,file \
        --datasets durecdial \
        --models ppdpp \
        --domain $domain \
        --is_so_game \
        --use_persona \
        --num_train_rl_epochs 10 \
        --gen_models llama3 \
        --model_type llama3 \
        --metrics acc,prf1,sr,total_reward,avg_turn
    done
done