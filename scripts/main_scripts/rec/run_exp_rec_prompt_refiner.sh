#
#!/bin/bash
domains=("movie")
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
        --loggers terminal,file,wandb \
        --datasets durecdial \
        --models prompt_refiner \
        --domain $domain \
        --is_so_game \
        --rewrite_action \
        --num_train_rl_epochs 6 \
        --use_persona \
        --gen_models llama3 \
        --model_type llama3 \
        --metrics acc,prf1,sr,total_reward,avg_turn
    done
done