#
#!/bin/bash
domains=("movie" "music" "poi")
expname="Main"
for i in 2 3
do
    for domain in "${domains[@]}"
    do
    CUDA_VISIBLE_DEVICES=4 accelerate launch --main_process_port 4040 --gpu_ids 4 --num_processes 1 run.py  \
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
        --use_persona \
        --gen_models llama3 \
        --model_type llama3 \
        --metrics acc,prf1,sr,total_reward,avg_turn
    done
done