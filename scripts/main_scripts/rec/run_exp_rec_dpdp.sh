#
#!/bin/bash
domains=("poi")
expname="Main"
for i in 3 4 5
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
        --models dpdp \
        --domain $domain \
        --prioritized_objective user_reward \
        --gen_models llama3 \
        --model_type llama3 \
        --metrics acc,prf1,sr,user_reward,item_freq,avg_turn

    CUDA_VISIBLE_DEVICES=2 accelerate launch --main_process_port 2020 --gpu_ids 2 --num_processes 1 run.py  \
        --exp_name $expname \
        --project_name MODPL \
        --seed $i \
        --scenario recommendation \
        --log_dir logs \
        --loggers terminal,wandb,file \
        --datasets durecdial \
        --models dpdp \
        --domain $domain \
        --prioritized_objective item_freq \
        --gen_models llama3 \
        --model_type llama3 \
        --metrics acc,prf1,sr,user_reward,item_freq,avg_turn

    CUDA_VISIBLE_DEVICES=2 accelerate launch --main_process_port 2020 --gpu_ids 2 --num_processes 1 run.py  \
        --exp_name $expname \
        --project_name MODPL \
        --seed $i \
        --scenario recommendation \
        --log_dir logs \
        --loggers terminal,wandb,file \
        --datasets durecdial \
        --models dpdp \
        --domain $domain \
        --prioritized_objective uniform \
        --gen_models llama3 \
        --model_type llama3 \
        --metrics acc,prf1,sr,user_reward,item_freq,avg_turn
    done
done