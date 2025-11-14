#
#!/bin/bash
domains=("movie" "music" "poi")
EXPNAME="Main"
NPREFERENCES=32
for i in 3 4 5
do
        for domain in "${domains[@]}"
        do
        CUDA_VISIBLE_DEVICES=4 accelerate launch --main_process_port 6060 --gpu_ids 4 --num_processes 1 run.py  \
            --exp_name $EXPNAME \
            --project_name MODPL \
            --seed $i \
            --scenario recommendation \
            --log_dir logs \
            --loggers terminal,wandb,file \
            --datasets durecdial \
            --models ct_modpl \
            --domain $domain \
            --use_persona \
            --use_gpi 1 \
            --n_preferences $NPREFERENCES \
            --gen_models llama3 \
            --model_type llama3 \
            --metrics acc,prf1,sr,user_reward,item_freq,avg_turn

        CUDA_VISIBLE_DEVICES=4 accelerate launch --main_process_port 6060 --gpu_ids 4 --num_processes 1 run.py  \
            --exp_name $EXPNAME \
            --project_name MODPL \
            --seed $i \
            --scenario recommendation \
            --log_dir logs \
            --loggers terminal,file,wandb \
            --datasets durecdial \
            --models ct_modpl \
            --test_phase \
            --use_persona \
            --domain $domain \
            --use_gpi 1 \
            --n_preferences $NPREFERENCES \
            --prioritized_objective user_reward \
            --gen_models llama3 \
            --model_type llama3 \
            --metrics acc,prf1,sr,user_reward,item_freq,avg_turn

        CUDA_VISIBLE_DEVICES=4 accelerate launch --main_process_port 6060 --gpu_ids 4 --num_processes 1 run.py  \
            --exp_name $EXPNAME \
            --project_name MODPL \
            --seed $i \
            --scenario recommendation \
            --log_dir logs \
            --loggers terminal,file,wandb \
            --datasets durecdial \
            --models ct_modpl \
            --test_phase \
            --use_persona \
            --domain $domain \
            --use_gpi 1 \
            --n_preferences $NPREFERENCES \
            --prioritized_objective item_freq \
            --gen_models llama3 \
            --model_type llama3 \
            --metrics acc,prf1,sr,user_reward,item_freq,avg_turn
        done
done