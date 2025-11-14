#
#!/bin/bash
domains=("movie")
EXPNAME="Main"
NEPOCHS=50
NPREFERENCES=32
for i in 3
do
        for domain in "${domains[@]}"
        do
        # CUDA_VISIBLE_DEVICES=2 accelerate launch --main_process_port 6060 --gpu_ids 2 --num_processes 1 run.py  \
        #     --exp_name $EXPNAME \
        #     --project_name MODPL \
        #     --seed $i \
        #     --scenario recommendation \
        #     --log_dir logs \
        #     --loggers terminal \
        #     --datasets durecdial \
        #     --models ct_modpl \
        #     --domain $domain \
        #     --use_persona \
        #     --use_gpi 1 \
        #     --n_preferences $NPREFERENCES \
        #     --num_train_rl_epochs $NEPOCHS \
        #     --gen_models llama3 \
        #     --model_type llama3 \
        #     --metrics acc,prf1,sr,user_reward,item_freq,avg_turn

        CUDA_VISIBLE_DEVICES=2 accelerate launch --main_process_port 6060 --gpu_ids 2 --num_processes 1 run.py  \
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
            --num_train_rl_epochs $NEPOCHS \
            --prioritized_objective user_reward \
            --gen_models llama3 \
            --model_type llama3 \
            --metrics acc,prf1,sr,user_reward,item_freq,avg_turn

        CUDA_VISIBLE_DEVICES=2 accelerate launch --main_process_port 6060 --gpu_ids 2 --num_processes 1 run.py  \
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
            --num_train_rl_epochs $NEPOCHS \
            --prioritized_objective item_freq \
            --gen_models llama3 \
            --model_type llama3 \
            --metrics acc,prf1,sr,user_reward,item_freq,avg_turn
        
        CUDA_VISIBLE_DEVICES=2 accelerate launch --main_process_port 6060 --gpu_ids 2 --num_processes 1 run.py  \
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
            --num_train_rl_epochs $NEPOCHS \
            --prioritized_objective uniform \
            --gen_models llama3 \
            --model_type llama3 \
            --metrics acc,prf1,sr,user_reward,item_freq,avg_turn

        done
done