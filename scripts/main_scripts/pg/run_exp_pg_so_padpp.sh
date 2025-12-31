EXPNAME="Main"
for i in 1
do
CUDA_VISIBLE_DEVICES=6 accelerate launch --main_process_port 69 --gpu_ids 7 --num_processes 1 run.py  \
        --exp_name $EXPNAME \
        --project_name ProactiveLLM \
        --seed $i \
        --scenario persuation \
        --log_dir logs \
        --loggers terminal,file,wandb \
        --datasets p4g \
        --models so_padpp \
        --gen_models llama3 \
        --num_train_rl_epochs 10 \
        --is_so_game \
        --use_persona \
        --model_type llama3 \
        --metrics acc,prf1,sr,total_reward,avg_turn
done
