EXPNAME="Main"
for i in 1 3
do
    CUDA_VISIBLE_DEVICES=7 accelerate launch --main_process_port 7070 --gpu_ids 7 --num_processes 1 run.py  \
        --exp_name $EXPNAME \
        --project_name ProactiveLLM \
        --seed $i \
        --scenario emotional_support \
        --log_dir logs \
        --loggers terminal,file,wandb \
        --datasets es_conv \
        --models ppdpp \
        --gen_models qwen \
        --is_so_game \
        --num_train_rl_epochs 10 \
        --model_type qwen \
        --metrics acc,prf1,sr,total_reward,avg_turn
done