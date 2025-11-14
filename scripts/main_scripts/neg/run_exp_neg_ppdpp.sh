EXPNAME="Main"
for i in 1 2 3
do
    CUDA_VISIBLE_DEVICES=2 accelerate launch --main_process_port 2020 --gpu_ids 2 --num_processes 1 run.py  \
        --exp_name $EXPNAME \
        --project_name ProactiveLLM \
        --seed $i \
        --scenario negotiation \
        --log_dir logs \
        --loggers terminal,file,wandb \
        --datasets craigslist_bargain \
        --models ppdpp \
        --gen_models qwen \
        --is_so_game \
        --num_train_rl_epochs 10 \
        --model_type qwen \
        --metrics acc,prf1,sr,sl_ratio,avg_turn
done