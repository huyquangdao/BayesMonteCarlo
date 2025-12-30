EXPNAME="Main"
for i in 42
do
    CUDA_VISIBLE_DEVICES=6 accelerate launch --main_process_port 62 --gpu_ids 6 --num_processes 1 run.py  \
        --exp_name $EXPNAME \
        --project_name ProactiveLLM \
        --seed $i \
        --scenario negotiation \
        --log_dir logs \
        --loggers terminal,file,wandb \
        --datasets craigslist_bargain \
        --models ppdpp \
        --gen_models llama3 \
        --is_so_game \
        --use_persona \
        --num_train_rl_epochs 5 \
        --model_type llama3 \
        --metrics acc,prf1,sr,sl_ratio,avg_turn
done