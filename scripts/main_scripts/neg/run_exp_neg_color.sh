EXPNAME="Main"
for i in 2
do
    CUDA_VISIBLE_DEVICES=3 accelerate launch --main_process_port 5050 --gpu_ids 3 --num_processes 1 run.py  \
        --exp_name $EXPNAME \
        --project_name ProactiveLLM \
        --seed $i \
        --scenario negotiation \
        --log_dir logs \
        --loggers terminal,file,wandb \
        --datasets craigslist_bargain \
        --models color \
        --gen_models llama3 \
        --is_so_game \
        --model_type llama3 \
        --metrics acc,prf1,sr,sl_ratio,avg_turn
done