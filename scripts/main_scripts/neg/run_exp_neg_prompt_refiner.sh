EXPNAME="Main"
for i in 1 2 3
do
        CUDA_VISIBLE_DEVICES=4 accelerate launch --main_process_port 4040 --gpu_ids 4 --num_processes 1 run.py  \
        --exp_name $EXPNAME \
        --project_name ProactiveLLM \
        --seed $i \
        --scenario negotiation \
        --log_dir logs \
        --loggers terminal,file,wandb \
        --datasets craigslist_bargain \
        --models prompt_refiner \
        --gen_models llama3 \
        --is_so_game \
        --rewrite_action \
        --model_type llama3 \
        --metrics acc,prf1,sr,sl_ratio,avg_turn
done