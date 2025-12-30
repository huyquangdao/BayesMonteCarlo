EXPNAME="Main"
for i in 1 2 3
do
CUDA_VISIBLE_DEVICES=4 accelerate launch --main_process_port 3030 --gpu_ids 4 --num_processes 1 run.py  \
        --exp_name $EXPNAME \
        --project_name ProactiveLLM \
        --seed $i \
        --scenario negotiation \
        --log_dir logs \
        --loggers terminal,file,wandb \
        --datasets craigslist_bargain \
        --models pro_cot \
        --gen_models llama3 \
        --is_so_game \
        --use_persona \
        --model_type llama3 \
        --metrics acc,prf1,sr,sl_ratio,avg_turn,bleu_n,rouge_n,dist_n
done