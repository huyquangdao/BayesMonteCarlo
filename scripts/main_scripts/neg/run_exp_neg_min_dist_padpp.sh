EXPNAME="Main"
for i in  3 4 5
do
CUDA_VISIBLE_DEVICES=4 accelerate launch --main_process_port 2020 --gpu_ids 4 --num_processes 1 run.py  \
    --exp_name $EXPNAME \
    --project_name MODPL \
    --seed $i \
    --scenario negotiation \
    --log_dir logs \
    --loggers terminal,file,wandb \
    --datasets craigslist_bargain \
    --models min_dist_padpp \
    --gen_models llama3 \
    --model_type llama3 \
    --metrics acc,prf1,sr,sl_ratio,fairness,avg_turn

CUDA_VISIBLE_DEVICES=4 accelerate launch --main_process_port 2020 --gpu_ids 4 --num_processes 1 run.py  \
    --exp_name $EXPNAME \
    --project_name MODPL \
    --seed $i \
    --scenario negotiation \
    --log_dir logs \
    --loggers terminal,file,wandb \
    --datasets craigslist_bargain \
    --models min_dist_padpp \
    --gen_models llama3 \
    --model_type llama3 \
    --test_phase \
    --prioritized_objective sl_ratio \
    --metrics acc,prf1,sr,sl_ratio,fairness,avg_turn

CUDA_VISIBLE_DEVICES=4 accelerate launch --main_process_port 2020 --gpu_ids 4 --num_processes 1 run.py  \
    --exp_name $EXPNAME \
    --project_name MODPL \
    --seed $i \
    --scenario negotiation \
    --log_dir logs \
    --loggers terminal,file,wandb \
    --datasets craigslist_bargain \
    --models min_dist_padpp \
    --gen_models llama3 \
    --model_type llama3 \
    --test_phase \
    --prioritized_objective fairness \
    --metrics acc,prf1,sr,sl_ratio,fairness,avg_turn


CUDA_VISIBLE_DEVICES=4 accelerate launch --main_process_port 2020 --gpu_ids 4 --num_processes 1 run.py  \
    --exp_name $EXPNAME \
    --project_name MODPL \
    --seed $i \
    --scenario negotiation \
    --log_dir logs \
    --loggers terminal,file,wandb \
    --datasets craigslist_bargain \
    --models min_dist_padpp \
    --gen_models llama3 \
    --model_type llama3 \
    --test_phase \
    --prioritized_objective sr \
    --metrics acc,prf1,sr,sl_ratio,fairness,avg_turn

done