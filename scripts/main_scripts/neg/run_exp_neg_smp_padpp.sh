EXPNAME="Main"
for i in  4
do
CUDA_VISIBLE_DEVICES=3 accelerate launch --main_process_port 2020 --gpu_ids 3 --num_processes 1 run.py  \
    --exp_name $EXPNAME \
    --project_name MODPL \
    --seed $i \
    --scenario negotiation \
    --log_dir logs \
    --loggers terminal \
    --datasets craigslist_bargain \
    --models smp_padpp \
    --gen_models llama3 \
    --model_type llama3 \
    --metrics acc,prf1,sr,sl_ratio,fairness,avg_turn

CUDA_VISIBLE_DEVICES=3 accelerate launch --main_process_port 2020 --gpu_ids 3 --num_processes 1 run.py  \
    --exp_name $EXPNAME \
    --project_name MODPL \
    --seed $i \
    --scenario negotiation \
    --log_dir logs \
    --loggers terminal,file,wandb \
    --datasets craigslist_bargain \
    --models smp_padpp \
    --test_phase \
    --prioritized_objective sl_ratio \
    --gen_models llama3 \
    --model_type llama3 \
    --metrics acc,prf1,sr,sl_ratio,fairness,avg_turn

CUDA_VISIBLE_DEVICES=3 accelerate launch --main_process_port 2020 --gpu_ids 3 --num_processes 1 run.py  \
    --exp_name $EXPNAME \
    --project_name MODPL \
    --seed $i \
    --scenario negotiation \
    --log_dir logs \
    --loggers terminal,file,wandb \
    --datasets craigslist_bargain \
    --models smp_padpp \
    --test_phase \
    --prioritized_objective fairness \
    --gen_models llama3 \
    --model_type llama3 \
    --metrics acc,prf1,sr,sl_ratio,fairness,avg_turn

CUDA_VISIBLE_DEVICES=3 accelerate launch --main_process_port 2020 --gpu_ids 3 --num_processes 1 run.py  \
    --exp_name $EXPNAME \
    --project_name MODPL \
    --seed $i \
    --scenario negotiation \
    --log_dir logs \
    --loggers terminal,file,wandb \
    --datasets craigslist_bargain \
    --models smp_padpp \
    --test_phase \
    --prioritized_objective sr \
    --gen_models llama3 \
    --model_type llama3 \
    --metrics acc,prf1,sr,sl_ratio,fairness,avg_turn    
done