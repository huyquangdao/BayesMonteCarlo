EXPNAME="Main"
NEPOCHS=50
for i in 3
do
# CUDA_VISIBLE_DEVICES=5 accelerate launch --main_process_port 5050 --gpu_ids 5 --num_processes 1 run.py  \
#     --exp_name $EXPNAME \
#     --project_name MODPL \
#     --seed $i \
#     --scenario negotiation \
#     --log_dir logs \
#     --loggers terminal,file,wandb \
#     --datasets craigslist_bargain \
#     --models dpdp \
#     --prioritized_objective sl_ratio \
#     --gen_models llama3 \
#     --model_type llama3 \
#     --num_train_rl_epochs $NEPOCHS \
#     --metrics acc,prf1,sr,sl_ratio,fairness,avg_turn


# CUDA_VISIBLE_DEVICES=5 accelerate launch --main_process_port 5050 --gpu_ids 5 --num_processes 1 run.py  \
#     --exp_name $EXPNAME \
#     --project_name MODPL \
#     --seed $i \
#     --scenario negotiation \
#     --log_dir logs \
#     --loggers terminal,file,wandb \
#     --datasets craigslist_bargain \
#     --models dpdp \
#     --prioritized_objective fairness \
#     --gen_models llama3 \
#     --model_type llama3 \
#     --num_train_rl_epochs $NEPOCHS \
#     --metrics acc,prf1,sr,sl_ratio,fairness,avg_turn


# CUDA_VISIBLE_DEVICES=5 accelerate launch --main_process_port 5050 --gpu_ids 5 --num_processes 1 run.py  \
#     --exp_name $EXPNAME \
#     --project_name MODPL \
#     --seed $i \
#     --scenario negotiation \
#     --log_dir logs \
#     --loggers terminal,file,wandb \
#     --datasets craigslist_bargain \
#     --models dpdp \
#     --prioritized_objective sr \
#     --gen_models llama3 \
#     --model_type llama3 \
#     --num_train_rl_epochs $NEPOCHS \
#     --metrics acc,prf1,sr,sl_ratio,fairness,avg_turn


CUDA_VISIBLE_DEVICES=5 accelerate launch --main_process_port 5050 --gpu_ids 5 --num_processes 1 run.py  \
    --exp_name $EXPNAME \
    --project_name MODPL \
    --seed $i \
    --scenario negotiation \
    --log_dir logs \
    --loggers terminal,file \
    --datasets craigslist_bargain \
    --models dpdp \
    --prioritized_objective uniform \
    --gen_models llama3 \
    --model_type llama3 \
    --num_train_rl_epochs $NEPOCHS \
    --metrics acc,prf1,sr,sl_ratio,fairness,avg_turn
done