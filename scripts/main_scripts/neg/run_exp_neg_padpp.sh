use_gpi=1
NEPOCHS=50
EXPNAME="Main"
npres=64
for i in 3
do
# CUDA_VISIBLE_DEVICES=7 accelerate launch --main_process_port 2020 --gpu_ids 7 --num_processes 1 run.py  \
#     --exp_name $EXPNAME \
#     --project_name MODPL \
#     --seed $i \
#     --scenario negotiation \
#     --log_dir logs \
#     --loggers terminal \
#     --datasets craigslist_bargain \
#     --models ct_modpl \
#     --gen_models llama3 \
#     --model_type llama3 \
#     --num_train_rl_epochs $NEPOCHS \
#     --n_preferences $npres \
#     --use_gpi $use_gpi \
#     --metrics acc,prf1,sr,sl_ratio,fairness,avg_turn

# CUDA_VISIBLE_DEVICES=7 accelerate launch --main_process_port 2020 --gpu_ids 7 --num_processes 1 run.py  \
#     --exp_name $EXPNAME \
#     --project_name MODPL \
#     --seed $i \
#     --scenario negotiation \
#     --log_dir logs \
#     --loggers terminal \
#     --datasets craigslist_bargain \
#     --models ct_modpl \
#     --test_phase \
#     --prioritized_objective sl_ratio \
#     --gen_models llama3 \
#     --model_type llama3 \
#     --num_train_rl_epochs $NEPOCHS \
#     --n_preferences $npres \
#     --use_gpi $use_gpi \
#     --metrics acc,prf1,sr,sl_ratio,fairness,avg_turn

# CUDA_VISIBLE_DEVICES=7 accelerate launch --main_process_port 2020 --gpu_ids 7 --num_processes 1 run.py  \
#     --exp_name $EXPNAME \
#     --project_name MODPL \
#     --seed $i \
#     --scenario negotiation \
#     --log_dir logs \
#     --loggers terminal,file,wandb \
#     --datasets craigslist_bargain \
#     --models ct_modpl \
#     --test_phase \
#     --prioritized_objective fairness \
#     --gen_models llama3 \
#     --model_type llama3 \
#     --num_train_rl_epochs $NEPOCHS \
#     --n_preferences $npres \
#     --use_gpi $use_gpi \
#     --metrics acc,prf1,sr,sl_ratio,fairness,avg_turn

CUDA_VISIBLE_DEVICES=7 accelerate launch --main_process_port 2020 --gpu_ids 7 --num_processes 1 run.py  \
    --exp_name $EXPNAME \
    --project_name MODPL \
    --seed $i \
    --scenario negotiation \
    --log_dir logs \
    --loggers terminal,file \
    --datasets craigslist_bargain \
    --models ct_modpl \
    --test_phase \
    --prioritized_objective uniform \
    --gen_models llama3 \
    --model_type llama3 \
    --num_train_rl_epochs $NEPOCHS \
    --n_preferences $npres \
    --use_gpi $use_gpi \
    --metrics acc,prf1,sr,sl_ratio,fairness,avg_turn    
done
