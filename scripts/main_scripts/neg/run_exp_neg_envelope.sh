use_gpi=1
NEPOCHS=50
EXPNAME="Main"
npres=64
for i in 5
do
CUDA_VISIBLE_DEVICES=7 accelerate launch --main_process_port 8080 --gpu_ids 7 --num_processes 1 run.py  \
    --exp_name $EXPNAME \
    --project_name MODPL \
    --seed $i \
    --scenario negotiation \
    --log_dir logs \
    --loggers terminal,file,wandb \
    --datasets craigslist_bargain \
    --models envelope \
    --gen_models llama3 \
    --model_type llama3 \
    --num_train_rl_epochs $NEPOCHS \
    --n_preferences $npres \
    --use_gpi $use_gpi \
    --metrics acc,prf1,sr,sl_ratio,fairness,avg_turn

CUDA_VISIBLE_DEVICES=7 accelerate launch --main_process_port 8080 --gpu_ids 7 --num_processes 1 run.py  \
    --exp_name $EXPNAME \
    --project_name MODPL \
    --seed $i \
    --scenario negotiation \
    --log_dir logs \
    --loggers terminal,file,wandb \
    --datasets craigslist_bargain \
    --models envelope \
    --test_phase \
    --prioritized_objective sl_ratio \
    --gen_models llama3 \
    --model_type llama3 \
    --num_train_rl_epochs $NEPOCHS \
    --n_preferences $npres \
    --use_gpi $use_gpi \
    --metrics acc,prf1,sr,sl_ratio,fairness,avg_turn

CUDA_VISIBLE_DEVICES=7 accelerate launch --main_process_port 8080 --gpu_ids 7 --num_processes 1 run.py  \
    --exp_name $EXPNAME \
    --project_name MODPL \
    --seed $i \
    --scenario negotiation \
    --log_dir logs \
    --loggers terminal,file,wandb \
    --datasets craigslist_bargain \
    --models envelope \
    --test_phase \
    --prioritized_objective fairness \
    --gen_models llama3 \
    --model_type llama3 \
    --num_train_rl_epochs $NEPOCHS \
    --n_preferences $npres \
    --use_gpi $use_gpi \
    --metrics acc,prf1,sr,sl_ratio,fairness,avg_turn

CUDA_VISIBLE_DEVICES=7 accelerate launch --main_process_port 8080 --gpu_ids 7 --num_processes 1 run.py  \
    --exp_name $EXPNAME \
    --project_name MODPL \
    --seed $i \
    --scenario negotiation \
    --log_dir logs \
    --loggers terminal,file,wandb \
    --datasets craigslist_bargain \
    --models envelope \
    --test_phase \
    --prioritized_objective sr \
    --gen_models llama3 \
    --model_type llama3 \
    --num_train_rl_epochs $NEPOCHS \
    --n_preferences $npres \
    --use_gpi $use_gpi \
    --metrics acc,prf1,sr,sl_ratio,fairness,avg_turn    
done