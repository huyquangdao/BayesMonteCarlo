EXPNAME="Main"
for i in 1
do
CUDA_VISIBLE_DEVICES=3 accelerate launch --main_process_port 76 --gpu_ids 2 --num_processes 1 run.py  \
        --exp_name $EXPNAME \
        --project_name ProactiveLLM \
        --seed $i \
        --scenario persuation \
        --log_dir logs \
        --loggers terminal,file,wandb \
        --datasets p4g \
        --models trip \
        --gen_models llama3 \
        --is_so_game \
        --use_persona \
        --num_train_rl_epochs 5 \
        --model_type llama3 \
        --analysis_bayes_monte_carlo \
        --metrics acc,prf1,sr,total_reward,avg_turn
done
