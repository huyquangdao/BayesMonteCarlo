EXPNAME="Main"
for i in 1 2 3
do
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 70 --gpu_ids 0 --num_processes 1 run.py  \
        --exp_name $EXPNAME \
        --project_name ProactiveLLM \
        --seed $i \
        --scenario persuation \
        --log_dir logs \
        --loggers terminal,file,wandb \
        --datasets p4g \
        --models prompt_refiner \
        --gen_models llama3 \
        --num_train_rl_epochs 6 \
        --rewrite_action \
        --is_so_game \
        --model_type llama3 \
        --metrics acc,prf1,sr,total_reward,avg_turn
done
