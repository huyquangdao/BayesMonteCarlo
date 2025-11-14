EXPNAME="Main"
llms=("llama3")
for i in 1 2 3
do
        for llm in "${llms[@]}"
        do
        CUDA_VISIBLE_DEVICES=2 accelerate launch --main_process_port 2020 --gpu_ids 2 --num_processes 1 run.py  \
                --exp_name $EXPNAME \
                --project_name ProactiveLLM \
                --seed $i \
                --scenario emotional_support \
                --log_dir logs \
                --loggers terminal,file,wandb \
                --datasets es_conv \
                --models prompt_refiner \
                --gen_models $llm \
                --is_so_game \
                --num_train_rl_epochs 6 \
                --rewrite_action \
                --model_type $llm \
                --metrics acc,prf1,sr,avg_turn,total_reward
        done
done