EXPNAME="Main"
for i in 1 2 3
do
    CUDA_VISIBLE_DEVICES=5 accelerate launch --main_process_port 6060 --gpu_ids 5 --num_processes 1 run.py  \
        --exp_name $EXPNAME \
        --project_name ProactiveLLM \
        --seed $i \
        --scenario emotional_support \
        --log_dir logs \
        --loggers terminal,file,wandb \
        --datasets es_conv \
        --models pro_llm \
        --gen_models wrapper_gen \
        --is_so_game \
        --num_train_rl_epochs 10 \
        --ablation \
        --ablation_mode rl \
        --model_type llama3 \
        --metrics acc,prf1,sr,total_reward,avg_turn,bleu_n,rouge_n,dist_n

    CUDA_VISIBLE_DEVICES=5 accelerate launch --main_process_port 6060 --gpu_ids 5 --num_processes 1 run.py  \
        --exp_name $EXPNAME \
        --project_name ProactiveLLM \
        --seed $i \
        --scenario emotional_support \
        --log_dir logs \
        --loggers terminal,file,wandb \
        --datasets es_conv \
        --models pro_llm \
        --gen_models wrapper_gen \
        --is_so_game \
        --num_train_rl_epochs 10 \
        --test_phase \
        --ablation \
        --ablation_mode sft \
        --model_type llama3 \
        --metrics acc,prf1,sr,total_reward,avg_turn,bleu_n,rouge_n,dist_n

    # CUDA_VISIBLE_DEVICES=5 accelerate launch --main_process_port 6060 --gpu_ids 5 --num_processes 1 run.py  \
    #     --exp_name $EXPNAME \
    #     --project_name ProactiveLLM \
    #     --seed $i \
    #     --scenario emotional_support \
    #     --log_dir logs \
    #     --loggers terminal,file,wandb \
    #     --datasets es_conv \
    #     --models pro_llm \
    #     --gen_models wrapper_gen \
    #     --is_so_game \
    #     --num_train_rl_epochs 10 \
    #     --test_phase \
    #     --ablation \
    #     --ablation_mode rl \
    #     --model_type llama3 \
    #     --metrics acc,prf1,sr,total_reward,avg_turn,bleu_n,rouge_n,dist_n

    # CUDA_VISIBLE_DEVICES=5 accelerate launch --main_process_port 6060 --gpu_ids 5 --num_processes 1 run.py  \
    #     --exp_name $EXPNAME \
    #     --project_name ProactiveLLM \
    #     --seed $i \
    #     --scenario emotional_support \
    #     --log_dir logs \
    #     --loggers terminal,file,wandb \
    #     --datasets es_conv \
    #     --models pro_llm \
    #     --gen_models wrapper_gen \
    #     --is_so_game \
    #     --ablation \
    #     --ablation_mode rl \
    #     --model_type llama3 \
    #     --metrics acc,prf1,sr,total_reward,avg_turn,bleu_n,rouge_n,dist_n

    # CUDA_VISIBLE_DEVICES=5 accelerate launch --main_process_port 6060 --gpu_ids 5 --num_processes 1 run.py  \
    #     --exp_name $EXPNAME \
    #     --project_name ProactiveLLM \
    #     --seed $i \
    #     --scenario emotional_support \
    #     --log_dir logs \
    #     --loggers terminal,file,wandb \
    #     --datasets es_conv \
    #     --models pro_llm \
    #     --gen_models wrapper_gen \
    #     --is_so_game \
    #     --ablation \
    #     --ablation_mode sft \
    #     --model_type llama3 \
    #     --metrics acc,prf1,sr,total_reward,avg_turn,bleu_n,rouge_n,dist_n

    # CUDA_VISIBLE_DEVICES=5 accelerate launch --main_process_port 6060 --gpu_ids 5 --num_processes 1 run.py  \
    #     --exp_name $EXPNAME \
    #     --project_name ProactiveLLM \
    #     --seed $i \
    #     --scenario emotional_support \
    #     --log_dir logs \
    #     --loggers terminal,file,wandb \
    #     --datasets es_conv \
    #     --models pro_llm \
    #     --gen_models chatgpt \
    #     --is_so_game \
    #     --ablation \
    #     --ablation_mode sft \
    #     --model_type llama3 \
    #     --metrics acc,prf1,sr,total_reward,avg_turn,bleu_n,rouge_n,dist_n
done