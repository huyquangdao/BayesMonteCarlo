EXPNAME="Main"
for i in 1
do
        CUDA_VISIBLE_DEVICES=5 accelerate launch --main_process_port 7070 --gpu_ids 5 --num_processes 1 run.py  \
        --exp_name $EXPNAME \
        --project_name ProactiveLLM \
        --seed $i \
        --scenario negotiation \
        --log_dir logs \
        --loggers terminal,file,wandb \
        --datasets craigslist_bargain \
        --models pro_llm \
        --gen_models wrapper_gen \
        --is_so_game \
        --num_train_rl_epochs 10 \
        --model_type llama3 \
        --metrics acc,prf1,sr,sl_ratio,avg_turn,bleu_n,rouge_n,dist_n

        # CUDA_VISIBLE_DEVICES=5 accelerate launch --main_process_port 5050 --gpu_ids 5 --num_processes 1 run.py  \
        # --exp_name $EXPNAME \
        # --project_name ProactiveLLM \
        # --seed $i \
        # --scenario negotiation \
        # --log_dir logs \
        # --loggers terminal,file,wandb \
        # --datasets craigslist_bargain \
        # --models pro_llm \
        # --gen_models wrapper_gen \
        # --is_so_game \
        # --ablation \
        # --ablation_mode rl \
        # --model_type llama3 \
        # --metrics acc,prf1,sr,sl_ratio,avg_turn,bleu_n,rouge_n,dist_n

        # CUDA_VISIBLE_DEVICES=5 accelerate launch --main_process_port 5050 --gpu_ids 5 --num_processes 1 run.py  \
        # --exp_name $EXPNAME \
        # --project_name ProactiveLLM \
        # --seed $i \
        # --scenario negotiation \
        # --log_dir logs \
        # --loggers terminal,file,wandb \
        # --datasets craigslist_bargain \
        # --models pro_llm \
        # --gen_models llama3 \
        # --is_so_game \
        # --ablation \
        # --ablation_mode rl \
        # --model_type llama3 \
        # --metrics acc,prf1,sr,sl_ratio,avg_turn,bleu_n,rouge_n,dist_n

        # CUDA_VISIBLE_DEVICES=5 accelerate launch --main_process_port 5050 --gpu_ids 5 --num_processes 1 run.py  \
        # --exp_name $EXPNAME \
        # --project_name ProactiveLLM \
        # --seed $i \
        # --scenario negotiation \
        # --log_dir logs \
        # --loggers terminal,file,wandb \
        # --datasets craigslist_bargain \
        # --models pro_llm \
        # --gen_models llama3 \
        # --is_so_game \
        # --ablation \
        # --ablation_mode sft \
        # --model_type llama3 \
        # --metrics acc,prf1,sr,sl_ratio,avg_turn,bleu_n,rouge_n,dist_n
done