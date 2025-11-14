EXPNAME="Main"
for i in 1 2 3
do
CUDA_VISIBLE_DEVICES=6 accelerate launch --main_process_port 6060 --gpu_ids 6 --num_processes 1 run.py  \
        --exp_name $EXPNAME \
        --project_name ProactiveLLM \
        --seed $i \
        --scenario emotional_support \
        --log_dir logs \
        --loggers terminal,file,wandb \
        --datasets es_conv \
        --models proactive \
        --gen_models llama3 \
        --is_so_game \
        --rewrite_action \
        --model_type llama3 \
        --metrics acc,prf1,sr,total_reward,avg_turn,bleu_n,rouge_n,dist_n
done