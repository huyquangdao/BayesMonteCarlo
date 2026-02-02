EXPNAME="Main"
for i in 1
do
CUDA_VISIBLE_DEVICES=2 accelerate launch --main_process_port 71 --gpu_ids 4 --num_processes 1 run.py  \
        --exp_name $EXPNAME \
        --project_name ProactiveLLM \
        --seed $i \
        --scenario persuation \
        --log_dir logs \
        --loggers terminal,file,wandb \
        --datasets p4g \
        --models proactive \
        --gen_models llama3 \
        --model_type llama3 \
        --analysis_bayes_monte_carlo \
        --is_so_game \
        --use_persona \
        --metrics acc,prf1,sr,total_reward,avg_turn,bleu_n,rouge_n,dist_n        
done
