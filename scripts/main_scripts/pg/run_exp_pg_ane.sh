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
        --models ane \
        --gen_models llama3 \
        --is_so_game \
        --model_type llama3 \
        --metrics acc,prf1,sr,total_reward,avg_turn,bleu_n,rouge_n,dist_n        
done
