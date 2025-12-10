#!/usr/bin/env bash

# Experiment script for Bayes-Adaptive LLM on P4G (persuasion) mirroring the TRIP runner.
# Requires config/models/BAYES_P4G.yaml to have run_preference_search: true (and SFT/DPO toggled as desired).

EXPNAME="P4G_BAYES_PREF"

for seed in 1
do
NCCL_IB_DISABLE=1
NCCL_P2P_DISABLE=1
NCCL_ASYNC_ERROR_HANDLING=1
CUDA_VISIBLE_DEVICES=5 accelerate launch --main_process_port 8081 --gpu_ids 4 --num_processes 1 run.py \
  --exp_name "${EXPNAME}" \
  --project_name ProactiveLLM \
  --seed "${seed}" \
  --scenario persuation \
  --log_dir logs \
  --loggers terminal \
  --datasets p4g \
  --models bayes_adaptive_llm \
  --gen_models llama3 \
  --model_type llama3 \
  --is_so_game \
  --use_persona \
  --rewrite_action \
  --num_train_rl_epochs 10 \
  --metrics acc,prf1,sr,total_reward,avg_turn

done
