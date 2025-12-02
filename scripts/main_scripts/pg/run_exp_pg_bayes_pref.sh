#!/usr/bin/env bash

# Experiment script for Bayes-Adaptive LLM on P4G (persuasion) mirroring the TRIP runner.
# Requires config/models/BAYES_P4G.yaml to have run_preference_search: true (and SFT/DPO toggled as desired).

EXPNAME="P4G_BAYES_PREF"

for seed in 1
do
CUDA_VISIBLE_DEVICES=5 accelerate launch --main_process_port 8081 --gpu_ids 1 --num_processes 2 run.py \
  --exp_name "${EXPNAME}" \
  --project_name ProactiveLLM \
  --seed "${seed}" \
  --scenario persuation \
  --log_dir logs \
  --loggers terminal \
  --datasets p4g \
  --models bayes_adaptive_llm \
  --gen_models qwen \
  --model_type qwen \
  --is_so_game \
  --use_persona \
  --num_train_rl_epochs 10 \
  --metrics acc,prf1,sr,total_reward,avg_turn

done
