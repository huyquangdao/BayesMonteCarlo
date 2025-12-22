#!/usr/bin/env bash

# Experiment script for Bayes-Adaptive LLM on NEG (negotiation) 
# Requires config/models/BAYES_NEG.yaml to have run_preference_search: true (and SFT/DPO toggled as desired).

EXPNAME="NEG_BAYES_PREF"

for seed in 1
do
# NCCL_IB_DISABLE=1
# NCCL_P2P_DISABLE=1
# NCCL_ASYNC_ERROR_HANDLING=1

#if analysis_bayes_monte_carlo phase is needed, add --analysis_bayes_monte_carlo
# ONLY add --is_utterance_based_action at evaluation stage
  accelerate launch --main_process_port 8081 --gpu_ids 5 --num_processes 1 run.py \
  --exp_name "${EXPNAME}" \
  --project_name ProactiveLLM \
  --seed "${seed}" \
  --scenario negotiation \
  --log_dir logs \
  --loggers terminal \
  --datasets craigslist_bargain \
  --models bayes_adaptive_llm \
  --gen_models llama3 \
  --model_type llama3 \
  --is_so_game \
  --is_utterance_based_action \
  --use_persona \
  --num_train_rl_epochs 10 \
  --metrics acc,prf1,sr,sl_ratio,avg_turn

done
