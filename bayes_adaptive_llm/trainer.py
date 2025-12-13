"""
Skeleton trainer implementation for the Bayes-Adaptive LLM pipeline.
This module mirrors the high-level structure of `baselines/TRIP/trainer.py`
so later we can port the actual logic with minimal friction.
"""

from __future__ import annotations

import gc
import os
import random
import json
import copy
import inspect
import math
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import torch.distributed as dist
import random
from collections import defaultdict

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from tqdm import tqdm
from bayes_adaptive_llm.utils import save_finetuned_model

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple
from loguru import logger
from datasets import Dataset, DatasetDict
from multiprocessing import cpu_count
from itertools import count
from json.decoder import JSONDecodeError

import numpy as np
import torch
from torch.optim import AdamW
from datasets import Dataset as HFDataset
from loguru import logger as loguru_logger
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.trainer_utils import IntervalStrategy
from transformers.trainer import Trainer as HFTrainer
from peft import LoraConfig, get_peft_model, PeftModel


try:
    # Some TRL installs can raise RuntimeError if optional deps (e.g., openai) are missing.
    from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer
except Exception:  # pragma: no cover
    DPOConfig = None
    DPOTrainer = None
    SFTConfig = None
    SFTTrainer = None

from base.trainer import Trainer

from bayes_adaptive_llm.data_processor import (
    BayesDataProcessorForEmotionalSupport,
    BayesDataProcessorForNegotiation,
    BayesDataProcessorForPersuation,
    BayesTorchDatasetForEmotionalSupport,
    BayesTorchDatasetForNegotiation,
    BayesTorchDatasetForPersuation,
    BayesTorchDatasetForRecommendation,
)
from bayes_adaptive_llm.utils import coerce_to_float, stringify_dialogue_context
from config.constants import PREFERENCE_PAIR_PROMPT_NEGOTIATION, PREFERENCE_PAIR_PROMPT_P4G, RECOMMENDATION, NEGOTIATION, EMOTIONAL_SUPPORT, SL_RATIO, SUCCESS_RATE, AVG_TURN, FAIRNESS, \
    TOXICITY, ITEM_FREQ, USER_REWARD, PERSUATION, P4G_GOAL2DESCRIPTION, NEGOTIATION_GOAL2DESCRIPTION, ES_CONV_GOAL2DESCRIPTION, \
    P4G_GOAL2DESCRIPTION

from baselines.GDP_Zero.game import DialogGame
from baselines.GDP_Zero.openloop_mcts import OpenLoopMCTS
from baselines.GDP_Zero.player import LLMPlayer
from baselines.GDP_Zero.utils import update_state_for_open_loop_mcts
from bayes_adaptive_llm.utils import (
    sanitize_persona_description,
    stringify_dialogue_context,
    get_preference_pair,
)
from utils.logging_utils import append_to_log
from config.constants import PERSUATION
from logger.wandb_logger import WanDBLogger

def cuda_bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability(0)
    except Exception:
        return False
    return major >= 8

class PatchedDPOTrainer(DPOTrainer):
    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """
        Preserve default DPOTrainer logging while accepting start_time for transformers>=4.47.
        """
        train_eval = "train" if "loss" in logs else "eval"
        for key, metrics in self._stored_metrics[train_eval].items():
            logs[key] = torch.tensor(metrics).mean().item()
        del self._stored_metrics[train_eval]
        return HFTrainer.log(self, logs, start_time)


class PersonaDialogGame(DialogGame):
    """
    DialogGame that injects per-turn persona hints into the state so the Persuader
    generation can condition on the current persuadee profile.
    """

    def get_next_state(self, state, action):
        # attach selected goal and persona hint to the state passed into generation
        state = state.copy()
        state["pred_goal"] = action

        raw_persona = getattr(self.user_simulator, "user_profile_description", None)
        persona_hint = sanitize_persona_description(raw_persona or "")
        if persona_hint:
            state["persona_hint"] = persona_hint
        
        system_response = self.generation_method.generate_response(state, llm_pipeline=self.llm_pipeline, terminators = self.terminators)
        user_response = self.user_simulator.respond(state, llm_pipeline = self.llm_pipeline, terminators = self.terminators)
                
        next_state = update_state_for_open_loop_mcts(
            state=state,
            action=action,
            system_response=system_response,
            user_response=user_response,
        )
        return next_state

class BayesAdaptiveLLMTrainer(Trainer):
    """
    High-level trainer skeleton for the Bayes-Adaptive LLM pipeline.
    """

    def __init__(self,
                game_config,
                model_config,
                accelerator,
                game,
                model,
                offline_evaluator,
                online_evaluator,
                loggers,
                generation_method=None) -> None:
        super().__init__(game_config, model_config, accelerator, game, model, offline_evaluator,
                        online_evaluator, loggers)
        self.generation_method = generation_method
        self.tokenizer = getattr(self.model, "tokenizer", None)
        loguru_logger.debug("Initialized BayesAdaptiveLLMTrainer skeleton.")

    def _instance_to_messages_for_persuasion(self, inst):
        """
        Convert a persuasion instance to chat messages for SFT.
        Support:
        - P4G-style: inst["dialog"] = [{"er": [...], "ee": [...]}, ...]
        - generic:   inst["messages"]
        """
        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        # persona
        persona = _get(inst, "persona") or _get(inst, "user_profile_description")

        # ===== ROLE / PERSONA / STYLE =====
        lines = []
        lines.append("ROLE:")
        lines.append("You are a Persuader trying to persuade the user to donate to a charity.")

        if persona:
            lines.append("")
            lines.append("[USER PERSONA]")
            lines.append("The current user's profile is: ")
            lines.append(persona)

        lines.append("")
        lines.append("[RESPONSE STYLE]")
        lines.append("- Always answer as the Persuader speaking to the user.")
        lines.append("- Respond in 1–2 short sentences only.")

        system_content = "\n".join(lines)

        messages = [{"role": "system", "content": system_content}]

            # ===== DIALOGUE CONTEXT =====
        dialog = _get(inst, "dialog")
        if dialog is not None:
            for turn in dialog:
                # er = persuader → assistant
                for utt in turn.get("er", []):
                    utt = (utt or "").strip()
                    if utt:
                        messages.append({"role": "assistant", "content": utt})
                # ee = persuadee → user
                for utt in turn.get("ee", []):
                    utt = (utt or "").strip()
                    if utt:
                        messages.append({"role": "user", "content": utt})
            return {"messages": messages}

        dialogue_context = _get(inst, "dialogue_context")
        target_resp = _get(inst, "response")
        if dialogue_context:
            for utt in dialogue_context:
                content = (utt.get("content") or "").strip()
                if not content:
                    continue
                role = "assistant" if utt.get("role", "user") == "assistant" else "user"
                messages.append({"role": role, "content": content})
            if target_resp:
                messages.append({"role": "assistant", "content": target_resp})
            return {"messages": messages}

        maybe_msgs = _get(inst, "messages")
        if maybe_msgs is not None:
            return {"messages": maybe_msgs}

        raise ValueError(
            "Cannot infer conversation structure from instance. "
            "Please adapt _instance_to_messages_for_persuasion."
        )

    
    def _instance_to_messages_for_negotiation(self, inst):
        """
        Convert a negotiation instance to chat messages for SFT.
        Support:
        - generic:   inst["messages"]
        """
        def _get(obj, key, default=None):
            if isinstance(obj, dict):
                return obj.get(key, default)
            return getattr(obj, key, default)

        lines = []
        lines.append("ROLE:")
        lines.append("You are a Negotiator trying to reach a fair agreement with the other party.")

        # persona
        persona = _get(inst, "persona") or _get(inst, "user_profile_description")

        if persona:
            lines.append("")
            lines.append("[USER PERSONA]")
            lines.append(persona)
        lines.append("")
        lines.append("[RESPONSE STYLE]")
        lines.append("- Answer as the Negotiator in the dialogue.")
        lines.append("- Respond in 1–2 short sentences only.")

        system_content = "\n".join(lines)

        messages = [{"role": "system", "content": system_content}]

        dialogue_context = _get(inst, "dialogue_context")
        target_resp = _get(inst, "response")
        if dialogue_context:
            for utt in dialogue_context:
                content = (utt.get("content") or "").strip()
                if not content:
                    continue
                role = "assistant" if utt.get("role", "user") == "assistant" else "user"
                messages.append({"role": role, "content": content})
            if target_resp:
                messages.append({"role": "assistant", "content": target_resp})
            return {"messages": messages}

        maybe_msgs = _get(inst, "messages")
        if maybe_msgs is not None:
            return {"messages": maybe_msgs}

        raise ValueError(
            "Cannot infer conversation structure from instance. "
            "Please adapt _instance_to_messages_for_negotiation."
        )

    def _build_sft_datasets_from_instances(self, train_instances, dev_instances):
        if self.tokenizer is None:
            raise ValueError("self.model.tokenizer is None; cannot run SFT.")

        tokenizer = self.tokenizer
        if self.game_config.name == PERSUATION:
            train_records = [
                self._instance_to_messages_for_persuasion(inst)
                for inst in train_instances
            ]
            dev_records = [
                self._instance_to_messages_for_persuasion(inst)
                for inst in dev_instances
            ]
        elif self.game_config.name == NEGOTIATION:
            train_records = [
                self._instance_to_messages_for_negotiation(inst)
                for inst in train_instances
            ]
            dev_records = [
                self._instance_to_messages_for_negotiation(inst)
                for inst in dev_instances
            ]
        else:
            raise NotImplementedError("SFT dataset construction not implemented for this scenario.")

        raw_datasets = DatasetDict(
            {
                "train": Dataset.from_list(train_records),
                "eval": Dataset.from_list(dev_records),
            }
        )

        def apply_chat_template(example):
            messages = list(example["messages"])
            if len(messages) == 0:
                messages = [{"role": "system", "content": ""}]
            elif messages[0]["role"] != "system":
                messages.insert(0, {"role": "system", "content": ""})

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
            )
            return {"text": text}

        if dist.is_available() and dist.is_initialized():
            num_proc = 1
        else:
            num_proc = min(4, cpu_count())

        raw_datasets = raw_datasets.map(
            apply_chat_template,
            num_proc=num_proc,
            remove_columns=raw_datasets["train"].column_names,
            desc="Applying chat template for SFT",
        )
        # For debugging: print a random sample
        # if self.accelerator.is_local_main_process:
        #     for idx in random.sample(range(min(1, len(train_records))), k=1):
        #         print("\n=== RAW INSTANCE ===")
        #         print(train_instances[idx])
        #         print("\n=== MESSAGES ===")
        #         msgs = self._instance_to_messages_for_persuasion(train_instances[idx])["messages"]
        #         for m in msgs:
        #             print(m["role"], ":", m["content"])
        #         print("=== TEMPLATE TEXT (first 400) ===")
        #         print(raw_datasets["train"][idx]["text"][:400])
        #         print("\n")

        return raw_datasets["train"], raw_datasets["eval"]

#region abstract methods
    def process_dataset(self, dataset) -> Tuple[Any, Any, Any]:
        """
        Process the raw dataset and return the training/validation/test splits.
        """
        return dataset.train_instances, dataset.dev_instances, dataset.test_instances
    
    def construct_dataloaders(self,
                            data_instances: Sequence[Any],
                            batch_size: int,
                            goal2id: Dict[str, int],
                            shuffle: bool = True,
                            num_workers: int = 1) -> DataLoader:
        """
        Build task-specific datasets and dataloaders.
        """
        if self.game_config.name == RECOMMENDATION:
            torch_dataset = BayesTorchDatasetForRecommendation(
                tokenizer=self.tokenizer,
                instances=data_instances,
                goal2id=goal2id,
                max_sequence_length=self.model_config.max_sequence_length,
                device=self.device,
                convert_example_to_feature=BayesDataProcessorForPersuation()
            )
        # negotiation scenario
        elif self.game_config.name == NEGOTIATION:
            torch_dataset = BayesTorchDatasetForNegotiation(
                tokenizer=self.tokenizer,
                instances=data_instances,
                goal2id=goal2id,
                max_sequence_length=self.model_config.max_sequence_length,
                device=self.device,
                convert_example_to_feature=BayesDataProcessorForNegotiation()
            )
        # emotional support conversation
        elif self.game_config.name == EMOTIONAL_SUPPORT:
            torch_dataset = BayesTorchDatasetForEmotionalSupport(
                tokenizer=self.tokenizer,
                instances=data_instances,
                goal2id=goal2id,
                max_sequence_length=self.model_config.max_sequence_length,
                device=self.device,
                convert_example_to_feature=BayesDataProcessorForEmotionalSupport()
            )
        # persuasion conversations
        elif self.game_config.name == PERSUATION:
            torch_dataset = BayesTorchDatasetForPersuation(
                    tokenizer=self.tokenizer,
                    instances=data_instances,
                    goal2id=goal2id,
                    max_sequence_length=self.model_config.max_sequence_length,
                    device=self.device,
                    convert_example_to_feature=BayesDataProcessorForPersuation()
                )
        else:
            raise Exception("Something is wrong here ....")

        dataloader = DataLoader(
            torch_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=torch_dataset.collate_fn,
        )
        return dataloader

    def create_criterion(self):
        """
        method that create the loss function to train the model
        :return: a torch.nn.CrossEntropyLoss object
        """
        return torch.nn.CrossEntropyLoss()

    def create_optimizer(self, model, learning_rate=1e-5):
        """
        method that create the optimizer to train the model
        :return: a torch.optim.Optimizer
        """
        # Ensure lr is numeric even if accidentally loaded as string from yaml/cli.
        try:
            lr_value = float(learning_rate)
        except Exception:
            lr_value = 1e-5
        modules = [model]
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for model in modules for n, p in model.named_parameters()
                        if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.model_config.weight_decay,
            },
            {
                "params": [p for model in modules for n, p in model.named_parameters()
                        if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=lr_value)
        return optimizer

    def create_scheduler(self, optimizer, num_warmup_steps, max_train_steps):
        """
        method that create the lr scheduler for training the model
        :param optimizer: the optimizer that we use to train the model
        :param num_warmup_steps: number of worm up steps
        :param max_train_steps: number of training steps.
        :return: a torch.optim.lr_scheduler
        """
        lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps, max_train_steps)
        return lr_scheduler

    def train_epoch(self, data_loader, optimizer, lr_scheduler, criterion, max_train_steps):
        """
        method that trains the model on one epoch
        :param data_loader: data loader used to train the model
        :param optimizer: the optimizer used to train the model
        :param lr_scheduler:  the lr scheduler used to train the model
        :param criterion: the loss function that we use to train the model
        :param max_train_steps: the maximum number of training steps
        :return: the training loss in the current epoch
        """
        stop = False
        train_loss = []
        grad_accum = getattr(self.model_config, "gradient_accumulation", 1)
        for step, batch in enumerate(data_loader):
            logits = self.model(batch)
            loss = criterion(logits, batch['labels']) / grad_accum
            self.accelerator.backward(loss)
            train_loss.append(float(loss))

            self.progress_bar.update(1)
            self.global_step += 1

            # optim step
            if step % grad_accum == 0 or step == len(data_loader) - 1:
                if self.model_config.max_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.model_config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if self.global_step >= max_train_steps:
                stop = True
                break

        # compute average train loss
        train_loss = np.mean(train_loss) * grad_accum
        return train_loss, stop

    def eval_epoch(self, data_loader, criterion):
        """
        method that evaluates the model on the validation set.
        :param data_loader:  the data loader used to evaluate the model
        :param criterion: the loss function
        :return: evaluation loss
        """
        dev_loss = []
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, disable=not self.accelerator.is_local_main_process):
                with torch.no_grad():
                    logits = self.model(batch)
                    loss = criterion(logits, batch['labels'])
                    self.offline_evaluator.record(logits, batch['labels'])
                    dev_loss.append(float(loss))

        dev_loss = np.mean(dev_loss) * getattr(self.model_config, "gradient_accumulation", 1)
        results = self.offline_evaluator.report()
        results['loss'] = dev_loss
        return results

    def load_preference_pairs(self, pref_path: str) -> List[Dict[str, Any]]:
        with open(pref_path, "r", encoding="utf-8") as f:
            text = f.read().strip()

        if not text:
            return []

        if text[0] == "[":
            return json.loads(text)

        lines = text.splitlines()
        if len(lines) > 1:
            try:
                return [json.loads(line) for line in lines if line.strip()]
            except JSONDecodeError:
                pass

        decoder = json.JSONDecoder()
        idx = 0
        n = len(text)
        objs = []

        while idx < n:
            while idx < n and text[idx].isspace():
                idx += 1
            if idx >= n:
                break

            obj, next_idx = decoder.raw_decode(text, idx)
            objs.append(obj)
            idx = next_idx

        return objs

#endregion

    def train_sft(self, dataset, device: Optional[torch.device] = None) -> None:
        """
        Supervised fine-tuning aligned with the TRIP trainer structure but using
        the configuration schema from the reference Hugging Face script.
        """
        # if device is None:
        #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        device = self.accelerator.device

        train_instances, dev_instances, _ = self.process_dataset(dataset)

        train_dataset, eval_dataset = self._build_sft_datasets_from_instances(
            train_instances, dev_instances
        )

        base_model = getattr(self.model, "plm", self.model)
        base_model.to(device)
        
        use_lora = getattr(self.model_config, "use_lora", True)
        peft_config = None
        if use_lora:
            peft_config = LoraConfig(
                r=32,
                lora_alpha=64,
                lora_dropout=0.05,
                bias="none",
                target_modules="all-linear",
                task_type="CAUSAL_LM",
            )

        sft_config = SFTConfig(
            ddp_find_unused_parameters=False,
            output_dir=self.model_config.saved_dir,
            num_train_epochs=self.model_config.num_train_epochs,
            per_device_train_batch_size=self.model_config.batch_size,
            per_device_eval_batch_size=self.model_config.batch_size,
            gradient_accumulation_steps=getattr(self.model_config, "gradient_accumulation", 8),
            learning_rate=float(self.model_config.learning_rate),
            warmup_ratio=getattr(self.model_config, "warmup_ratio", 0.03),
            weight_decay=getattr(self.model_config, "weight_decay", 0.0),
            max_seq_length=getattr(self.model_config, "max_sequence_length", 1024),
            lr_scheduler_type=getattr(self.model_config, "lr_scheduler_type", "cosine"),
            logging_steps=getattr(self.model_config, "logging_steps", 10),
            save_steps=getattr(self.model_config, "save_steps", 500),
            eval_steps=getattr(self.model_config, "eval_steps", 500),
            eval_strategy="steps",
            save_total_limit=getattr(self.model_config, "save_total_limit", 3),
            fp16=getattr(self.model_config, "fp16", False),
            bf16=getattr(self.model_config, "bf16", True),
            gradient_checkpointing=getattr(self.model_config, "gradient_checkpointing", False),
            gradient_checkpointing_kwargs={"use_reentrant": False},
            optim=getattr(self.model_config, "optim", "paged_adamw_8bit"),
            packing=False,
            dataset_text_field="text",
            report_to=["none"],
        )

        loguru_logger.info("Initializing TRL SFTTrainer for persuasion SFT...")
        self.tokenizer.pad_token = self.tokenizer.eos_token
        sft_trainer = SFTTrainer(
            model=base_model,
            args=sft_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            peft_config=peft_config,
        )

        sft_trainer.train()

        trained_plm = sft_trainer.model
        if hasattr(self.model, "plm"):
            self.model.plm = trained_plm
        else:
            self.model = trained_plm
        sft_save_dir = os.path.join(self.model_config.saved_dir, self.model_config.sft_adapter_folder)
        save_finetuned_model(self, save_dir=sft_save_dir)
        loguru_logger.info("SFT training completed. Updated backbone LM with SFT weights.")
        loguru_logger.info("Saved SFT checkpoint to {}", sft_save_dir)

    def train_dpo(self, pref_path, device: Optional[torch.device] = None) -> None:
        """
        Run DPO fine-tuning on preference pairs using TRL's DPOTrainer.
        Loads the preference json/jsonl, feeds it directly to DPOTrainer (no custom collator),
        logs epoch losses, and saves a checkpoint.
        """
        gc.collect()
        device = self.accelerator.device
        if DPOTrainer is None or DPOConfig is None:
            loguru_logger.warning("trl DPOTrainer/DPOConfig unavailable; skipping DPO training.")
            return

        with open(pref_path, "r", encoding="utf-8") as handle:
            if pref_path.endswith(".jsonl"):
                preference_pairs = [json.loads(line) for line in handle if line.strip()]
            else:
                preference_pairs = json.load(handle)

        required_keys = {"prompt", "chosen", "rejected"}
        preference_pairs = [row for row in preference_pairs if isinstance(row, dict) and required_keys.issubset(row)]
        if not preference_pairs:
            loguru_logger.warning("No valid preference pairs (missing prompt/chosen/rejected); skipping DPO.")
            return

        peft_config = LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            bias="none",
            target_modules="all-linear",
            task_type="CAUSAL_LM",
        )

        base_plm = self.model.plm
        if not isinstance(base_plm, PeftModel):
            base_plm = get_peft_model(base_plm, peft_config)
            self.model.plm = base_plm
        base_plm.to(device)
        tokenizer = self.tokenizer
        tokenizer.pad_token = tokenizer.eos_token

        # Hyperparameters
        max_length = getattr(self.model_config, "dpo_max_length", 1024)
        max_prompt_length = getattr(self.model_config, "max_prompt_length", 512)
        per_device_train_batch_size = getattr(self.model_config, "dpo_train_batch_size", 12)
        per_device_eval_batch_size = getattr(self.model_config, "dpo_eval_batch_size", 4)
        gradient_checkpointing = getattr(self.model_config, "gradient_checkpointing", False)
        epochs = getattr(self.model_config, "dpo_epochs", 3)
        learning_rate = float(getattr(self.model_config, "dpo_learning_rate", 1e-5))
        beta = float(getattr(self.model_config, "dpo_beta", 0.1))
        warmup_ratio = float(getattr(self.model_config, "dpo_warmup_ratio", 0.1))
        grad_accum = int(getattr(self.model_config, "dpo_gradient_accumulation", 8))
        optimizer_type = getattr(self.model_config, "dpo_optimizer", "adamw_torch_fused")
        lr_scheduler_type = getattr(self.model_config, "dpo_lr_scheduler_type", "cosine")
        max_grad_norm = float(getattr(self.model_config, "dpo_max_grad_norm", 0.3))
        use_fp16 = bool(getattr(self.model_config, "dpo_fp16", False))
        use_bf16 = bool(getattr(self.model_config, "dpo_bf16", True))
        loss_type = getattr(self.model_config, "dpo_loss_type", None)
        save_dir = getattr(self.model_config, "saved_dir", "./dpo_output")

        hf_dataset = HFDataset.from_list(preference_pairs)
        split = hf_dataset.train_test_split(
            test_size=0.1,   
            shuffle=True,
            seed=42,         
        )

        train_dataset = split["train"]
        val_dataset   = split["test"]

        training_args = DPOConfig(
            output_dir=save_dir,
            num_train_epochs=epochs,                                         # number of training epochs
            per_device_train_batch_size=per_device_train_batch_size,         # batch size per device during training
            per_device_eval_batch_size=per_device_eval_batch_size,           # batch size for evaluation
            gradient_accumulation_steps=grad_accum,                          # number of steps before performing a backward/update pass
            gradient_checkpointing=gradient_checkpointing,                   # use gradient checkpointing to save memory
            optim=optimizer_type,                                            # use fused adamw optimizer
            learning_rate=learning_rate,                                     # 10x higher LR than QLoRA paper
            max_grad_norm=max_grad_norm,                                     # max gradient norm based on QLoRA paper
            warmup_ratio=warmup_ratio,                                       # warmup ratio based on QLoRA paper
            lr_scheduler_type=lr_scheduler_type,                             # use cosine learning rate scheduler
            logging_steps=25,                                                # log every 25 steps
            save_steps=500,                                                  # when to save checkpoint
            save_total_limit=2,                                              # limit the total amount of checkpoints
            eval_strategy="steps",                                           # evaluate every 1000 steps
            eval_steps=700,                                                  # when to evaluate
            bf16=use_bf16,                                                   # use bfloat16 precision
            tf32=use_fp16,                                                   # use tf32 precision
            max_length=max_length,
            max_prompt_length=max_prompt_length,
        )
        
        trainer_kwargs = dict(
            ref_model=None,
            peft_config=peft_config,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            tokenizer=tokenizer,
            model=base_plm,
            loss_type=loss_type,
            beta=beta,
        )

        trainer_cls = PatchedDPOTrainer or DPOTrainer
        dpo_trainer = trainer_cls(
            **trainer_kwargs
        )

        loguru_logger.info(
            f"Starting DPO training: {len(preference_pairs)} pairs, epochs={epochs}, "
            f"batch_size={per_device_train_batch_size}, lr={learning_rate:.1e}, beta={beta:.2f}, grad_accum={grad_accum}"
        )

        dpo_trainer.train()

        trained_plm = dpo_trainer.model
        if hasattr(self.model, "plm"):
            self.model.plm = trained_plm
        else:
            self.model = trained_plm

        dpo_save_dir = os.path.join(self.model_config.saved_dir, self.model_config.dpo_adapter_folder)
        save_finetuned_model(self, save_dir=dpo_save_dir)
        loguru_logger.info("Saved DPO checkpoint to {}", dpo_save_dir)


    def predict(self,
                instance: Dict[str, Any],
                action_mapping: Optional[Dict[str, int]] = None,
                is_test: bool = False) -> Tuple[Any, torch.Tensor]:
        """
        Select the next action (e.g., conversation goal) conditioned on the state.
        """
        # no ground-truth response during inference
        if is_test:
            instance.update({'response': None})
        
        # create input example for response generation
        train_dataset, _ = self._build_sft_datasets_from_instances(
            [instance], [instance] 
        )
        input_prompt = train_dataset[0]['text']
        # print("Input prompt for generation:", input_prompt)
        response = self.model.generate_text(input_prompt, max_new_tokens=50)
        assert response is not None
        # print("Generated response:", response)
        return response

    def select_action(self, logits: torch.Tensor, is_test: bool = True) -> Tuple[Any, torch.Tensor]:
        """
        Convert model logits to discrete actions.
        """
        raise NotImplementedError("Action selection is not implemented.")

    def test(self, dataset) -> Dict[str, float]:
        """
        Offline evaluation entry point.
        """
        raise NotImplementedError("Test routine is not implemented.")


    def generate_preference_pairs_with_mcts(self, train_cases, dev_simulators, action_mapping) -> Sequence[Dict[str, Any]]:
        """
        Simulate persuasion dialogues with OpenLoopMCTS to extract preference pairs.
        The pairs are also written to disk if `model_config.preference_pairs_path` is provided,
        and the in-memory dataset is overwritten so DPO training can consume them directly.
        """
        if self.game_config.name == PERSUATION:
            prompt_preference_by_game = PREFERENCE_PAIR_PROMPT_P4G
        elif self.game_config.name == NEGOTIATION:
            prompt_preference_by_game = PREFERENCE_PAIR_PROMPT_NEGOTIATION
        else:
            prompt_preference_by_game = ""

        if not dev_simulators:
            raise ValueError("User simulators are required for MCTS preference generation.")

        simulators = dev_simulators
        if simulators is None or len(simulators) == 0:
            raise ValueError("No simulators available for preference generation.")

        # expose mapping to player via model_config for LLMPlayer compatibility
        setattr(self.model_config, "action_mapping", action_mapping)
        logger.info("Action mapping: {}", action_mapping)
        dialog_acts = [goal for goal, _ in sorted(action_mapping.items(), key=lambda kv: kv[1])]
        player = LLMPlayer(self.game_config, action_mapping, self.model_config)
        # expose generation pipeline to player heuristics if needed
        setattr(self.model_config, "llm_pipeline", self.game_config.llm_pipeline)
        setattr(self.model_config, "terminators", self.game_config.terminators)

        # MCTS configuration
        num_MCTS_sims = getattr(self.model_config, "num_mcts_sims", 30)
        max_realizations = getattr(self.model_config, "max_realizations", 3)
        mcts_cfg = SimpleNamespace(
            cpuct=1.0,
            Q_0=getattr(self.model_config, "Q_0", 0.25),
            max_realizations=max_realizations,
        )

        preference_pairs: List[Dict[str, Any]] = []
        preference_path: Optional[Path] = None
        if getattr(self.model_config, "preference_pairs_path", None):
            preference_path = Path(self.model_config.preference_pairs_path)
            preference_path.parent.mkdir(parents=True, exist_ok=True)

        # logging raw responses to bayes_adaptive_llm/logs
        log_dir = Path(__file__).resolve().parent / "logs"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"pref_gen_{timestamp}.log"

        def _log_line(text: str) -> None:
            append_to_log(log_file, [text])

        def _run_mcts_worker(base_state, sims: int):
            """
            Run a chunk of MCTS simulations in a dedicated planner instance so we
            can merge the results later without worrying about thread safety.
            """
            worker_planner = OpenLoopMCTS(
                PersonaDialogGame(
                    self.game,
                    self.generation_method,
                    simulator,
                    llm_pipeline=self.game_config.llm_pipeline,
                    terminators=self.game_config.terminators,
                ),
                player,
                mcts_cfg,
            )
            # prime the root so that every worker counts all requested simulations
            worker_planner._init_node(copy.deepcopy(base_state))
            for _ in range(sims):
                # use a copy to prevent shared-state mutation across workers
                worker_planner.search(copy.deepcopy(base_state))
            return worker_planner

        def _merge_mcts_workers(target_planner, worker_planners, root_state):
            """
            Aggregate visit counts / values from multiple planners into a single one.
            """
            root_key = target_planner._to_string_rep(root_state)
            target_planner.simulation_counter = 0

            for wp in worker_planners:
                target_planner.simulation_counter += getattr(wp, "simulation_counter", 0)

                for state_key, vm in wp.valid_moves.items():
                    target_planner.valid_moves.setdefault(state_key, vm)

                for state_key, prior in wp.P.items():
                    # keep the first prior we see; priors are fixed per state
                    target_planner.P.setdefault(state_key, copy.deepcopy(prior))

                for state_key, ns_val in wp.Ns.items():
                    target_planner.Ns[state_key] = target_planner.Ns.get(state_key, 0) + ns_val

                for state_key, action_counts in wp.Nsa.items():
                    tgt_counts = target_planner.Nsa.setdefault(state_key, {})
                    tgt_q = target_planner.Q.setdefault(state_key, {})
                    for action_idx, count in action_counts.items():
                        prev_count = tgt_counts.get(action_idx, 0)
                        prev_q = tgt_q.get(action_idx, self.model_config.Q_0 if hasattr(self.model_config, "Q_0") else 0.0)
                        src_q = wp.Q.get(state_key, {}).get(action_idx, prev_q)

                        new_count = prev_count + count
                        tgt_counts[action_idx] = new_count

                        if new_count > 0:
                            tgt_q[action_idx] = (prev_q * prev_count + src_q * count) / new_count

                for state_key, utt_dict in wp.realizations_Vs.items():
                    tgt_vs = target_planner.realizations_Vs.setdefault(state_key, {})
                    tgt_ns = target_planner.realizations_Ns.setdefault(state_key, {})
                    src_ns = wp.realizations_Ns.get(state_key, {})
                    for utt, v_val in utt_dict.items():
                        add_n = src_ns.get(utt, 1)
                        prev_n = tgt_ns.get(utt, 0)
                        prev_v = tgt_vs.get(utt, 0.0)
                        merged_n = prev_n + add_n
                        tgt_ns[utt] = merged_n
                        tgt_vs[utt] = (prev_v * prev_n + v_val * add_n) / merged_n

            # fabricate a single trace entry for logging convenience
            if root_key in target_planner.Nsa and target_planner.Nsa[root_key]:
                if root_key in target_planner.Ns:
                    target_planner.simulation_counter = target_planner.Ns[root_key]
                prob_dict = target_planner._get_prob_distribution(root_key)
                target_planner.action_prob_traces[root_key] = [
                    {
                        "simulation": target_planner.simulation_counter,
                        "state": root_key,
                        "prob": prob_dict,
                    }
                ]

        for dialog_idx, case in enumerate(tqdm(train_cases, desc="Generating preference pairs")):
            # skip to dialog_idx
            if dialog_idx < getattr(self.model_config, "skip_to_dialog_idx", 120):
                continue
            # fix persuadee (simulator/persona) per dialog
            # sample a simulator from the simulator pool
            simulator = random.choice(simulators)
            dialog_game = PersonaDialogGame(self.game, 
                                            self.generation_method, 
                                            simulator,
                                            #### 
                                            llm_pipeline = self.game_config.llm_pipeline, 
                                            terminators = self.game_config.terminators
                                            )
                        
            # reset the initial state            
            state = self.game.reset(case, simulator)
            persona_history: List[Dict[str, str]] = []
            persona_hint = {}
            if hasattr(simulator, "user_profile_description"):
                raw_desc = getattr(simulator, "user_profile_description", "")
                persona_hint["description"] = sanitize_persona_description(raw_desc)
            if persona_hint:
                persona_history.append({"turn": 0, **persona_hint})

            dialog_pairs: List[Dict[str, Any]] = []                        
            for turn in count():
                outcome = dialog_game.get_dialog_ended(state)
                if outcome == 1.0 or outcome == -1.0:
                    logger.info("Dialog {} ended early with outcome={}; stop turn loop.", dialog_idx, outcome)
                    break
                
                logger.info("Dialog {} turn {} | state={}", dialog_idx, turn, stringify_dialogue_context(state["dialogue_context"]))

                # initialize the mcts planner
                planner = OpenLoopMCTS(
                    dialog_game, 
                    player, 
                    mcts_cfg,
                    
                )
                max_parallel_workers = max(
                    1,
                    getattr(self.model_config, "num_mcts_workers", min(cpu_count(), 4)),
                )
                min_sims_per_worker = max(1, getattr(self.model_config, "min_sims_per_worker", 8))
                # keep each worker busy with enough simulations to explore the tree
                num_mcts_workers = min(
                    max_parallel_workers,
                    max(1, num_MCTS_sims // min_sims_per_worker),
                )
                use_parallel = num_mcts_workers > 1

                if use_parallel:
                    # split simulations across workers and merge the results
                    sims_per_worker = math.ceil(num_MCTS_sims / num_mcts_workers)
                    logger.info(
                        "Running MCTS in parallel | sims={} | workers={} | sims/worker~{}",
                        num_MCTS_sims,
                        num_mcts_workers,
                        sims_per_worker,
                    )
                    worker_planners = []
                    with ThreadPoolExecutor(max_workers=num_mcts_workers) as executor:
                        futures = []
                        remaining = num_MCTS_sims
                        for _ in range(num_mcts_workers):
                            if remaining <= 0:
                                break
                            sims_this_worker = min(sims_per_worker, remaining)
                            futures.append(executor.submit(_run_mcts_worker, state, sims_this_worker))
                            remaining -= sims_this_worker

                        for fut in as_completed(futures):
                            worker_planners.append(fut.result())

                    _merge_mcts_workers(planner, worker_planners, state)
                else:
                    # search for the best action
                    for _ in range(num_MCTS_sims):
                        planner.search(state)

                # make sure we hit the requested number of root-level simulations
                root_key = planner._to_string_rep(state)
                sims_recorded = planner.Ns.get(root_key, 0)
                remaining_sims = num_MCTS_sims - sims_recorded
                if remaining_sims > 0:
                    logger.info(
                        "Top-up sequential MCTS sims to reach target | recorded={} | target={} | extra={}",
                        sims_recorded,
                        num_MCTS_sims,
                        remaining_sims,
                    )
                    if root_key not in planner.P:
                        planner._init_node(copy.deepcopy(state))
                    for _ in range(remaining_sims):
                        planner.search(state)

                action_prob = planner.get_action_prob(state)
                prob_trace = planner.get_action_prob_trace(state)
                last_prob = prob_trace[-1]["prob"] if prob_trace else {}
                logger.info(
                    "Dialog {} turn {} | sims={} | prob={}",
                    dialog_idx,
                    turn,
                    planner.simulation_counter,
                    last_prob,
                )
                
                if np.sum(action_prob) == 0:
                    logger.info("Zero action probability encountered; stopping dialog {} turn {}", dialog_idx, turn)
                    break

                
                state_rep = planner._to_string_rep(state)
                valid_moves = planner.valid_moves.get(state_rep, [])

                # sample to avoid repeatedly picking the first action when all priors are flat
                if valid_moves is not None and len(valid_moves) > 0:
                    prob = action_prob.copy()
                    prob = prob / prob.sum() if prob.sum() > 0 else np.ones_like(prob) / len(prob)
                    best_action = int(np.random.choice(len(prob), p=prob))
                else:
                    best_action = int(np.argmax(action_prob))
                goal = player.id2goal[best_action]

                prompt_dialogue_context = prompt_preference_by_game + stringify_dialogue_context(state["dialogue_context"])

                # Step environment to obtain next state and utterances
                state["dialog_id"] = dialog_idx
                state["turn_id"] = turn

                next_state, _, done, _ = self.game.step(state, 
                                                goal, 
                                                self.generation_method, 
                                                simulator
                                                )
                
                sys_utt = next_state["dialogue_context"][-2]["content"]
                user_utt = next_state["dialogue_context"][-1]["content"]
                print("done: ", done)

                # construct the preference pair
                pair = get_preference_pair(
                    action_prob,
                    state_rep,
                    dialog_acts,
                    valid_moves,
                    planner.realizations_Vs,
                    selected_action=best_action
                )

                # print full history up to current turn
                history_str = stringify_dialogue_context(next_state["dialogue_context"])

                if pair is None:
                    logger.info("Not enough realizations to form preference pair; skipping turn.")
                    state = next_state
                    continue
                
                _, best_pair, worst_pair = pair

                logger.info(
                    "Pref pair | dialog={} turn={} action={} | chosen={} (V={:.4f}) | rejected={} (V={:.4f})",
                    dialog_idx,
                    turn,
                    goal,
                    best_pair[0],
                    float(best_pair[1]),
                    worst_pair[0],
                    float(worst_pair[1]),
                )

                _log_line(
                    f"[Dialog {dialog_idx} | Turn {turn}] History+Pref:\n{history_str}\n"
                    f"Chosen: {best_pair[0]} (V={best_pair[1]:.4f})\n"
                    f"Rejected: {worst_pair[0]} (V={worst_pair[1]:.4f})"
                )

                dialog_pairs.append(
                    {
                        "prompt": prompt_dialogue_context,
                        "chosen": best_pair[0],
                        "rejected": worst_pair[0],
                        "turn": turn,
                        "action": goal,
                        "dialog_index": dialog_idx,
                        "system_utterance": sys_utt,
                        "user_utterance": user_utt,
                        "persona_hint": persona_hint or None,
                    }
                )

                # update the current state of the conversation
                state = next_state
                
                if len(state['dialogue_context']) >= self.game_config.max_horizon:
                    break

            outcome = dialog_game.get_dialog_ended(state)
            if outcome > 0.3:
                preference_pairs.extend(dialog_pairs)
                # log full dialog transcript
                full_dialog = stringify_dialogue_context(state["dialogue_context"])
                _log_line(f"=== Dialog {dialog_idx} transcript ===\n{full_dialog}\n=== End Dialog {dialog_idx} ===")
                # flush dialog pairs to disk incrementally if path provided
                if preference_path and dialog_pairs:
                    with preference_path.open("a", encoding="utf-8") as f:
                        for item in dialog_pairs:
                            f.write(json.dumps(item, ensure_ascii=False) + "\n")
                    logger.info("Appended {} pairs from dialog {} to {}", len(dialog_pairs), dialog_idx, preference_path)
            else:
                logger.debug("Dialog {} did not succeed (outcome={:.1f}); skipping its preference pairs.", dialog_idx, outcome)

        if preference_path and preference_pairs:
            # already appended per dialog; nothing more to write here.
            logger.info("Total pairs written so far: {} (path: {})", len(preference_pairs), preference_path)

        # Overwrite dataset splits so DPO trainer can consume them directly.
        if preference_pairs:
            self.dataset.train_instances = preference_pairs
            self.dataset.dev_instances = []
            self.dataset.test_instances = []

        logger.info("Generated {} preference pairs from {} dialogs.", len(preference_pairs), len(train_cases))
        return preference_pairs

    def online_test(self,
                    cases: Sequence[Any],
                    device: Optional[torch.device] = None,
                    simulators: Optional[Sequence[Any]] = None,
                    action_mapping: Optional[Dict[str, int]] = None) -> Dict[str, float]:
        """
        Simulate the policy against online simulators for evaluation.
        """
        loguru_logger.warning(f"Online Testing on Target Item in the Test Set ......")
        loguru_logger.warning(f"Num Target Items: {len(cases)}, Num Simulators: {len(simulators)}")
        # success rate and average number of conversation turns.
        SR, AvgT, total_reward = 0., 0., 0.

        turn_level_results = defaultdict(list)
        # loss = torch.tensor(0, dtype=torch.float, device=device)
        # randomly sample persona information
        # simulator = np.random.choice(simulators)
        # select a particular simulator
        # and promote items to this simulator
        # simulator = simulators[0]
        convs = []
        
        # loop over the item set
        # make sure each item is associated with one user profile
        for idx, (case, simulator) in tqdm(enumerate(list(zip(cases, simulators)))):

            # randomly sample persona information
            # simulator = np.random.choice(simulators)

            loguru_logger.info('\n================Item Num:{}===================='.format(idx))

            # reset the game state
            # construct a new game state based on the given case and the current simulator
            state = self.game.reset(case, simulator)

            # recommendation scenario
            if self.game_config.name == RECOMMENDATION:
                loguru_logger.info(f"[Target Item]: {state['task_background']['target_topic']}")
                loguru_logger.info(f"[Target Goal]: {state['task_background']['target_goal']}")

            # negotiation scenario
            elif self.game_config.name == NEGOTIATION:
                loguru_logger.info(f"[Item Name]: {state['task_background']['item_name']}")
                loguru_logger.info(f"[Seller Desired Price]: {state['task_background']['seller_price']}")
                loguru_logger.info(f"[Buyer Desired Price]: {state['task_background']['buyer_price']}")

            loguru_logger.info(f"[System]: {state['dialogue_context'][0]['content']}")
            loguru_logger.info(f"[USER]: {state['dialogue_context'][1]['content']}")

            # episode-level reward
            # more than 1 objectives, therefore the reward is a vector
            epi_reward = []
            done = False

            # create two lists to store the rewards and lob probs
            rewards = []
            log_probs = []

            # a flag to check if the conversation is successful
            # for computing the success rate
            is_successful = False
            conv_turn = 0

            # flag for checking if the target is mentioned during the conversation
            o_flag = False
            prev_reward = 0

            # interactive simulation
            for t in count():  # user  dialog

                # predict the action
                # the action in this case is a natural language utterance
                action = self.predict(state, action_mapping = None, is_test = True)
                                
                # employing the action to observe the next state
                # and the corresponding rewards
                    
                state, reward, done, o_done = self.game.step(state, action, self.generation_method, simulator)

                # check if target is mentioned during the conversation
                if o_done == 1:
                    o_flag = True

                # storing the reward
                # reward = torch.tensor([reward], device=device, dtype=torch.float)
                reward = torch.tensor([reward], dtype=torch.float)
                rewards.append(reward)
                epi_reward.append(reward)
    
                # current turn reward + past reward
                tmp_reward = (reward + prev_reward).tolist()[0]

                # cummulated reward
                turn_level_results[t].append(tmp_reward)
                prev_reward = reward

                # evaluate the outcome of the conversation
                if done:
                    # successful case
                    # if the sub_reward is greater than epsilon
                    # and the target is mentioned in the conversation
                    if done == 1 and o_flag:
                        # increase the SR
                        SR += 1
                        is_successful = True

                    AvgT += t + 1
                    conv_turn = len(state['dialogue_context'])
                    # total_reward += epi_reward
                    break

            convs.append(state)
            # compute the loss function, e.g a proxy of the policy gradient
            # newloss = self.compute_rl_policy_loss(rewards, log_probs)

            # log the results
            # if newloss is not None:
            # loss += newloss
            # construct the epi reward tensor
            if not self.game_config.is_so_game:
                epi_reward = torch.cat(epi_reward, dim=0)

                # objective-based epi reward
                objective_based_reward = epi_reward.sum(dim=0)
                turn_reward = objective_based_reward[-1].item()

                # update the online evaluator
                # recommendation scenario
                if self.game_config.name == RECOMMENDATION:
                    
                    # for recommendation
                    # the first objective is subjective reward
                    user_reward = objective_based_reward[0].item()

                    # three objectives
                    # i.e user reward, item_freq, turn_reward
                    if len(objective_based_reward) == 3:
                        item_freq = objective_based_reward[-2].item()
                        turn_reward = objective_based_reward[-1].item()
                    
                    # two objectives
                    # user_reward, item_freq
                    elif len(objective_based_reward) == 2:
                        item_freq = objective_based_reward[-1].item()
                        turn_reward = -1

                    # the second objective is target item frequency
                    self.online_evaluator.record(
                        {
                            # use to compute the success rate and avg conv turn.
                            SUCCESS_RATE: int(is_successful),
                            AVG_TURN: [conv_turn, turn_reward],
                            USER_REWARD: user_reward,
                            
                            # rewards on objectives of interest
                            # target item frequency for recommendation
                            ITEM_FREQ: item_freq

                        }
                    )

                # negotiation scenario
                elif self.game_config.name == NEGOTIATION:

                    epi_reward = epi_reward.mean(dim=0)
                    sl_ratio_reward = epi_reward[0].item()
                    fairness_reward = epi_reward[1].item()
                    turn_reward = epi_reward[-1].item()

                    # three objectives
                    # i.e user reward, item_freq, turn_reward
                    if len(objective_based_reward) == 2:
                        turn_reward = -1

                    self.online_evaluator.record(
                        {
                            # use to compute the success rate and avg conv turn.
                            SUCCESS_RATE: is_successful,
                            AVG_TURN: [conv_turn, turn_reward],
                            
                            # rewards on objectives of interest
                            # this can be used to compute the SL_ratio, Fairness Score for negotiation
                            SL_RATIO: sl_ratio_reward,
                            FAIRNESS: fairness_reward

                        }
                    )
                # emotional support conversation
                elif self.game_config.name == EMOTIONAL_SUPPORT:
                    # the first objective is the conversational sr
                    # need to be normalized to [0,1]
                    toxicity = objective_based_reward[1].item()
                    # user reward
                    user_reward = objective_based_reward[0].item() / epi_reward.shape[0]
                    # the second objective is toxicity
                    self.online_evaluator.record(
                        {
                            # use to compute the success rate and avg conv turn.
                            SUCCESS_RATE: is_successful,
                            AVG_TURN: [conv_turn, turn_reward],
                            # user-oriented reward
                            USER_REWARD: user_reward,
                            # rewards on objectives of interest
                            # toxicity for emotional support conversation
                            TOXICITY: toxicity
                        }
                    )
            # single objective game:
            else:
                epi_reward = torch.cat(epi_reward, dim=0)
                
                # objective-based epi reward
                total_reward = epi_reward.sum(dim=0)

                # update the online evaluator
                # recommendation scenario
                if self.game_config.name == RECOMMENDATION:
                    # the second objective is target item frequency
                    self.online_evaluator.record(
                        {
                            # use to compute the success rate and avg conv turn.
                            SUCCESS_RATE: int(is_successful),
                            "total_reward": total_reward.item(),
                            AVG_TURN: conv_turn
                        }
                    )

                # negotiation scenario
                elif self.game_config.name == NEGOTIATION:
                    self.online_evaluator.record(
                        {
                            # use to compute the success rate and avg conv turn.
                            SUCCESS_RATE: is_successful,
                            AVG_TURN: conv_turn,
                            SL_RATIO: total_reward.item()
                        }
                    )
                # emotional support conversation
                elif self.game_config.name == EMOTIONAL_SUPPORT:
                    self.online_evaluator.record(
                        {
                            # use to compute the success rate and avg conv turn.
                            SUCCESS_RATE: int(is_successful),
                            "total_reward": total_reward.item(),
                            AVG_TURN: conv_turn
                        }
                    )
                # persuation conversations
                elif self.game_config.name == PERSUATION:
                    self.online_evaluator.record(
                        {
                            # use to compute the success rate and avg conv turn.
                            SUCCESS_RATE: int(is_successful),
                            "total_reward": total_reward.item(),
                            AVG_TURN: conv_turn
                        }
                    )
        # multi objective game
        if not self.game_config.is_so_game:
            final_result_turns = defaultdict(list)

            for k, v in turn_level_results.items():
                final_result_turns[k] = defaultdict(list)
                final_result_turns[k] = defaultdict(list)
                final_result_turns[k] = defaultdict(list)
            
            for k,v in turn_level_results.items():
                for l in v:
                    final_result_turns[k]['gain'].append(l[0])
                    final_result_turns[k]['fair'].append(l[1])
                    final_result_turns[k]['deal'].append(l[2])

            for k, v in final_result_turns.items():
                final_result_turns[k]['gain'] = np.mean(final_result_turns[k]['gain'])
                final_result_turns[k]['fair'] = np.mean(final_result_turns[k]['fair'])
                final_result_turns[k]['deal'] = np.mean(final_result_turns[k]['deal'])
            
            for k, v in final_result_turns.items():
                print(f"turn {k}, values: {v}")

        # compute the results using the evaluator
        results = self.online_evaluator.report()

        # log the results to terminal or file
        for logger in self.loggers:
            if not isinstance(logger, WanDBLogger):
                logger.record(results, "Testing")

            # # save conversations for human evaluation
            # if isinstance(logger, FileLogger):
            #     for idx, conv in enumerate(convs):
            #         save_conv_path = os.path.join(logger.log_dir, f"conversation_{idx}.txt")
            #         save_conversation_for_human_evaluation(save_conv_path, conv)    

        # return the results of the online evaluation
        print(results)
        return results 


