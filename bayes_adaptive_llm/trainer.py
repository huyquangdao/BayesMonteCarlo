"""
Skeleton trainer implementation for the Bayes-Adaptive LLM pipeline.
This module mirrors the high-level structure of `baselines/TRIP/trainer.py`
so later we can port the actual logic with minimal friction.
"""

from __future__ import annotations

import math
import os
import random
import warnings
import json
import copy
import inspect
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from datasets import Dataset as HFDataset
from loguru import logger as loguru_logger
from torch.optim import AdamW, Optimizer
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, TaskType, get_peft_model
from transformers.trainer_utils import IntervalStrategy

try:
    # Some TRL installs can raise RuntimeError if optional deps (e.g., openai) are missing.
    from trl import DPOConfig, DPOTrainer
except Exception:  # pragma: no cover
    DPOConfig = None
    DPOTrainer = None

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
from config.constants import RECOMMENDATION, NEGOTIATION, EMOTIONAL_SUPPORT, SL_RATIO, SUCCESS_RATE, AVG_TURN, FAIRNESS, \
    TOXICITY, ITEM_FREQ, USER_REWARD, PERSUATION, P4G_GOAL2DESCRIPTION, NEGOTIATION_GOAL2DESCRIPTION, ES_CONV_GOAL2DESCRIPTION, \
    P4G_GOAL2DESCRIPTION


def cuda_bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        major, _ = torch.cuda.get_device_capability(0)
    except Exception:
        return False
    return major >= 8


class PreferenceDPODataset(Dataset):
    """
    Minimal DPO dataset for prompt / chosen / rejected triples.
    """

    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        prompt_enc = self.tokenizer(
            item["prompt"], truncation=True, max_length=self.max_length // 3, return_tensors="pt"
        )
        chosen_enc = self.tokenizer(
            item["chosen"], truncation=True, max_length=self.max_length // 3, return_tensors="pt"
        )
        rejected_enc = self.tokenizer(
            item["rejected"], truncation=True, max_length=self.max_length // 3, return_tensors="pt"
        )

        return {
            "prompt_input_ids": prompt_enc["input_ids"].squeeze(0),
            "prompt_attention_mask": prompt_enc["attention_mask"].squeeze(0),
            "chosen_input_ids": chosen_enc["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_enc["attention_mask"].squeeze(0),
            "rejected_input_ids": rejected_enc["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_enc["attention_mask"].squeeze(0),
            "chosen_score": torch.tensor(item.get("chosen_score", 0.9), dtype=torch.float32),
            "rejected_score": torch.tensor(item.get("rejected_score", 0.2), dtype=torch.float32),
        }

    def collate_fn(self, batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        def _pad(seqs, pad_val):
            return pad_sequence(seqs, batch_first=True, padding_value=pad_val)

        def _pad_mask(masks):
            return pad_sequence(masks, batch_first=True, padding_value=0)

        pad_id = self.tokenizer.pad_token_id
        prompt_input_ids = _pad([item["prompt_input_ids"] for item in batch], pad_id)
        prompt_attention_mask = _pad_mask([item["prompt_attention_mask"] for item in batch])
        chosen_input_ids = _pad([item["chosen_input_ids"] for item in batch], pad_id)
        chosen_attention_mask = _pad_mask([item["chosen_attention_mask"] for item in batch])
        rejected_input_ids = _pad([item["rejected_input_ids"] for item in batch], pad_id)
        rejected_attention_mask = _pad_mask([item["rejected_attention_mask"] for item in batch])
        chosen_score = torch.stack([item["chosen_score"] for item in batch])
        rejected_score = torch.stack([item["rejected_score"] for item in batch])

        return {
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "chosen_input_ids": chosen_input_ids,
            "chosen_attention_mask": chosen_attention_mask,
            "rejected_input_ids": rejected_input_ids,
            "rejected_attention_mask": rejected_attention_mask,
            "chosen_score": chosen_score,
            "rejected_score": rejected_score,
        }


class PreferenceDPOTrainer:
    """
    Lightweight DPO trainer with LoRA adapters.
    """

    def __init__(
        self,
        model_path: str,
        tokenizer,
        device: torch.device,
        learning_rate: float = 3e-6,
        beta: float = 0.1,
        weight_decay: float = 0.01,
        optim_name: str = "adamw_torch",
        use_fp16: bool = False,
        use_bf16: bool = False,
    ):
        self.device = device
        self.tokenizer = tokenizer
        self.beta = beta

        dtype = torch.float32
        if use_bf16 and cuda_bf16_supported():
            dtype = torch.bfloat16
        elif use_fp16 and torch.cuda.is_available():
            dtype = torch.float16

        device_map = "auto" if torch.cuda.is_available() else None

        policy_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
        )
        policy_model = self._ensure_padding(policy_model)
        if device_map is None:
            policy_model = policy_model.to(device)

        lora_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=8, lora_alpha=16, lora_dropout=0.05)
        self.policy_model = get_peft_model(policy_model, lora_config)

        reference_model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=dtype, device_map=device_map, trust_remote_code=True
        )
        self.reference_model = self._ensure_padding(reference_model)
        if device_map is None:
            self.reference_model = self.reference_model.to(device)

        opt_lower = (optim_name or "adamw_torch").lower()
        if opt_lower in ("adamw", "adamw_torch"):
            OptimCls = torch.optim.AdamW
        elif opt_lower == "adam":
            OptimCls = torch.optim.Adam
        else:
            OptimCls = torch.optim.AdamW
            loguru_logger.warning("Unknown optimizer %s; defaulting to AdamW.", optim_name)

        self.optimizer = OptimCls(self.policy_model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    def _ensure_padding(self, model):
        if model.config.pad_token_id is None:
            model.config.pad_token_id = self.tokenizer.eos_token_id
        return model

    def compute_log_probs(self, model, input_ids, attention_mask=None):
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = torch.gather(log_probs, -1, input_ids.unsqueeze(-1)).squeeze(-1)

        if attention_mask is not None:
            token_log_probs = token_log_probs * attention_mask

        seq_log_probs = token_log_probs.sum(dim=-1) / (attention_mask.sum(dim=-1) + 1e-8)
        return seq_log_probs

    def dpo_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        prompt_input_ids = batch["prompt_input_ids"].to(self.device)
        prompt_attention_mask = batch["prompt_attention_mask"].to(self.device)
        chosen_input_ids = batch["chosen_input_ids"].to(self.device)
        chosen_attention_mask = batch["chosen_attention_mask"].to(self.device)
        rejected_input_ids = batch["rejected_input_ids"].to(self.device)
        rejected_attention_mask = batch["rejected_attention_mask"].to(self.device)

        chosen_full_input_ids = torch.cat([prompt_input_ids, chosen_input_ids], dim=1)
        chosen_full_attention_mask = torch.cat([prompt_attention_mask, chosen_attention_mask], dim=1)
        rejected_full_input_ids = torch.cat([prompt_input_ids, rejected_input_ids], dim=1)
        rejected_full_attention_mask = torch.cat([prompt_attention_mask, rejected_attention_mask], dim=1)

        chosen_policy_log_probs = self.compute_log_probs(
            self.policy_model, chosen_full_input_ids, chosen_full_attention_mask
        )
        rejected_policy_log_probs = self.compute_log_probs(
            self.policy_model, rejected_full_input_ids, rejected_full_attention_mask
        )

        with torch.no_grad():
            chosen_ref_log_probs = self.compute_log_probs(
                self.reference_model, chosen_full_input_ids, chosen_full_attention_mask
            )
            rejected_ref_log_probs = self.compute_log_probs(
                self.reference_model, rejected_full_input_ids, rejected_full_attention_mask
            )

        chosen_ratio = chosen_policy_log_probs - chosen_ref_log_probs
        rejected_ratio = rejected_policy_log_probs - rejected_ref_log_probs
        losses = -F.logsigmoid(self.beta * (chosen_ratio - rejected_ratio))
        return losses.mean()

    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        self.policy_model.train()
        loss = self.dpo_loss(batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.item())


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
            for batch in tqdm.tqdm(data_loader, disable=not self.accelerator.is_local_main_process):
                with torch.no_grad():
                    logits = self.model(batch)
                    loss = criterion(logits, batch['labels'])
                    self.offline_evaluator.record(logits, batch['labels'])
                    dev_loss.append(float(loss))

        dev_loss = np.mean(dev_loss) * getattr(self.model_config, "gradient_accumulation", 1)
        results = self.offline_evaluator.report()
        results['loss'] = dev_loss
        return results

#region Preparation for DPO training
#endregion

    def train_sft(self, dataset, device: Optional[torch.device] = None) -> None:
        """
        Supervised fine-tuning aligned with the TRIP trainer structure but using
        the configuration schema from the reference Hugging Face script.
        """
        train_instances, dev_instances, _ = self.process_dataset(dataset)

        action_mapping = dataset.construct_action_mapping(
            combine=self.model_config.combined_action if not self.game_config.is_so_game else False
        )

        num_workers = getattr(self.model_config, "num_workers", 0)

        train_loader = self.construct_dataloaders(
            train_instances,
            batch_size=self.model_config.batch_size,
            goal2id=action_mapping,
            shuffle=True,
            num_workers=num_workers,
        )

        dev_loader = self.construct_dataloaders(
            dev_instances,
            batch_size=self.model_config.batch_size,
            goal2id=action_mapping,
            shuffle=False,
            num_workers=num_workers,
        )

        best_loss = math.inf
        optimizer = self.create_optimizer(self.model, self.model_config.learning_rate)

        self.model, optimizer, train_dataloader = self.accelerator.prepare(self.model, optimizer, train_loader)

        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.model_config.gradient_accumulation)
        max_train_steps = self.model_config.num_train_epochs * num_update_steps_per_epoch

        warmup_steps = int(self.model_config.warmup_ratio * max_train_steps)
        lr_scheduler = self.create_scheduler(optimizer, warmup_steps, max_train_steps)

        self.criterion = self.create_criterion()
        self.progress_bar = tqdm.tqdm(range(max_train_steps), disable=not self.accelerator.is_local_main_process)

        self.model.to(device)
        for epoch in range(self.model_config.num_train_epochs):
            self.model.train()
            self.offline_evaluator.reset()

            train_loss, stop = self.train_epoch(
                data_loader=train_dataloader,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                criterion=self.criterion,
                max_train_steps=max_train_steps,
            )

            results = self.eval_epoch(dev_loader, self.criterion)
            for logger in self.loggers:
                logger.record(results, epoch + 1)

            if results['loss'] < best_loss:
                loguru_logger.info("Performance improved. Saving the model .....")
                best_loss = results['loss']

                if self.game_config.name == RECOMMENDATION:
                    file_path = os.path.join(self.model_config.saved_dir, f"model_{self.model_config.domain}.pth")
                    
                elif self.game_config.name == NEGOTIATION:
                    file_path = os.path.join(self.model_config.saved_dir, f"model.pth")
                
                elif self.game_config.name == EMOTIONAL_SUPPORT:
                    file_path = os.path.join(self.model_config.saved_dir, f"model.pth")
                
                elif self.game_config.name == PERSUATION:
                    file_path = os.path.join(self.model_config.saved_dir, f"model.pth")
                self.save_model(file_path)

                if getattr(self.model_config, "save_hf_checkpoint", False):
                    hf_subdir = getattr(self.model_config, "hf_checkpoint_subdir", "hf_checkpoint") or "hf_checkpoint"
                    hf_dir = os.path.join(self.model_config.saved_dir, hf_subdir)
                    os.makedirs(hf_dir, exist_ok=True)
                    try:
                        if hasattr(self.model, "plm"):
                            self.model.plm.save_pretrained(hf_dir)
                        if hasattr(self.model, "tokenizer"):
                            self.model.tokenizer.save_pretrained(hf_dir)
                        loguru_logger.info("Saved HF-format checkpoint for DPO at %s", hf_dir)
                    except Exception as exc:
                        loguru_logger.warning("Failed to export HF-format checkpoint to %s: %s", hf_dir, exc)

            if stop:
                loguru_logger.info("Training process is completed.")
                break

    def train_dpo(self, dataset, device: Optional[torch.device] = None) -> None:
        """
        Preference-based DPO fine-tuning with a lightweight LoRA adapter.
        Expects dataset.train_instances to contain dictionaries with keys:
        - prompt
        - chosen
        - rejected
        Optionally reads from `model_config.preference_pairs_path` if the dataset is empty.
        """
        device = device or getattr(self, "device", None) or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        pref_path = getattr(self.model_config, "preference_pairs_path", None)
        preference_pairs: List[Dict[str, Any]] = []
        if pref_path is None:
            loguru_logger.warning("No preference_pairs_path provided; skipping DPO.")
            return
        if not os.path.exists(pref_path):
            loguru_logger.warning("Preference pairs file not found at %s; skipping DPO.", pref_path)
            return
        with open(pref_path, "r", encoding="utf-8") as handle:
            if pref_path.endswith(".jsonl"):
                preference_pairs = [json.loads(line) for line in handle if line.strip()]
            else:
                preference_pairs = json.load(handle)

        required_keys = {"prompt", "chosen", "rejected"}
        filtered_pairs = [row for row in preference_pairs if isinstance(row, dict) and required_keys.issubset(row)]
        dropped = len(preference_pairs) - len(filtered_pairs)
        if dropped:
            loguru_logger.warning(
                "Filtered out %d preference items missing required keys %s.", dropped, ", ".join(sorted(required_keys))
            )

        if not filtered_pairs:
            loguru_logger.warning("No valid preference pairs (missing prompt/chosen/rejected); skipping DPO.")
            return
        preference_pairs = filtered_pairs

        # Resolve initialization checkpoint: prefer explicit DPO path; optionally override with SFT checkpoint.
        model_path = getattr(self.model_config, "dpo_model_path", None)
        if getattr(self.model_config, "dpo_use_sft_checkpoint", False):
            saved_dir = getattr(self.model_config, "saved_dir", None)
            candidate = None
            adapter_only = None
            if saved_dir and os.path.isdir(saved_dir):
                if os.path.exists(os.path.join(saved_dir, "config.json")):
                    candidate = saved_dir
                else:
                    for entry in os.listdir(saved_dir):
                        subdir = os.path.join(saved_dir, entry)
                        if not os.path.isdir(subdir):
                            continue
                        if os.path.exists(os.path.join(subdir, "config.json")):
                            candidate = subdir
                            break
                        if os.path.exists(os.path.join(subdir, "dpo_adapter", "adapter_config.json")):
                            adapter_only = os.path.join(subdir, "dpo_adapter")
                    if not candidate and os.path.exists(os.path.join(saved_dir, "dpo_adapter", "adapter_config.json")):
                        adapter_only = os.path.join(saved_dir, "dpo_adapter")
            if candidate:
                loguru_logger.info("Using SFT checkpoint at %s for DPO initialization.", candidate)
                model_path = candidate
            else:
                if adapter_only:
                    loguru_logger.warning(
                        "Found LoRA adapter at %s but no base HF checkpoint (config.json); falling back to dpo_model_path/plm.",
                        adapter_only,
                    )
                else:
                    loguru_logger.warning("Requested SFT checkpoint for DPO but no HF config.json found under saved_dir; falling back.")
            # ensure candidate is a decoder-style LM; otherwise ignore and fallback
            if candidate:
                try:
                    cand_cfg = AutoConfig.from_pretrained(candidate)
                    is_decoder = bool(getattr(cand_cfg, "is_decoder", False))
                    model_type = getattr(cand_cfg, "model_type", "")
                    if not is_decoder:
                        loguru_logger.warning(
                            "SFT checkpoint at %s has model_type=%s and is not a decoder LM; skipping for DPO init.",
                            candidate,
                            model_type,
                        )
                        candidate = None
                except Exception as exc:
                    loguru_logger.warning("Failed to read config from %s (%s); skipping for DPO init.", candidate, exc)
            if candidate:
                loguru_logger.info("Using SFT checkpoint at %s for DPO initialization.", candidate)
                model_path = candidate

        if not model_path:
            model_path = getattr(self.model_config, "plm", "gpt2")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        max_length = getattr(self.model_config, "dpo_max_length", 1024)
        batch_size = getattr(self.model_config, "dpo_batch_size", 2)
        epochs = getattr(self.model_config, "dpo_epochs", 3)
        learning_rate = getattr(self.model_config, "dpo_learning_rate", 1e-5)
        beta = getattr(self.model_config, "dpo_beta", 0.1)
        weight_decay = getattr(self.model_config, "dpo_weight_decay", 0.01)
        warmup_ratio = getattr(self.model_config, "dpo_warmup_ratio", 0.1)
        optim_name = getattr(self.model_config, "dpo_optim", "adamw_torch")
        grad_accum = max(1, int(getattr(self.model_config, "dpo_gradient_accumulation", 4)))
        use_fp16 = bool(getattr(self.model_config, "dpo_fp16", False))
        use_bf16 = bool(getattr(self.model_config, "dpo_bf16", False))

        dpo_dataset = PreferenceDPODataset(preference_pairs, tokenizer, max_length=max_length)
        dataloader = DataLoader(
            dpo_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=dpo_dataset.collate_fn,
        )

        dpo_trainer = PreferenceDPOTrainer(
            model_path=model_path,
            tokenizer=tokenizer,
            device=device,
            learning_rate=learning_rate,
            beta=beta,
            weight_decay=weight_decay,
            optim_name=optim_name,
            use_fp16=use_fp16,
            use_bf16=use_bf16,
        )

        disable_tqdm = False
        if hasattr(self, "accelerator"):
            disable_tqdm = not self.accelerator.is_local_main_process

        loguru_logger.info(
            f"Starting DPO training: {len(preference_pairs)} valid pairs, "
            f"epochs={epochs}, batch_size={batch_size}, lr={learning_rate:.1e}, "
            f"beta={beta:.2f}, grad_accum={grad_accum}, optim={optim_name}"
        )

        total_steps = math.ceil(len(dataloader) / grad_accum) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
        scheduler = None
        if warmup_steps > 0:
            scheduler = get_linear_schedule_with_warmup(
                dpo_trainer.optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )

        for epoch in range(epochs):
            step_losses: List[float] = []
            progress = tqdm.tqdm(dataloader, disable=disable_tqdm, desc=f"DPO epoch {epoch+1}/{epochs}")
            dpo_trainer.policy_model.train()
            dpo_trainer.optimizer.zero_grad()

            for step_idx, batch in enumerate(progress):
                loss = dpo_trainer.dpo_loss(batch) / grad_accum
                loss.backward()

                if (step_idx + 1) % grad_accum == 0 or step_idx == len(dataloader) - 1:
                    dpo_trainer.optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    dpo_trainer.optimizer.zero_grad()

                step_losses.append(float(loss.item()) * grad_accum)
                progress.set_postfix(loss=step_losses[-1])

            avg_loss = float(np.mean(step_losses)) if step_losses else 0.0
            loguru_logger.info(f"Epoch {epoch + 1} completed. Avg loss: {avg_loss:.4f}")

        adapter_dir = getattr(self.model_config, "dpo_adapter_path", None)
        if not adapter_dir:
            adapter_dir = os.path.join(self.model_config.saved_dir, "dpo_adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        dpo_trainer.policy_model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        file_path = os.path.join(self.model_config.saved_dir, f"model_dpo.pth")
        self.save_model(file_path)
        loguru_logger.info("Saved LoRA DPO adapter to %s", adapter_dir)

    def predict(self,
                instance: Dict[str, Any],
                action_mapping: Optional[Dict[str, int]] = None,
                is_test: bool = False) -> Tuple[Any, torch.Tensor]:
        """
        Select the next action (e.g., conversation goal) conditioned on the state.
        """
        raise NotImplementedError("Predict method is not implemented.")

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

    def online_test(self,
                    cases: Sequence[Any],
                    device: Optional[torch.device] = None,
                    simulators: Optional[Sequence[Any]] = None,
                    action_mapping: Optional[Dict[str, int]] = None) -> Dict[str, float]:
        """
        Simulate the policy against online simulators for evaluation.
        """
        raise NotImplementedError("Online testing is not implemented.")
