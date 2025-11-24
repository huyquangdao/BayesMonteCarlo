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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import tqdm
from datasets import Dataset as HFDataset
from loguru import logger as loguru_logger
from torch.optim import AdamW, Optimizer
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
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
    def _preference_example_to_dict(self, example: Any) -> Dict[str, str]:
        if isinstance(example, dict):
            prompt = example.get("prompt")
            chosen = example.get("chosen")
            rejected = example.get("rejected")
        else:
            prompt = getattr(example, "prompt", None)
            chosen = getattr(example, "chosen", None)
            rejected = getattr(example, "rejected", None)
        if prompt is None or chosen is None or rejected is None:
            raise ValueError("Preference example must contain prompt, chosen and rejected fields.")
        return {"prompt": prompt, "chosen": chosen, "rejected": rejected}

    def prepare_preference_datasets(self, train_pref: List[Any], val_pref: Optional[List[Any]] = None):
        if not train_pref:
            raise ValueError("No preference examples available for training.")

        rng = random.Random(getattr(self.model_config, "seed", 42))
        rng.shuffle(train_pref)

        max_samples = getattr(self.model_config, "max_samples", None)
        if max_samples is not None and max_samples > 0:
            train_pref = train_pref[:max_samples]

        if val_pref:
            warnings.warn(
                "Provided validation preference set is ignored; deriving validation split from training data.",
                RuntimeWarning,
            )

        validation_ratio = getattr(self.model_config, "validation_ratio", 0.0)
        val_size = int(len(train_pref) * validation_ratio)
        eval_pref = train_pref[:val_size] if val_size > 0 else []
        train_pref = train_pref[val_size:]
        if not train_pref:
            raise ValueError("Not enough preference examples for training after validation split.")

        train_dataset = HFDataset.from_list([self._preference_example_to_dict(ex) for ex in train_pref])
        eval_dataset = HFDataset.from_list([self._preference_example_to_dict(ex) for ex in eval_pref]) if eval_pref else None
        return train_dataset, eval_dataset

    def setup_dpo_config(self, do_eval: bool, effective_max_length: Optional[int], max_prompt_length: Optional[int]) -> DPOConfig:
        if DPOConfig is None:
            raise ImportError("trl is required for DPO training. Install it with `pip install trl`.")

        output_dir = getattr(self.model_config, "output_dir", getattr(self.model_config, "saved_dir", "."))
        batch_size = getattr(self.model_config, "batch_size", getattr(self.model_config, "per_device_train_batch_size", 1))

        return DPOConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=getattr(self.model_config, "gradient_accumulation", getattr(self.model_config, "gradient_accumulation_steps", 1)),
            learning_rate=self.model_config.learning_rate,
            num_train_epochs=self.model_config.num_train_epochs,
            weight_decay=self.model_config.weight_decay,
            warmup_ratio=self.model_config.warmup_ratio,
            logging_steps=self.model_config.logging_steps,
            eval_strategy=IntervalStrategy.EPOCH if do_eval else IntervalStrategy.NO,
            save_strategy=IntervalStrategy.EPOCH,
            save_total_limit=self.model_config.save_total_limit,
            report_to=[],
            fp16=getattr(self.model_config, "fp16", False) and torch.cuda.is_available(),
            bf16=getattr(self.model_config, "bf16", False) and cuda_bf16_supported(),
            remove_unused_columns=False,
            beta=getattr(self.model_config, "dpo_beta", 0.1),
            max_length=effective_max_length,
            max_prompt_length=max_prompt_length,
            gradient_checkpointing=getattr(self.model_config, "gradient_checkpointing", False),
            ddp_find_unused_parameters=False,
            do_train=True,
            do_eval=do_eval,
            optim="adamw_torch",
        )

    def prepare_reference_model(self):
        reference_model = getattr(self.model_config, "reference_model", None)
        if reference_model is None:
            # Default to a frozen copy of the current model if none is provided.
            loguru_logger.warning("reference_model not provided; cloning current model for DPO.")
            reference_model = copy.deepcopy(self.model)
            # cache on config so repeated calls reuse the same copy
            setattr(self.model_config, "reference_model", reference_model)
        reference_model.requires_grad_(False)
        reference_model.eval()
        return reference_model
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

            if stop:
                loguru_logger.info("Training process is completed.")
                break

    def train_dpo(self, dataset, device: Optional[torch.device] = None) -> None:
        """
        Train the policy using Direct Preference Optimisation.
        """
        if DPOTrainer is None or DPOConfig is None:
            raise ImportError("trl is required for DPO training. Install it with `pip install trl`.")

        train_pref, dev_pref, _ = self.process_dataset(dataset)

        def _has_pref_fields(ex: Any) -> bool:
            if isinstance(ex, dict):
                return all(k in ex for k in ("prompt", "chosen", "rejected"))
            return all(hasattr(ex, k) for k in ("prompt", "chosen", "rejected"))

        # If dataset is not already preference-formatted, try loading from jsonl.
        if (not train_pref) or not _has_pref_fields(train_pref[0]):
            pref_path = getattr(self.model_config, "preference_pairs_path", None)
            if pref_path is None or not os.path.exists(pref_path):
                raise ValueError(
                    "Preference pairs are missing. Set model_config.preference_pairs_path to a jsonl file "
                    "with prompt/chosen/rejected fields, or ensure dataset.train_instances contain them."
                )
            loguru_logger.info("Loading preference pairs from %s", pref_path)
            with open(pref_path, "r", encoding="utf-8") as f:
                pref_data = [json.loads(line) for line in f if line.strip()]
            if not pref_data:
                raise ValueError(f"No preference pairs loaded from {pref_path}.")
            dataset.set_instances(pref_data, [], [])
            train_pref, dev_pref, _ = self.process_dataset(dataset)

        train_dataset, eval_dataset = self.prepare_preference_datasets(train_pref, dev_pref)

        reference_model = self.prepare_reference_model()

        max_length = getattr(self.model_config, "max_length", None)
        effective_max_length = max_length
        if max_length is not None and hasattr(self.model, "config"):
            model_max = getattr(self.model.config, "max_position_embeddings", max_length)
            effective_max_length = min(model_max, max_length)
            if effective_max_length < max_length:
                warnings.warn(
                    f"Requested max_length={max_length} exceeds model capacity ({model_max}). "
                    f"Using {effective_max_length} instead.",
                    RuntimeWarning,
                )

        max_prompt_length = getattr(self.model_config, "max_prompt_length", None)
        if max_prompt_length is None and effective_max_length is not None:
            max_prompt_length = effective_max_length
        elif max_prompt_length is not None and effective_max_length is not None:
            max_prompt_length = min(max_prompt_length, effective_max_length)

        # Ensure numeric hyperparameters before building DPO args.
        self.model_config.learning_rate = coerce_to_float(getattr(self.model_config, "learning_rate", 1e-5), 1e-5)
        self.model_config.weight_decay = coerce_to_float(getattr(self.model_config, "weight_decay", 0.0), 0.0)

        dpo_args = self.setup_dpo_config(do_eval=eval_dataset is not None,
                                         effective_max_length=effective_max_length,
                                         max_prompt_length=max_prompt_length)

        dpo_trainer = DPOTrainer(
            self.model,
            reference_model,
            args=dpo_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        dpo_trainer.train()
        dpo_trainer.save_model()

        output_dir = getattr(self.model_config, "output_dir", getattr(self.model_config, "saved_dir", "."))
        os.makedirs(output_dir, exist_ok=True)
        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(output_dir)
        else:
            self.save_model(os.path.join(output_dir, "model.pth"))
        if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
            self.tokenizer.save_pretrained(output_dir)

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
