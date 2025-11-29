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
from tqdm import tqdm

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple
from loguru import logger
from itertools import count

import numpy as np
import torch
from torch.optim import AdamW
from datasets import Dataset as HFDataset
from loguru import logger as loguru_logger
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup
from transformers.trainer_utils import IntervalStrategy
from transformers.trainer import Trainer as HFTrainer


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
        self.progress_bar = tqdm(range(max_train_steps), disable=not self.accelerator.is_local_main_process)

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
                        loguru_logger.info("Saved HF-format checkpoint for DPO at {}", hf_dir)
                    except Exception as exc:
                        loguru_logger.warning("Failed to export HF-format checkpoint to %s: %s", hf_dir, exc)

            if stop:
                loguru_logger.info("Training process is completed.")
                break

    def train_dpo(self, pref_path, device: Optional[torch.device] = None) -> None:
        """
        Run DPO fine-tuning on preference pairs using TRL's DPOTrainer.
        Loads the preference json/jsonl, feeds it directly to DPOTrainer (no custom collator),
        logs epoch losses, and saves a checkpoint.
        """
        device = device or getattr(self, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
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

        model_path = getattr(self.model_config, "dpo_model_path", None) or getattr(self.model_config, "plm", "gpt2")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Hyperparameters
        max_length = getattr(self.model_config, "dpo_max_length", getattr(self.model_config, "max_length", 1024))
        max_prompt_length = getattr(self.model_config, "max_prompt_length", max_length)
        batch_size = getattr(self.model_config, "dpo_batch_size", getattr(self.model_config, "batch_size", 2))
        epochs = getattr(self.model_config, "dpo_epochs", getattr(self.model_config, "num_train_epochs", 3))
        learning_rate = getattr(self.model_config, "dpo_learning_rate", getattr(self.model_config, "learning_rate", 1e-5))
        beta = getattr(self.model_config, "dpo_beta", 0.1)
        warmup_ratio = getattr(self.model_config, "dpo_warmup_ratio", getattr(self.model_config, "warmup_ratio", 0.1))
        grad_accum = max(
            1, int(getattr(self.model_config, "dpo_gradient_accumulation", getattr(self.model_config, "gradient_accumulation", 1)))
        )
        use_fp16 = bool(getattr(self.model_config, "dpo_fp16", getattr(self.model_config, "fp16", False)))
        use_bf16 = bool(getattr(self.model_config, "dpo_bf16", getattr(self.model_config, "bf16", False)))
        loss_type = getattr(self.model_config, "dpo_loss_type", None)
        save_dir = getattr(self.model_config, "saved_dir", "./dpo_output")

        hf_dataset = HFDataset.from_list(preference_pairs)

        training_args = DPOConfig(
            output_dir=save_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            num_train_epochs=epochs,
            learning_rate=learning_rate,
            warmup_ratio=warmup_ratio,
            fp16=use_fp16,
            bf16=use_bf16,
            save_strategy=IntervalStrategy.NO,
            logging_strategy="epoch",
            report_to="none",
            remove_unused_columns=False,
            logging_steps=getattr(self.model_config, "logging_steps", 10),
        )
        trainer_kwargs = dict(
            model=model_path,
            loss_type=loss_type,
            args=training_args,
            train_dataset=hf_dataset,
            max_length=max_length,
            max_prompt_length=max_prompt_length,
        )

        trainer_cls = PatchedDPOTrainer or DPOTrainer
        try:
            dpo_trainer = trainer_cls(processing_class=tokenizer, **trainer_kwargs)
        except TypeError:
            dpo_trainer = trainer_cls(tokenizer=tokenizer, **trainer_kwargs)

        # ensure models are on the requested device (Trainer will handle wrapping later)
        try:
            dpo_trainer.model.to(device)
            if hasattr(dpo_trainer, "ref_model"):
                dpo_trainer.ref_model.to(device)
        except Exception as exc:
            loguru_logger.warning("Could not move DPO models to device %s: %s", device, exc)

        loguru_logger.info(
            f"Starting DPO training: {len(preference_pairs)} pairs, epochs={epochs}, "
            f"batch_size={batch_size}, lr={learning_rate:.1e}, beta={beta:.2f}, grad_accum={grad_accum}"
        )

        dpo_trainer.train()
        for log_row in dpo_trainer.state.log_history:
            if "train_loss" in log_row:
                epoch_val = log_row.get("epoch", "?")
                loss_val = log_row.get("train_loss")
                loguru_logger.info("DPO epoch {} train_loss={:.4f}", epoch_val, float(loss_val))

        adapter_dir = getattr(self.model_config, "dpo_adapter_path", None)
        if not adapter_dir:
            adapter_dir = os.path.join(save_dir, "dpo_adapter")
        os.makedirs(adapter_dir, exist_ok=True)
        dpo_trainer.save_model(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        file_path = os.path.join(self.model_config.saved_dir, f"model_dpo.pth")
        self.save_model(file_path)
        loguru_logger.info("Saved DPO checkpoint to {}", adapter_dir)

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


    def generate_preference_pairs_with_mcts(self, train_cases, dev_simulators, action_mapping) -> Sequence[Dict[str, Any]]:
        """
        Simulate persuasion dialogues with OpenLoopMCTS to extract preference pairs.
        The pairs are also written to disk if `model_config.preference_pairs_path` is provided,
        and the in-memory dataset is overwritten so DPO training can consume them directly.
        """
        if self.game_config.name != PERSUATION:
            logger.warning("Preference search is currently implemented for persuasion only; skipping.")
            return []

        if not dev_simulators:
            raise ValueError("User simulators are required for MCTS preference generation.")

        simulators = dev_simulators
        if simulators is None or len(simulators) == 0:
            raise ValueError("No simulators available for preference generation.")

        # expose mapping to player via model_config for LLMPlayer compatibility
        setattr(self.model_config, "action_mapping", action_mapping)
        dialog_acts = [goal for goal, _ in sorted(action_mapping.items(), key=lambda kv: kv[1])]
        player = LLMPlayer(self.game_config, action_mapping, self.model_config)

        # MCTS configuration
        num_MCTS_sims = getattr(self.model_config, "num_mcts_sims", 30)
        max_realizations = getattr(self.model_config, "max_realizations", 8)
        extra_pair_rollouts = getattr(self.model_config, "extra_pair_rollouts", 10)
        # max_turns = getattr(self.model_config, "max_turns", 12)
        mcts_cfg = SimpleNamespace(
            cpuct=1.0,
            Q_0=getattr(self.model_config, "Q_0", 0.25),
            max_realizations=max_realizations,
        )

        # Dialog seeds
        cases = train_cases
        max_cases = getattr(self.model_config, "mcts_num_evaluate", None)
        if max_cases is not None and max_cases > 0:
            cases = cases[:max_cases]

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

        for dialog_idx, case in enumerate(tqdm(train_cases, desc="Generating preference pairs")):
            # fix persuadee (simulator/persona) per dialog, similar to TRIP
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
            # for turn in range(max_turns):
            # interactive conversations
                        
            for turn in count():
                # outcome = dialog_game.get_dialog_ended(state)
                # if outcome != 0:
                #     break
                
                logger.info("Dialog {} turn {} | state={}", dialog_idx, turn, stringify_dialogue_context(state["dialogue_context"]))

                # initialize the mcts planner
                planner = OpenLoopMCTS(
                    dialog_game, 
                    player, 
                    mcts_cfg,
                    
                )
                
                # search for the best action
                for _ in range(num_MCTS_sims):
                    planner.search(state)

                action_prob = planner.get_action_prob(state)
                prob_trace = planner.get_action_prob_trace(state)
                last_prob = prob_trace[-1]["prob"] if prob_trace else {}
                _log_line(
                    f"[Dialog {dialog_idx} | Turn {turn}] MCTS sims={planner.simulation_counter} "
                    f"prob_trace={json.dumps(prob_trace, ensure_ascii=False)}"
                )
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

                # Step environment to obtain next state and utterances
                state["dialog_id"] = dialog_idx
                state["turn_id"] = turn
                
                # next_state = dialog_game.get_next_state(state, goal)
                next_state, _, done, _ = self.game.step(state, 
                                                goal, 
                                                self.generation_method, 
                                                simulator
                                                )
                
                sys_utt = next_state["dialogue_context"][-2]["content"]
                user_utt = next_state["dialogue_context"][-1]["content"]

                # print full history up to current turn
                history_str = stringify_dialogue_context(next_state["dialogue_context"])

                # construct the preference pair
                pair = get_preference_pair(
                    action_prob,
                    state_rep,
                    dialog_acts,
                    valid_moves,
                    planner.realizations_Vs,
                )
                if pair is None:
                    # try additional focused rollouts to collect more realizations
                    for _ in range(extra_pair_rollouts):
                        planner.search(state)
                        action_prob = planner.get_action_prob(state)
                        state_rep = planner._to_string_rep(state)
                        valid_moves = planner.valid_moves.get(state_rep, [])
                        pair = get_preference_pair(
                            action_prob,
                            state_rep,
                            dialog_acts,
                            valid_moves,
                            planner.realizations_Vs,
                        )
                        if pair is not None:
                            break
                    if pair is None:
                        logger.info(
                            "Not enough realizations to form preference pair after extra rollouts; skipping turn."
                        )
                        state = next_state
                        continue
                
                _, best_pair, worst_pair = pair
                sample_scores = planner.get_realization_traces(state, best_action)
                if sample_scores:
                    _log_line(
                        f"[Dialog {dialog_idx} | Turn {turn}] Sample scores (by realization): "
                        f"{json.dumps(sample_scores, ensure_ascii=False)}"
                    )

                _log_line(
                    f"[Dialog {dialog_idx} | Turn {turn}] History+Pref:\n{history_str}\n"
                    f"Chosen: {best_pair[0]} (V={best_pair[1]:.4f})\n"
                    f"Rejected: {worst_pair[0]} (V={worst_pair[1]:.4f})"
                )

                dialog_pairs.append(
                    {
                        "prompt": stringify_dialogue_context(state["dialogue_context"]),
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
                
                # print("done: ", done)
                if len(state['dialogue_context']) >= self.game_config.max_horizon:
                    break

            outcome = dialog_game.get_dialog_ended(state)
            if outcome == 1.0:
                preference_pairs.extend(dialog_pairs)
                # log full dialog transcript
                full_dialog = stringify_dialogue_context(state["dialogue_context"])
                _log_line(f"=== Dialog {dialog_idx} transcript ===\n{full_dialog}\n=== End Dialog {dialog_idx} ===")
            else:
                logger.debug("Dialog {} did not succeed (outcome={:.1f}); skipping its preference pairs.", dialog_idx, outcome)

        if preference_path and preference_pairs:
            with preference_path.open("w", encoding="utf-8") as f:
                for item in preference_pairs:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info("Wrote {} preference pairs to {}", len(preference_pairs), preference_path)

        # Overwrite dataset splits so DPO trainer can consume them directly.
        if preference_pairs:
            self.dataset.train_instances = preference_pairs
            self.dataset.dev_instances = []
            self.dataset.test_instances = []

        logger.info("Generated %d preference pairs from %d dialogs.", len(preference_pairs), len(cases))
        return preference_pairs
