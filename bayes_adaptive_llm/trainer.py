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
import pickle
import copy
import inspect
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
import torch.multiprocessing as mp
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


def _log_line_to_file(log_file: Optional[Path], text: str) -> None:
    """
    Append a single log line to a file, creating parent directories on demand.
    """
    if not log_file:
        return
    log_file = Path(log_file)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    append_to_log(log_file, [text])


def _select_preference_prompt(game_name: str) -> str:
    """
    Map game name to the preference prompt template.
    """
    if game_name == PERSUATION:
        return PREFERENCE_PAIR_PROMPT_P4G
    if game_name == NEGOTIATION:
        return PREFERENCE_PAIR_PROMPT_NEGOTIATION
    return ""


def _serialize_generation_method(generation_method: Any) -> Optional[Dict[str, Any]]:
    """
    Extract a picklable snapshot of the generation method so workers can
    re-instantiate it without sharing process state.
    """
    if generation_method is None:
        return None

    spec: Dict[str, Any] = {
        "cls": generation_method.__class__,
        "config": None,
        "pipeline": None,
        "is_test": getattr(generation_method, "is_test", True),
    }

    if hasattr(generation_method, "generation_config"):
        try:
            spec["config"] = copy.deepcopy(generation_method.generation_config)
        except Exception:
            spec["config"] = getattr(generation_method, "generation_config", None)

    if hasattr(generation_method, "pipeline"):
        pipeline = getattr(generation_method, "pipeline")
        try:
            pickle.dumps(pipeline)
            spec["pipeline"] = pipeline
        except Exception:
            spec["pipeline"] = None
    return spec


def _build_generation_method_from_spec(spec: Optional[Dict[str, Any]]) -> Any:
    """
    Rebuild a generation method instance from a serialized snapshot.
    """
    if not spec:
        return None

    gen_cls = spec.get("cls")
    if gen_cls is None:
        return None

    gen_config = copy.deepcopy(spec.get("config"))
    pipeline = spec.get("pipeline")
    is_test = spec.get("is_test", True)

    # try common constructor signatures
    for ctor in (
        lambda: gen_cls(gen_config, pipeline, is_test),
        lambda: gen_cls(gen_config, pipeline),
        lambda: gen_cls(gen_config),
        lambda: gen_cls(),
    ):
        try:
            return ctor()
        except TypeError:
            continue
        except Exception as exc:
            logger.warning("Failed to rebuild generation method in worker: {}", exc)
            continue
    return None


def _load_pairs_from_file(path: Path) -> List[Dict[str, Any]]:
    """
    Read preference pairs from a jsonl file.
    """
    pairs: List[Dict[str, Any]] = []
    if not path.exists():
        return pairs

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                pairs.append(json.loads(line))
            except JSONDecodeError as exc:
                logger.warning("Skipping malformed preference pair line in {}: {}", path, exc)
    return pairs


def _run_dialog_with_mcts(
    dialog_idx: int,
    case: Any,
    simulator: Any,
    player: LLMPlayer,
    dialog_acts: Sequence[str],
    num_MCTS_sims: int,
    mcts_cfg: SimpleNamespace,
    game: Any,
    generation_method: Any,
    prompt_prefix: str,
    max_horizon: int,
    skip_to_dialog_idx: int,
    log_file: Optional[Path],
) -> Tuple[List[Dict[str, Any]], float, Any]:
    """
    Run a single dialog episode with MCTS to collect preference pairs.
    Returns (pairs, outcome, final_state).
    """
    if dialog_idx < skip_to_dialog_idx:
        return [], 0.0, None

    dialog_game = PersonaDialogGame(
        game,
        generation_method,
        simulator,
        llm_pipeline=getattr(game.game_config, "llm_pipeline", None),
        terminators=getattr(game.game_config, "terminators", None),
    )

    state = game.reset(case, simulator)
    persona_hint = {}
    if hasattr(simulator, "user_profile_description"):
        raw_desc = getattr(simulator, "user_profile_description", "")
        persona_hint["description"] = sanitize_persona_description(raw_desc)

    dialog_pairs: List[Dict[str, Any]] = []
    for turn in count():
        outcome = dialog_game.get_dialog_ended(state)
        if outcome == 1.0 or outcome == -1.0:
            break

        logger.info(
            "Dialog {} turn {} | state={}",
            dialog_idx,
            turn,
            stringify_dialogue_context(state.get("dialogue_context", [])),
        )

        planner = OpenLoopMCTS(dialog_game, player, mcts_cfg)
        for _ in range(num_MCTS_sims):
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

        if valid_moves is not None and len(valid_moves) > 0:
            prob = action_prob.copy()
            prob = prob / prob.sum() if prob.sum() > 0 else np.ones_like(prob) / len(prob)
            best_action = int(np.random.choice(len(prob), p=prob))
        else:
            best_action = int(np.argmax(action_prob))
        goal = player.id2goal[best_action]

        prompt_dialogue_context = prompt_prefix + stringify_dialogue_context(state["dialogue_context"])
        state["dialog_id"] = dialog_idx
        state["turn_id"] = turn

        next_state, _, _, _ = game.step(state, goal, generation_method, simulator)
        sys_utt = next_state["dialogue_context"][-2]["content"]
        user_utt = next_state["dialogue_context"][-1]["content"]

        pair = get_preference_pair(
            action_prob,
            state_rep,
            dialog_acts,
            valid_moves,
            planner.realizations_Vs,
            selected_action=best_action,
        )

        history_str = stringify_dialogue_context(next_state["dialogue_context"])
        if pair is None:
            logger.info("Not enough realizations to form preference pair; skipping turn.")
            state = next_state
            if len(state["dialogue_context"]) >= max_horizon:
                break
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

        _log_line_to_file(
            log_file,
            f"[Dialog {dialog_idx} | Turn {turn}] History+Pref:\n{history_str}\n"
            f"Chosen: {best_pair[0]} (V={best_pair[1]:.4f})\n"
            f"Rejected: {worst_pair[0]} (V={worst_pair[1]:.4f})",
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

        state = next_state
        if len(state["dialogue_context"]) >= max_horizon:
            break

    outcome = dialog_game.get_dialog_ended(state)
    if outcome > 0.3:
        full_dialog = stringify_dialogue_context(state["dialogue_context"])
        _log_line_to_file(
            log_file,
            f"=== Dialog {dialog_idx} transcript ===\n{full_dialog}\n=== End Dialog {dialog_idx} ===",
        )
    else:
        logger.debug(
            "Dialog {} did not succeed (outcome={:.1f}); skipping its preference pairs.",
            dialog_idx,
            outcome,
        )
    return dialog_pairs, outcome, state


def _preference_worker_main(
    rank: int,
    world_size: int,
    simulators: Sequence[Any],
    action_mapping: Dict[str, int],
    model_config: Any,
    game_config: Any,
    game_cls: Any,
    dataset_config: Any,
    generation_method_spec: Optional[Dict[str, Any]],
    out_dir: str,
    prompt_prefix: str,
    skip_to_dialog_idx: int,
    max_horizon: int,
    tasks_by_rank: Sequence[Sequence[Tuple[int, Any]]],
):
    """
    Worker entry point for multiprocessing preference generation.
    """
    if torch.cuda.is_available():
        try:
            torch.cuda.set_device(rank)
        except Exception as exc:
            logger.warning("Worker {} failed to pin CUDA device: {}", rank, exc)

    local_model_config = copy.deepcopy(model_config)
    setattr(local_model_config, "action_mapping", action_mapping)
    setattr(local_model_config, "llm_pipeline", getattr(game_config, "llm_pipeline", None))
    setattr(local_model_config, "terminators", getattr(game_config, "terminators", None))

    dialog_acts = [goal for goal, _ in sorted(action_mapping.items(), key=lambda kv: kv[1])]
    player = LLMPlayer(game_config, action_mapping, local_model_config)

    num_MCTS_sims = getattr(local_model_config, "num_mcts_sims", 30)
    max_realizations = getattr(local_model_config, "max_realizations", 3)
    mcts_cfg = SimpleNamespace(
        cpuct=1.0,
        Q_0=getattr(local_model_config, "Q_0", 0.25),
        max_realizations=max_realizations,
    )

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    worker_path = out_dir / f"pref_pairs_rank{rank}.jsonl"
    log_file = out_dir / f"pref_gen_rank{rank}.log"

    generation_method = _build_generation_method_from_spec(generation_method_spec)
    if generation_method is None:
        raise RuntimeError("Generation method could not be rebuilt in worker.")

    game = game_cls(game_config, dataset_config)

    tasks = tasks_by_rank[rank] if rank < len(tasks_by_rank) else []
    for dialog_idx, case in tasks:
        simulator = random.choice(simulators)
        dialog_pairs, outcome, _ = _run_dialog_with_mcts(
            dialog_idx=dialog_idx,
            case=case,
            simulator=simulator,
            player=player,
            dialog_acts=dialog_acts,
            num_MCTS_sims=num_MCTS_sims,
            mcts_cfg=mcts_cfg,
            game=game,
            generation_method=generation_method,
            prompt_prefix=prompt_prefix,
            max_horizon=max_horizon,
            skip_to_dialog_idx=skip_to_dialog_idx,
            log_file=log_file,
        )

        if outcome > 0.3 and dialog_pairs:
            with worker_path.open("a", encoding="utf-8") as f:
                for item in dialog_pairs:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

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
        prompt_preference_by_game = _select_preference_prompt(self.game_config.name)

        if not dev_simulators:
            raise ValueError("User simulators are required for MCTS preference generation.")

        simulators = dev_simulators
        if simulators is None or len(simulators) == 0:
            raise ValueError("No simulators available for preference generation.")

        if len(train_cases) == 0:
            logger.warning("No train cases provided; skipping preference generation.")
            return []

        # expose mapping to player via model_config for LLMPlayer compatibility
        setattr(self.model_config, "action_mapping", action_mapping)
        setattr(self.model_config, "llm_pipeline", getattr(self.game_config, "llm_pipeline", None))
        setattr(self.model_config, "terminators", getattr(self.game_config, "terminators", None))
        logger.info("Action mapping: {}", action_mapping)

        dialog_acts = [goal for goal, _ in sorted(action_mapping.items(), key=lambda kv: kv[1])]
        player = LLMPlayer(self.game_config, action_mapping, self.model_config)

        # MCTS configuration
        num_MCTS_sims = getattr(self.model_config, "num_mcts_sims", 30)
        max_realizations = getattr(self.model_config, "max_realizations", 3)
        mcts_cfg = SimpleNamespace(
            cpuct=1.0,
            Q_0=getattr(self.model_config, "Q_0", 0.25),
            max_realizations=max_realizations,
        )

        skip_to_dialog_idx = getattr(self.model_config, "skip_to_dialog_idx", 120)
        max_horizon = getattr(self.game_config, "max_horizon", 10)
        dataset_config = getattr(self.game, "dataset_config", None)
        tasks = [(idx, case) for idx, case in enumerate(train_cases) if idx >= skip_to_dialog_idx]
        if not tasks:
            logger.warning("No tasks to process after applying skip_to_dialog_idx={}; nothing to generate.", skip_to_dialog_idx)
            return []

        preference_pairs: List[Dict[str, Any]] = []
        preference_path: Optional[Path] = None
        if getattr(self.model_config, "preference_pairs_path", None):
            preference_path = Path(self.model_config.preference_pairs_path)
            preference_path.parent.mkdir(parents=True, exist_ok=True)
            preference_path.write_text("", encoding="utf-8")

        out_dir = preference_path.parent if preference_path else Path("preference_out")
        out_dir.mkdir(parents=True, exist_ok=True)
        merged_path = preference_path if preference_path else out_dir / "preference_pairs_merged.jsonl"

        generation_method_spec = _serialize_generation_method(self.generation_method)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        requested_workers = getattr(self.model_config, "pref_generation_workers", 2)
        world_size = requested_workers or torch.cuda.device_count()
        world_size = 1 if world_size is None else int(world_size)
        world_size = min(world_size, len(tasks)) if len(tasks) > 0 else 1
        worker_tasks = [tasks[i::world_size] for i in range(world_size)]

        can_use_mp = world_size > 1
        if can_use_mp:
            try:
                pickle.dumps(
                    (
                        self.model_config,
                        self.game_config,
                        dataset_config,
                        worker_tasks,
                        generation_method_spec,
                        action_mapping,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Multiprocess preference generation disabled; context not picklable: {}", exc
                )
                can_use_mp = False

        if can_use_mp:
            mp.set_start_method("spawn", force=True)
            try:
                mp.spawn(
                    _preference_worker_main,
                    args=(
                        world_size,
                        simulators,
                        action_mapping,
                        self.model_config,
                        self.game_config,
                        self.game.__class__,
                        dataset_config,
                        generation_method_spec,
                        str(out_dir),
                        prompt_preference_by_game,
                        skip_to_dialog_idx,
                        max_horizon,
                        worker_tasks,
                    ),
                    nprocs=world_size,
                    join=True,
                )
                with merged_path.open("w", encoding="utf-8") as out:
                    for r in range(world_size):
                        p = out_dir / f"pref_pairs_rank{r}.jsonl"
                        if p.exists():
                            out.write(p.read_text(encoding="utf-8"))
                preference_pairs = _load_pairs_from_file(merged_path)
                logger.info("Merged preference pairs to {}", merged_path)
            except Exception as exc:
                logger.warning(
                    "Multiprocess preference generation failed, falling back to single process: {}", exc
                )
                can_use_mp = False
        if not can_use_mp:
            log_file = out_dir / f"pref_gen_{timestamp}.log"
            for dialog_idx, case in tqdm(tasks, desc="Generating preference pairs"):
                simulator = random.choice(simulators)
                dialog_pairs, outcome, _ = _run_dialog_with_mcts(
                    dialog_idx=dialog_idx,
                    case=case,
                    simulator=simulator,
                    player=player,
                    dialog_acts=dialog_acts,
                    num_MCTS_sims=num_MCTS_sims,
                    mcts_cfg=mcts_cfg,
                    game=self.game,
                    generation_method=self.generation_method,
                    prompt_prefix=prompt_preference_by_game,
                    max_horizon=max_horizon,
                    skip_to_dialog_idx=skip_to_dialog_idx,
                    log_file=log_file,
                )

                if outcome > 0.3 and dialog_pairs:
                    preference_pairs.extend(dialog_pairs)
                    if preference_path and dialog_pairs:
                        with preference_path.open("a", encoding="utf-8") as f:
                            for item in dialog_pairs:
                                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                        logger.info("Appended {} pairs from dialog {} to {}", len(dialog_pairs), dialog_idx, preference_path)

        if preference_path and preference_pairs:
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


