"""
Pipeline skeleton for Bayes-Adaptive LLM.
Modeled after `baselines/TRIP/pipeline.py` so we can swap pipelines with minimal changes
and later plug in MCTS-based preference generation plus DPO training.
"""

import os
import json
import random
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from loguru import logger

from base.pipeline import Pipeline
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

        system_response = self.generation_method.generate_response(state)
        user_response = self.user_simulator.respond(state)
        next_state = update_state_for_open_loop_mcts(
            state=state,
            action=action,
            system_response=system_response,
            user_response=user_response,
        )
        return next_state



class BayesAdaptiveLLMPipeline(Pipeline):

    def load_pretrained_model(self, is_rl: bool = False, is_last: bool = False):
        """
        Load the latest supervised/DPO checkpoint.
        RL support is not wired yet but the signature mirrors TRIP/PPDPP.
        """
        if is_rl:
            saved_model_path = os.path.join(self.model_config.saved_dir, "rl_model.pth")
        else:
            saved_model_path = os.path.join(self.model_config.saved_dir, "model.pth")

        if not os.path.exists(saved_model_path):
            raise FileNotFoundError("No pretrained model found at {}".format(saved_model_path))

        self.model = self.trainer.load_model(saved_model_path)

    def execute(self):
        """
        High-level pipeline:
        1) optional SFT
        2) optional preference generation via MCTS
        3) optional DPO training on generated pairs
        4) optional offline/online evaluation
        """
        offline_eval_results, online_eval_results, preference_pairs = None, None, None

        if getattr(self.model_config, "run_sft", False):
            logger.info("Running supervised fine-tuning ...")
            self.run_sft()

        if getattr(self.model_config, "run_preference_search", False):
            logger.info("Generating preference pairs with MCTS loop ...")
            preference_pairs = self.generate_preference_pairs_with_mcts()

        if getattr(self.model_config, "run_dpo", False):
            logger.info("Training with DPO on preference pairs ...")
            # assuming dataset already carries preference data or was just generated
            self.load_pretrained_model(is_rl=False)
            self.trainer.train_dpo(self.dataset, self.device)

        if getattr(self.model_config, "run_offline_eval", False):
            logger.info("Offline evaluation ...")
            self.load_pretrained_model(is_rl=False)
            offline_eval_results = self.run_offline_test()

        if getattr(self.model_config, "run_online_eval", False):
            logger.info("Online evaluation ...")
            self.load_pretrained_model(is_rl=False)
            online_eval_results = self.run_online_test()

        return offline_eval_results, online_eval_results, preference_pairs

    def run_offline_test(self):
        """
        Evaluate the model on a static test set.
        """
        self.trainer.offline_evaluator.reset()
        self.load_pretrained_model(is_rl=False)
        results = self.trainer.test(self.dataset)

        for lg in self.trainer.loggers:
            if not isinstance(lg, WanDBLogger):
                lg.record(results, "Test Set")
        return results

    def run_sft(self):
        """
        Supervised fine-tuning entrypoint.
        """
        return self.trainer.train_sft(self.dataset, self.device)

    def inference(self, instance: Dict[str, Any], action_mapping=None):
        """
        Predict next action for a single instance.
        """
        return self.trainer.predict(instance, action_mapping=action_mapping)

    def generate_preference_pairs_with_mcts(self) -> Sequence[Dict[str, Any]]:
        """
        Simulate persuasion dialogues with OpenLoopMCTS to extract preference pairs.
        The pairs are also written to disk if `model_config.preference_pairs_path` is provided,
        and the in-memory dataset is overwritten so DPO training can consume them directly.
        """
        if self.game_config.name != PERSUATION:
            logger.warning("Preference search is currently implemented for persuasion only; skipping.")
            return []

        if not self.dev_simulators and not self.test_simulators:
            raise ValueError("User simulators are required for MCTS preference generation.")

        simulators = self.dev_simulators or self.test_simulators
        if simulators is None or len(simulators) == 0:
            raise ValueError("No simulators available for preference generation.")

        # Action space / player
        action_mapping = self.dataset.construct_action_mapping(
            combine=self.model_config.combined_action if not getattr(self.game_config, "is_so_game", False) else False
        )
        # expose mapping to player via model_config for LLMPlayer compatibility
        setattr(self.model_config, "action_mapping", action_mapping)
        dialog_acts = [goal for goal, _ in sorted(action_mapping.items(), key=lambda kv: kv[1])]
        player = LLMPlayer(self.game_config, action_mapping, self.model_config)

        # MCTS configuration
        num_MCTS_sims = getattr(self.model_config, "num_mcts_sims", 15)
        max_realizations = getattr(self.model_config, "max_realizations", 5)
        max_turns = getattr(self.model_config, "max_turns", 5)
        mcts_cfg = SimpleNamespace(
            cpuct=1.0,
            Q_0=getattr(self.model_config, "Q_0", 0.25),
            max_realizations=max_realizations,
        )

        # Dialog seeds
        cases: List[Any] = list(self.dataset.train_instances)
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

        for dialog_idx, case in enumerate(cases):
            # fix persuadee (simulator/persona) per dialog, similar to TRIP
            simulator = random.choice(simulators)
            dialog_game = PersonaDialogGame(self.game, self.trainer.generation_method, simulator)
            state = self.game.reset(case, simulator)
            persona_history: List[Dict[str, str]] = []
            persona_hint = {}
            if hasattr(simulator, "user_profile_description"):
                raw_desc = getattr(simulator, "user_profile_description", "")
                persona_hint["description"] = sanitize_persona_description(raw_desc)
            if persona_hint:
                persona_history.append({"turn": 0, **persona_hint})

            dialog_pairs: List[Dict[str, Any]] = []
            for turn in range(max_turns):
                outcome = dialog_game.get_dialog_ended(state)
                if outcome != 0:
                    break

                planner = OpenLoopMCTS(dialog_game, player, mcts_cfg)
                for _ in range(num_MCTS_sims):
                    planner.search(state)

                action_prob = planner.get_action_prob(state)
                if np.sum(action_prob) == 0:
                    logger.debug("Zero action probability encountered; stopping dialog %s turn %s", dialog_idx, turn)
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
                next_state = dialog_game.get_next_state(state, goal)
                sys_utt = next_state["dialogue_context"][-2]["content"]
                user_utt = next_state["dialogue_context"][-1]["content"]

                # print full history up to current turn
                history_str = stringify_dialogue_context(next_state["dialogue_context"])

                pair = get_preference_pair(
                    action_prob,
                    state_rep,
                    dialog_acts,
                    valid_moves,
                    planner.realizations_Vs,
                )
                if pair is None:
                    logger.info("Not enough realizations to form preference pair; skipping turn.")
                    state = next_state
                    continue
                
                _, best_pair, worst_pair = pair

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

                state = next_state
                if dialog_game.get_dialog_ended(state) != 0:
                    break

            outcome = dialog_game.get_dialog_ended(state)
            if outcome == 1.0:
                preference_pairs.extend(dialog_pairs)
                # log full dialog transcript
                full_dialog = stringify_dialogue_context(state["dialogue_context"])
                _log_line(f"=== Dialog {dialog_idx} transcript ===\n{full_dialog}\n=== End Dialog {dialog_idx} ===")
            else:
                logger.debug("Dialog %s did not succeed (outcome=%.1f); skipping its preference pairs.", dialog_idx, outcome)

        if preference_path and preference_pairs:
            with preference_path.open("w", encoding="utf-8") as f:
                for item in preference_pairs:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            logger.info("Wrote %d preference pairs to %s", len(preference_pairs), preference_path)

        # Overwrite dataset splits so DPO trainer can consume them directly.
        if preference_pairs:
            self.dataset.train_instances = preference_pairs
            self.dataset.dev_instances = []
            self.dataset.test_instances = []

        logger.info("Generated %d preference pairs from %d dialogs.", len(preference_pairs), len(cases))
        return preference_pairs

    def run_online_test(self):
        """
        Stub for online evaluation with simulators.
        """
        raise NotImplementedError("Online evaluation is not implemented for Bayes-Adaptive LLM.")
