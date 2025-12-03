"""
Pipeline skeleton for Bayes-Adaptive LLM.
Modeled after `baselines/TRIP/pipeline.py` so we can swap pipelines with minimal changes
and later plug in MCTS-based preference generation plus DPO training.
"""

import os

from typing import Any, Dict
from logger.wandb_logger import WanDBLogger
from utils.game import create_target_set, create_cases

from bayes_adaptive_llm.utils import load_model 
import torch
import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split

from base.pipeline import Pipeline


class BayesAdaptiveLLMPipeline(Pipeline):

    def load_pretrained_model(self, is_rl: bool = False, is_last: bool = False):
        """
        Load the latest supervised/DPO (hoặc RL) checkpoint.
        """
        if is_rl:
            ckpt_name = "rl_model.pth"
        else:
            ckpt_name = "model.pth"

        saved_model_path = os.path.join(self.model_config.saved_dir, ckpt_name)

        if not os.path.exists(saved_model_path):
            raise FileNotFoundError(f"No pretrained model found at {saved_model_path}")

        self.trainer.model = self.model
        device = getattr(
            self,
            "device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

        self.model = load_model(saved_model_path, device=device)


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
            self.trainer.train_sft(self.dataset, self.device)

        if getattr(self.model_config, "run_preference_search", False):
            self.load_pretrained_model(is_rl=False)
            logger.info("Generating preference pairs with MCTS loop ...")
            preference_pairs = self.generate_preference_data()

        if getattr(self.model_config, "run_dpo", False):
            logger.info("Training with DPO on preference pairs ...")
            # assuming dataset already carries preference data or was just generated
            # self.load_pretrained_model(is_rl=False)
            pref_path = getattr(self.model_config, "preference_pairs_path", None)
            if not pref_path or not os.path.exists(pref_path):
                logger.warning("No preference pairs found or path does not exist; skipping DPO.")
                return
            self.trainer.train_dpo(pref_path, self.device)

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


    def inference(self, instance: Dict[str, Any], action_mapping=None):
        """
        Predict next action for a single instance.
        """
        return self.trainer.predict(instance, action_mapping=action_mapping)


    def run_online_test(self):
        """
        Stub for online evaluation with simulators.
        """
        raise NotImplementedError("Online evaluation is not implemented for Bayes-Adaptive LLM.")

    def generate_preference_data(self, dev_ratio = 0.2):
        """
        Stub for preference data generation.
        """
        # create training cases
        train_cases = create_cases(test_instances=self.dataset.train_instances,
                                    num_cases=self.dataset_config.num_train_cases
                                    )

        # split the simulators to train and dev simulators
        train_simulators, dev_simulators = train_test_split(self.dev_simulators,
                                                            test_size=dev_ratio,
                                                            random_state=self.game_config.seed
                                                            )

        action_mapping = self.dataset.construct_action_mapping(combine=self.model_config.combined_action)


        preference_pairs = self.trainer.generate_preference_pairs_with_mcts(
                                                                            train_cases,
                                                                            dev_simulators,
                                                                            action_mapping = action_mapping
                                                                            )
        return preference_pairs
