"""
Pipeline skeleton for Bayes-Adaptive LLM.
Modeled after `baselines/TRIP/pipeline.py` so we can swap pipelines with minimal changes
and later plug in MCTS-based preference generation plus DPO training.
"""

import os
import random 
from typing import Any, Dict
from logger.wandb_logger import WanDBLogger
from utils.game import create_target_set, create_cases

from bayes_adaptive_llm.utils import load_legacy_checkpoint, has_meta_checkpoint, load_from_meta_checkpoint
import torch
import numpy as np
from loguru import logger
from sklearn.model_selection import train_test_split

from base.pipeline import Pipeline


class BayesAdaptiveLLMPipeline(Pipeline):

    def load_pretrained_model(self, model_dir, is_rl: bool = False) -> None:
        if model_dir is None:
            model_dir = self.model_config.saved_dir
        print("Loading pretrained model from:", model_dir)
        if has_meta_checkpoint(model_dir):
            load_from_meta_checkpoint(self, model_dir)
        else:
            load_legacy_checkpoint(self, model_dir, is_rl)


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
            self.load_pretrained_model(model_dir=self.model_config.saved_dir, is_rl=False)
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
            # self.load_pretrained_model(model_dir=self.model_config.dpo_adapter_path, is_rl=False)
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


    def run_online_test(self, cases = None, simulators = None):
        """
        Stub for online evaluation with simulators.
        """
                # if we wish to run the model on a different set of negotiation situations.
        # and a set of given user simulators
        if cases is not None:
            test_cases = cases
            test_simulators = simulators
        else:
            # creating the target item set
            test_cases = create_cases(test_instances=self.dataset.test_instances,
                                      num_cases=self.dataset_config.num_test_cases)

            # get the simulators from the test set
            test_simulators = self.test_simulators

            # sample to make sure the number of test simulators equal to the number of test items
            # please carefully managae the random seed for fair performance comparison
            # test_simulators = random.sample(test_simulators, len(test_target_items))

        # test_target_items = test_target_items
        # construct the goal, topic mapping
        action_mapping = self.dataset.construct_action_mapping(
            # for single objective game we dont need to combine the goals and (bins/topics)
            combine=self.model_config.combined_action if not self.game_config.is_so_game else False
        )

        # make sure the number of simulator equal to the number of target item
        # this make the performance comparison fair.
        # please manage the randon seed carefully.
        if len(test_simulators) > len(test_cases):
            test_simulators = random.sample(test_simulators, len(test_cases))

        # make sure there is no gradient-relevant computation
        with torch.no_grad():
            # there should be two kinds of evaluation
            # item centric: prompt 1 item to different users
            # user centric: prompt different items to 1 users
            # this should be implemented later
            # make sure the test simulators is not None
            assert self.test_simulators is not None
            # run online evaluation
            # run online test sequentially
            # i.e using 1 process.
            results = self.trainer.online_test(test_cases,
                                               device=self.device,
                                               simulators=test_simulators,
                                               action_mapping=action_mapping)

            return results
        

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
