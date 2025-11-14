import os
import random

from loguru import logger
import torch

from base.pipeline import Pipeline
from logger.wandb_logger import WanDBLogger

from utils.game import create_target_set, create_cases
from text_gen.bart_generation import BARTGeneration


class COLORPipeline(Pipeline):

    def load_pretrained_model(self, stage='planner'):
        """
        method that load the pretrained checkpoint from a particular stage
        :param stage: the stage that we wish to load the checkpoint
        :return: None
        """
        # create the model path
        # this is the path for sft model for a specific stage
        saved_model_path = os.path.join(self.model_config.saved_dir, stage, "model.pth")
        if not os.path.exists(saved_model_path):
            assert 1 == 0
        # load the model from the checkpoint
        self.trainer.load_model(saved_model_path)

    def execute(self):
        """
        function that executes the pipeline
        including supervised fine tuning, reinforcement learning and performance evaluation
        :return: the results
        """
        offline_eval_results, online_eval_results = None, None
        # run the supervised fintuning process
        if self.model_config.run_sft:
            logger.info("Performing supervised fine-tuning on the background dataset ...")
            self.run_sft()

        if self.model_config.run_offline_eval:
            # first, we need to load the supervised fine tuning model
            logger.info("Performing Offline Evaluation ....")
            offline_eval_results = self.run_offline_test()

        # run the online evaluation process
        if self.model_config.run_online_eval:
            logger.info("Loading the supervised fine-tuning model .....")
            # first, we need to load the supervised fine tuning model
            self.load_pretrained_model(stage='planner')

            # for testing, the response generation which uses either Chatgpt for Vicuna
            # should have a temperature of 0.0
            if not isinstance(self.trainer.generation_method, BARTGeneration):
                self.trainer.generation_method.generation_config.set_params(
                    {
                        'temperature': 0.0,
                    }
                )
            logger.info("Performing Online Evaluation ....")
            # then we fine tune the model with reinforcement learning
            online_eval_results = self.run_online_test()

        # return the supervised tuning and rl tuning results
        return offline_eval_results, online_eval_results

    def run_offline_test(self):
        """
        method that run the offline evaluation on the test set
        :return: the results of offline evaluation
        """
        # evaluate the performance on the test set
        # reset the offline evaluator
        self.trainer.offline_evaluator.reset()

        # load the best check point for testing
        self.load_pretrained_model(stage='planner')

        # compute the performance on test set and generated responses
        results = self.trainer.test(self.dataset)

        # logging the test results
        for logger in self.trainer.loggers:
            # log the results to file or terminal
            # currently it is not available for Wandb Logger
            if not isinstance(logger, WanDBLogger):
                logger.record(results, "Test Set")

        # return the results of the current run
        return results

    def run_sft(self):
        """
        This method run the whole pipeline for model training, selection and evaluation
        :return: the results of the current run.
        """

        # # freezing parameters if required
        # # borrowing from official COLOR implementation
        for name, param in self.trainer.model.named_parameters():
            if "freeze_plm" in name or "transform_layers" in name or "feature_conversion" in name or "feature_projection" in name or "feedback_estimation" in name:
                if self.model_config.freeze_plm and "freeze_plm" in name:
                    param.requires_grad = False
                else:
                    param.requires_grad = True
            else:
                param.requires_grad = False
        # for color model, we have two training stages
        # the first stage is the brownian bridge training stage
        # the second stage is the planner training stage
        # train the brownian bridge
        # select the best model on the current dataset
        logger.info("Training the Brownian Bridge ......")
        _ = self.trainer.train_sft(stage='bridge', dataset=self.dataset, device=self.device)

        # load the best model on the bridge training stage
        self.load_pretrained_model(stage='bridge')
        self.trainer.global_step = 0

        # freezing parameters if required
        # borrowing from official COLOR implementation
        for name, param in self.trainer.model.named_parameters():
            if "freeze_plm" in name or "feature_conversion" in name or "feedback_estimation" in name or "feature_projection" in name or "transform_layers" in name:
                param.requires_grad = False
            else:
                param.requires_grad = True

        logger.info("Training the COLOR planner ......")
        # train the planner and select the best model on the current dataset
        results = self.trainer.train_sft(stage='planner', dataset=self.dataset, device=self.device)

        # return the results of the model on the dev set
        return results

    def inference(self, instance, action_mapping=None):
        """
        method that predict the output response given an particular input instance
        :param instance: the given input instance
        :return: a generated response in the format of text
        """
        predicted_action = self.trainer.predict(self.model, instance)
        return predicted_action


class COLORPipelineForRecommendation(COLORPipeline):

    def run_online_test(self, target_items=None, simulators=None):
        """
        method that run the online evaluation on the test set.
        :return: the results of the online eval
        """
        # if we wish to run the model on a different target item set.
        # and a set of given user simulators
        if target_items is not None:
            test_target_items = target_items
            test_simulators = simulators
        # run the model on the truth test set
        else:
            # creating the target item set
            test_target_items = create_target_set(self.dataset.train_convs,
                                                  test_instances=self.dataset.test_instances,
                                                  num_items=self.dataset_config.num_test_items)
            # get the simulators from the test set
            test_simulators = self.test_simulators

            # sample to make sure the number of test simulators equal to the number of test items
            # please carefully managae the random seed for fair performance comparison
            test_simulators = random.sample(test_simulators, len(test_simulators))

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
            results = self.trainer.online_test(test_target_items,
                                               device=self.device,
                                               simulators=test_simulators,
                                               )

            return results
