import os
import random

from loguru import logger
import torch

from base.pipeline import Pipeline
from logger.wandb_logger import WanDBLogger

from utils.game import create_target_set
from text_gen.bart_generation import BARTGeneration


class UNIMINDPipeline(Pipeline):

    def load_pretrained_model(self, task='goal'):
        """
        method that laod the model checkpoint to the current model class
        :param task: if we load the rl pretrained model
        :return: None
        """
        # create the model path
        # this is the path for sft model for a specific task
        if task != '':
            saved_model_path = os.path.join(self.model_config.saved_dir, task, "model.pth")
        # loading the pretraining checkpoint
        else:
            saved_model_path = os.path.join(self.model_config.saved_dir, "model.pth")
        # load the model from the checkpoint
        model = self.trainer.load_model(saved_model_path)
        return model

    def execute(self):
        """
        function that executes the pipeline
        including supervised fine tuning and performance evaluation
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
            # for unimind, we need to load both the goal and topic model
            # loading the goal model ....
            self.trainer.model_goal = self.load_pretrained_model(task='goal')

            # loading the topic model
            self.trainer.model_topic = self.load_pretrained_model(task="topic")

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
        overall_results = []
        for task in self.model_config.tasks:
            # evaluate the performance on the test set
            # reset the offline evaluator
            self.trainer.offline_evaluator.reset()

            # load the best check point for testing
            # this load the checkpoint of a specific task
            logger.info(f"Loading the checkpoint for [TASK]: {task}")
            self.model = self.load_pretrained_model(task=task)

            # compute the performance on test set and generated responses
            results = self.trainer.test([task], self.dataset)

            # logging the test results
            for l in self.trainer.loggers:
                # log the results to file or terminal
                # currently it is not available for Wandb Logger
                if not isinstance(l, WanDBLogger):
                    l.record(results, f"{task}/Test Set")

            # get the results of the current task
            overall_results.append(results)

        # return the results of the current run
        return overall_results

    def run_sft(self):
        """
        This method run the whole pipeline for model training, selection and evaluation
        :return: the results of the current run.
        """
        # train and select the best model on the current dataset
        # pretraining the model on 3 different tasks
        if self.model_config.do_pretrain:
            logger.info("Performing the pretraining Step .....")
            results = self.trainer.train_sft(tasks=self.model_config.tasks,
                                             epochs=self.model_config.num_train_epochs,
                                             dataset=self.dataset,
                                             device=self.device)

        # then finetune the model on each task individually
        if self.model_config.do_finetune:
            logger.info("Performing the Fine-tuning Step .....")
            for task in self.model_config.tasks:
                # re-assign the global step
                self.trainer.global_step = 0
                # loading the best checkpoint for model fine-tuning
                # here we load the best pretraining checkpoint
                logger.info("Loading the pretrained checkpoint .....")
                self.model = self.load_pretrained_model(task='')

                # performing the fine-tuning stage for each task
                logger.info(f"Fine-tuning on the [TASK]: {task} .....")
                results = self.trainer.train_sft(tasks=[task],
                                                 epochs=self.model_config.num_finetune_epochs,
                                                 dataset=self.dataset,
                                                 device=self.device)

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


class UNIMINDPipelineForRecommendation(UNIMINDPipeline):

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
