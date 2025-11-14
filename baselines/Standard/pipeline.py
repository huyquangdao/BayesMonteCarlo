import random

from loguru import logger
import torch

from base.pipeline import Pipeline

from utils.game import create_cases
from text_gen.bart_generation import BARTGeneration


class StandardPromptPipeline(Pipeline):

    def execute(self):
        """
        function that executes the pipeline
        including supervised fine tuning, reinforcement learning and performance evaluation
        :return: the results
        """
        # run the online evaluation process
        if self.model_config.run_online_eval:
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
        return None, online_eval_results

    def inference(self, instance, action_mapping=None):
        """
        method that predict the output response given an particular input instance
        :param instance: the given input instance
        :return: a generated response in the format of text
        """
        predicted_action = self.trainer.predict(self.model, instance)
        return predicted_action


class StandardPromptPipelineForNegotiation(StandardPromptPipeline):
    def run_online_test(self, cases=None, simulators=None):
        """
        method that run the online evaluation on the test set.
        :return: the results of the online eval
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
            # truncating the target item set
            # test_target_items = test_target_items[:50]

            # get the simulators from the test set
            test_simulators = self.test_simulators

            # sample to make sure the number of test simulators equal to the number of test items
            # please carefully managae the random seed for fair performance comparison
            # test_simulators = random.sample(test_simulators, len(test_target_items))

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
                                               action_mapping=self.model_config.action_mapping
                                               )

            return results


class StandardPromptPipelineForEmotionalSupport(StandardPromptPipeline):

    def run_online_test(self, cases=None, simulators=None):
        """
        method that run the online evaluation on the test set.
        :return: the results of the online eval
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
            # truncating the target item set
            # test_target_items = test_target_items[:50]

            # get the simulators from the test set
            test_simulators = self.test_simulators

            # sample to make sure the number of test simulators equal to the number of test items
            # please carefully managae the random seed for fair performance comparison
            # test_simulators = random.sample(test_simulators, len(test_target_items))

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
                                               action_mapping=self.model_config.action_mapping
                                               )

            return results
