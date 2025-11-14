import random
from loguru import logger
import torch
from sklearn.model_selection import train_test_split

from base.pipeline import Pipeline
from logger.wandb_logger import WanDBLogger

from utils.game import create_target_set, create_cases
from text_gen.bart_generation import BARTGeneration


class EnvelopePipeline(Pipeline):

    def run_sft(self):
        """
        This method run the whole pipeline for model training, selection and evaluation
        :return: the results of the current run.
        """
        # train and select the best model on the current dataset
        results = self.trainer.train_sft(self.dataset, self.device)

        # return the results of the model on the dev set
        return results

    def run_offline_test(self):
        """
        method that run the offline evaluation on the test set
        :return: the results of offline evaluation
        """
        # evaluate the performance on the test set
        # reset the offline evaluator
        self.trainer.offline_evaluator.reset()

        # load the best check point for testing
        self.load_pretrained_model(is_rl=False)

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

    def execute(self):
        """
        function that executes the pipeline
        including supervised fine tuning, reinforcement learning and performance evaluation
        :return: the results
        """
        offline_eval_results, online_eval_results = None, None

        # if we are in the testing phase
        # then we only perform the online evaluation on the testset
        if self.model_config.test_phase:
            # run the online evaluation process
            if self.model_config.run_online_eval:
                # if we're running the ablation study
                # we directly load the supervised finetuning model for online evaluation
                if self.model_config.ablation != '':
                    logger.info(f"Performing ablation study: {self.model_config.ablation} .....")
                    if 'rl' in self.model_config.ablation:
                        self.load_pretrained_model(is_rl=False)
                else:
                    logger.info("Loading the supervised RL fine-tuning model .....")
                    # first, we need to load the rl fine tuning model
                    self.load_pretrained_model(is_rl=True)

                logger.info("Performing Online Evaluation ....")
                # for testing, the response generation which uses either Chatgpt for Vicuna
                # should have a temperature of 0.0
                if not isinstance(self.trainer.generation_method, BARTGeneration):
                    self.trainer.generation_method.generation_config.set_params(
                        {
                            'temperature': 0.0,
                        }
                    )

                # then we fine tune the model with reinforcement learning
                online_eval_results = self.run_online_test()
        # we perform the full pipeline including supervised training
        # offline evaluation
        # rl fine-tuning
        # online evaluation
        else:
            # run the supervised finetuning process
            if self.model_config.run_sft:
                logger.info("Performing supervised fine-tuning on the background dataset ...")
                self.run_sft()
                self.trainer.global_step = 0

            if self.model_config.run_offline_eval:
                logger.info("Loading the supervised fine-tuning model .....")
                # first, we need to load the supervised fine tuning model
                self.load_pretrained_model(is_rl=False)
                logger.info("Performing Offline Evaluation ....")
                offline_eval_results = self.run_offline_test()

            # run the rl fine-tuning process
            if self.model_config.run_rlt:
                # re-assign the global step to zero
                self.trainer.global_step = 0

                logger.info("Loading the supervised fine-tuning model .....")
                # first, we need to load the supervised fine tuning model
                self.load_pretrained_model(is_rl=False)

                logger.info("Performing reinforcement learning fine-tuning ....")
                # then we fine tune the model with reinforcement learning
                self.run_rlt()

            # run the online evaluation process
            if self.model_config.run_online_eval:
                # loading the rl finetuned model for online evluation
                self.load_pretrained_model(is_rl=True)
                logger.info("Performing Online Evaluation ....")
                # for testing, the response generation which uses either Chatgpt for Vicuna
                # should have a temperature of 0.0
                if not isinstance(self.trainer.generation_method, BARTGeneration):
                    self.trainer.generation_method.generation_config.set_params(
                        {
                            'temperature': 0.0,
                        }
                    )

                # then we fine tune the model with reinforcement learning
                online_eval_results = self.run_online_test()

        # return the supervised tuning and rl tuning results
        return offline_eval_results, online_eval_results

    def inference(self, instance, action_mapping=None):
        """
        method that predict the output response given an particular input instance
        :param instance: the given input instance
        :return: a generated response in the format of text
        """
        predicted_action = self.trainer.predict(self.model, instance)
        return predicted_action


class EnvelopePipelineForRecommendation(EnvelopePipeline):

    def run_rlt(self, dev_ratio=0.1):
        """
        method that run the reinforcement learning fine tuning on the given dataset
        :return: the results on the given dataset
        """
        # process dataset
        # for reinforcement learning, we fine-tune the model on the dev set.
        # create the target set
        dev_target_items = create_target_set(self.dataset.train_convs,
                                             test_instances=self.dataset.dev_instances,
                                             num_items=self.dataset_config.num_dev_items)

        # construct the goal, topic mapping
        action_mapping = self.dataset.construct_action_mapping(combine=self.model_config.combined_action)

        # make sure we have a set of simulators for training the rl agent.
        assert self.dev_simulators is not None

        # compute split length
        # i.e 0.8 for train
        # 0.2 for testing
        # create the training and testing set
        # remember using the same random seed for each model
        train_items, dev_items = train_test_split(dev_target_items, test_size=dev_ratio,
                                                  random_state=self.game_config.seed)

        # split the simulators to train and dev simulators
        train_simulators, dev_simulators = train_test_split(self.dev_simulators, test_size=dev_ratio,
                                                            random_state=self.game_config.seed)

        # make sure the number of dev simulators is equal to the number of dev target items
        # this make sure a fair comparison between different models
        # please carefully manage the random seed
        if len(dev_simulators) > len(dev_items):
            dev_simulators = random.sample(dev_simulators, len(dev_items))

        # run the reinforcement learning fine tuning
        # # we use the target items in the dev set and user profile in the dev set
        results = self.trainer.train_rlt(cases=train_items,
                                         dev_cases=dev_items,
                                         device=self.device,
                                         simulators=train_simulators,
                                         dev_simulators=dev_simulators,
                                         action_mapping=action_mapping)

        # return the results of the rl fine tuning on the test set
        return results

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
        else:
            # creating the target item set
            test_target_items = create_target_set(self.dataset.train_convs,
                                                  test_instances=self.dataset.test_instances,
                                                  num_items=self.dataset_config.num_test_items)
            # truncating the target item set
            # test_target_items = test_target_items[:10]

            # get the simulators from the test set
            test_simulators = self.test_simulators

            # if we only with to perform only evaluation on a sub-set of the test set
            if self.model_config.num_test_cases:

                random.seed(self.game_config.seed)
                test_target_items = random.sample(test_target_items, self.model_config.num_test_cases) 

            # sample to make sure the number of test simulators equal to the number of test items
            # please carefully managae the random seed for fair performance comparison
            # test_simulators = random.sample(test_simulators, len(test_target_items))

        # test_target_items = test_target_items
        # construct the goal, topic mapping
        action_mapping = self.dataset.construct_action_mapping(combine=self.model_config.combined_action)

        # make sure the number of simulator equal to the number of target item
        # this make the performance comparison fair.
        # please manage the randon seed carefully.
        if len(test_simulators) > len(test_target_items):
            test_simulators = random.sample(test_simulators, len(test_target_items))

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
                                               action_mapping=action_mapping,
                                               stage='test',
                                               )

            return results


class EnvelopePipelineForNegotiation(EnvelopePipeline):

    def run_rlt(self, dev_ratio=0.1):
        """
        method that runs the reinforcement learning fine-tuning on the development set
        :param dev_ratio: the ratio of development set
        :return:
        """
        # process dataset
        # for reinforcement learning, we fine-tune the model on the dev set.
        # create the target cases
        dev_target_cases = create_cases(test_instances=self.dataset.dev_instances,
                                        num_cases=self.dataset_config.num_dev_cases)

        # construct the goal mapping
        action_mapping = self.dataset.construct_action_mapping(combine=self.model_config.combined_action)

        # make sure we have a set of simulators for training the rl agent.
        assert self.dev_simulators is not None

        # compute split length
        # i.e 0.8 for train
        # 0.2 for testing
        # create the training and testing set
        # remember using the same random seed for each model
        train_cases, dev_cases = train_test_split(dev_target_cases,
                                                  test_size=dev_ratio,
                                                  random_state=self.game_config.seed)

        # split the simulators to train and dev simulators
        train_simulators, dev_simulators = train_test_split(self.dev_simulators,
                                                            test_size=dev_ratio,
                                                            random_state=self.game_config.seed)

        # make sure the number of dev simulators is equal to the number of dev cases
        # this make sure a fair comparison between different models
        # please carefully manage the random seed
        if len(dev_simulators) > len(dev_cases):
            dev_simulators = random.sample(dev_simulators, len(dev_cases))

        # run the reinforcement learning fine tuning
        # # we use the target items in the dev set and user profile in the dev set
        results = self.trainer.train_rlt(cases=train_cases,
                                         dev_cases=dev_cases,
                                         device=self.device,
                                         simulators=train_simulators,
                                         dev_simulators=dev_simulators,
                                         action_mapping=action_mapping)

        # return the results of the rl fine tuning on the test set
        return results

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
            # test_cases = test_cases[:10]

            # # if we only wish to perform only evaluation on a sub-set of the test set
            # if self.model_config.num_test_cases:
            #     random.seed(self.game_config.seed)
            #     test_cases = random.sample(test_cases, self.model_config.num_test_cases) 


            # get the simulators from the test set
            test_simulators = self.test_simulators

            # sample to make sure the number of test simulators equal to the number of test items
            # please carefully managae the random seed for fair performance comparison
            # test_simulators = random.sample(test_simulators, len(test_target_items))

        # test_target_items = test_target_items
        # construct the goal, topic mapping
        action_mapping = self.dataset.construct_action_mapping(combine=self.model_config.combined_action)

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
                                               action_mapping=action_mapping,
                                               stage='test')

            return results


class EnvelopePipelineForEmotionalSupport(EnvelopePipeline):

    def run_rlt(self, dev_ratio=0.1):
        """
        method that runs the reinforcement learning fine-tuning on the development set
        :param dev_ratio: the ratio of development set
        :return:
        """
        # process dataset
        # for reinforcement learning, we fine-tune the model on the dev set.
        # create the target cases
        dev_target_cases = create_cases(test_instances=self.dataset.dev_instances,
                                        num_cases=self.dataset_config.num_dev_cases)

        # construct the goal mapping
        action_mapping = self.dataset.construct_action_mapping(combine=self.model_config.combined_action)

        # make sure we have a set of simulators for training the rl agent.
        assert self.dev_simulators is not None

        # compute split length
        # i.e 0.8 for train
        # 0.2 for testing
        # create the training and testing set
        # remember using the same random seed for each model
        train_cases, dev_cases = train_test_split(dev_target_cases,
                                                  test_size=dev_ratio,
                                                  random_state=self.game_config.seed)

        # split the simulators to train and dev simulators
        train_simulators, dev_simulators = train_test_split(self.dev_simulators,
                                                            test_size=dev_ratio,
                                                            random_state=self.game_config.seed)

        # make sure the number of dev simulators is equal to the number of dev cases
        # this make sure a fair comparison between different models
        # please carefully manage the random seed
        if len(dev_simulators) > len(dev_cases):
            dev_simulators = random.sample(dev_simulators, len(dev_cases))

        # run the reinforcement learning fine tuning
        # # we use the target items in the dev set and user profile in the dev set
        results = self.trainer.train_rlt(cases=train_cases,
                                         dev_cases=dev_cases,
                                         device=self.device,
                                         simulators=train_simulators,
                                         dev_simulators=dev_simulators,
                                         action_mapping=action_mapping)

        # return the results of the rl fine tuning on the test set
        return results

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
            # test_cases = test_cases[:20]

            # get the simulators from the test set
            test_simulators = self.test_simulators

            # sample to make sure the number of test simulators equal to the number of test items
            # please carefully managae the random seed for fair performance comparison
            # test_simulators = random.sample(test_simulators, len(test_target_items))

        # test_target_items = test_target_items
        # construct the goal, topic mapping
        action_mapping = self.dataset.construct_action_mapping(combine=self.model_config.combined_action)

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
                                               action_mapping=action_mapping,
                                               stage='test')

            return results
