import time
from itertools import count

from tqdm import tqdm
import torch

from loguru import logger as loguru_logger

from base.trainer import Trainer
from logger.wandb_logger import WanDBLogger

from config.constants import RECOMMENDATION, NEGOTIATION, EMOTIONAL_SUPPORT, SUCCESS_RATE, AVG_TURN, \
    USER_REWARD, SL_RATIO, FAIRNESS, TOXICITY


class GDPZeroTrainer(Trainer):

    def __init__(self, game_config, model_config, accelerator, game, model, offline_evaluator, online_evaluator,
                 loggers, generation_method=None):
        """
        constructor for class PPDPP pipeline training
        :param game_config: the configuration of the game
        :param model_config: the configuration of the model
        :param accelerator: the accelerator
        :param game: instance of the game class
        :param model: instance of the model class
        :param offline_evaluator: instance of the offline evaluator class
        :param online_evaluator: an instance of the online evaluator class
        :param loggers: a set of loggers
        :param generation_method: an instance of the generation method
        """
        super().__init__(game_config, model_config, accelerator, game, model, offline_evaluator, online_evaluator,
                         loggers)
        self.generation_method = generation_method

    def construct_dataloaders(self, data_instances, batch_size, shuffle=True, num_workers=1):
        pass

    def create_criterion(self):
        pass

    def create_optimizer(self, learning_rate=1e-5):
        pass

    def eval_epoch(self, data_loader, criterion):
        pass

    def process_dataset(self, dataset):
        pass

    def train_epoch(self, data_loader, optimizer, lr_scheduler, criterion, max_train_steps):
        pass

    def train_sft(self, dataset, device=None):
        pass

    def predict(self, instance, simulator, action_mapping=None):
        """
        method that predicts the dialogue action given the dialogue history
        :param instance: the current state of the conversation
        :return: the predicted dialogue action
        """
        action = self.model(instance,
                            game=self.game,
                            generation_method=self.generation_method,
                            user_simulator=simulator,
                            action_mapping=action_mapping)
        
        # try-catch for handling un-desired prompting behavior
        # if the predicted action is not in the considered list
        # we consider the action as inquire.
        return action

    def online_test(self, cases, device=None, simulators=None, action_mapping=None):
        """
        method that evaluate the dialogue policy model on the test set
        :param cases: a list of situations, target_items, e.g....
        :param device: the device that we use to train the model
        :param simulators: a set of simulators to train the rl agent
        :param action_mapping: a dictionary
        :return:
        """

        loguru_logger.warning(f"Online Testing on Target Item in the Test Set ......")
        loguru_logger.warning(f"Num Target Items: {len(cases)}, Num Simulators: {len(simulators)}")
        # success rate and average number of conversation turns.
        SR, AvgT, total_reward = 0., 0., 0.

        # loop over the item set
        # make sure each item is associated with one user profile
        for idx, (case, simulator) in tqdm(enumerate(list(zip(cases, simulators)))):

            # randomly sample persona information
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

            # a flag to check if the conversation is successful
            # for computing the success rate
            is_successful = False
            conv_turn = 0

            # create two lists to store the rewards
            rewards = []

            # flag for checking if the target is mentioned during the conversation
            o_flag = False

            t1 = time.time()

            # interactive simulation
            for t in count():  # user  dialog

                # predict the action
                action = self.predict(state, simulator, action_mapping)
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

            epi_reward = torch.cat(epi_reward, dim=0)
            # objective-based epi reward
            objective_based_reward = epi_reward.sum(dim=0)
            # turn_reward = objective_based_reward[-1].item()

            # negotiation scenario
            if self.game_config.name == NEGOTIATION:
                # for the negotiation
                # the first objective is sl_ratio
                # the second objective is fairness
                epi_reward = epi_reward.mean(dim=0)
                sl_ratio_reward = epi_reward[0].item()
                fairness_reward = epi_reward[1].item()
                turn_reward = objective_based_reward[-1].item()

                # three objectives
                # i.e user reward, item_freq, turn_reward
                if len(objective_based_reward) == 2:
                    turn_reward = -1

                # we do not consider sl_ratio and fairness score for failed conversations
                # if not is_successful:
                #     sl_ratio_reward = 0
                #     fairness_reward = 0

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
            # TODO: to be modified
            elif self.game_config.name == EMOTIONAL_SUPPORT:
                # the first objective is the conversational sr
                # need to be normalized to [0,1]
                toxicity = objective_based_reward[1].item()
                user_reward = objective_based_reward[0].item() / epi_reward.shape[0]
                turn_reward = -1
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

            print("Running time: ", time.time() - t1)

        # compute the results using the evaluator
        results = self.online_evaluator.report()

        # log the results to terminal or file
        for logger in self.loggers:
            if not isinstance(logger, WanDBLogger):
                logger.record(results, "Testing")

        # return the results of the online evaluation
        return results
