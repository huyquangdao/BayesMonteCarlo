import os
from itertools import count
import time
from collections import defaultdict

import torch
from tqdm import tqdm
from loguru import logger as loguru_logger

from base.trainer import Trainer
from logger.wandb_logger import WanDBLogger

from config.constants import RECOMMENDATION, NEGOTIATION, EMOTIONAL_SUPPORT, SL_RATIO, SUCCESS_RATE, AVG_TURN, FAIRNESS, \
    TOXICITY, USER_REWARD, NEGOTIATION_GOAL2DESCRIPTION, ES_CONV_GOAL2DESCRIPTION, PERSUATION, P4G_GOAL2DESCRIPTION, DURECDIAL_GOAL2DESCRIPTION


class ProCOTTrainer(Trainer):

    def __init__(self, game_config, model_config, accelerator, game, model, offline_evaluator, online_evaluator,
                 loggers, generation_method=None):
        """
        constructor for class ProCOT model
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

    def predict(self, instance, action_mapping=None, goals = None, domain = None):
        """
        method that predicts the dialogue action given the dialogue history
        :param instance: the current state of the conversation
        :return: the predicted dialogue action
        """
        # for the recommendation scenario
        if self.game_config.name == RECOMMENDATION:
            action = self.model(instance['dialogue_context'],
                                target_item = instance['task_background']['target_topic'], # this one is for recommendation scenario,
                                goals = goals, # this one is for recommendation scenario.
                                domain = domain, # this one is for recommendation scenario
                                llm_pipeline = self.game_config.llm_pipeline,
                                terminators = self.game_config.terminators
                                )
        # for the other scenarios
        else:
            action = self.model(instance['dialogue_context'],
                                llm_pipeline = self.game_config.llm_pipeline,
                                terminators = self.game_config.terminators
                                )
        # try-catch for handling un-desired prompting behavior
        # if the predicted action is not in the considered list
        # we consider the action as inquire.
        try:
            action = action_mapping[action]
        except:
            # handling exception cases
            temp_action = "None"
            for k, v in self.model_config.action_mapping.items():
                if k.lower() in action.lower():
                    temp_action = v
            # Heuristic approaches
            # the better solution is to obtain more information
            if temp_action == "None":
                if self.game_config.name == RECOMMENDATION:
                    temp_action = "Q&A"                
                if self.game_config.name == NEGOTIATION:
                    temp_action = "inquire"
                elif self.game_config.name == EMOTIONAL_SUPPORT:
                    temp_action = "Question"
                elif self.game_config.name == PERSUATION:
                    temp_action = "greeting"    
            action = temp_action
        
        # rewriting the action
        if self.model_config.rewrite_action:
            if self.game_config.name == RECOMMENDATION:
                action = DURECDIAL_GOAL2DESCRIPTION[action]            
                
            elif self.game_config.name == NEGOTIATION:
                action = NEGOTIATION_GOAL2DESCRIPTION[action]
                
            elif self.game_config.name == EMOTIONAL_SUPPORT:
                action = ES_CONV_GOAL2DESCRIPTION[action]
            
            elif self.game_config.name == PERSUATION:
                action = P4G_GOAL2DESCRIPTION[action]
                    
            action = self.model.rewrite_action(instance['dialogue_context'],
                                               action,
                                               llm_pipeline = self.game_config.llm_pipeline,
                                               terminators = self.game_config.terminators
                                               )
            
        return action

    def online_test(self, cases, device=None, simulators=None, action_mapping=None, goals = None, domain = None):
        """
        method that evaluate the rl-finetuned model on the test set
        :param cases: a list of situations, target_items, e.g....
        :param device: the device that we use to train the model
        :param simulators: a set of simulators to train the rl agent
        :param action_mapping: a dictionary that map (goa, topic) to index
        :return:
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
                action = self.predict(state, 
                                    action_mapping,
                                    goals = goals,
                                    domain = domain
                                    )
                    
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
                            AVG_TURN: conv_turn,
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
        return results 
