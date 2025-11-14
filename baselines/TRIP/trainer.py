import copy
import math
import os
import gc
import random
from itertools import count
from collections import defaultdict

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim import AdamW

from loguru import logger as loguru_logger
from transformers import get_linear_schedule_with_warmup

from baselines.TRIP.data_processor import TRIPDataProcessorForRecommendation, TRIPTorchDatasetForRecommendation, \
    TRIPTorchDatasetForNegotiation, TRIPDataProcessorForNegotiation, TRIPDataProcessorForEmotionalSupport, \
    TRIPTorchDatasetForEmotionalSupport, TRIPDataProcessorForPersuation, TRIPTorchDatasetForPersuation
    
from base.trainer import Trainer
from logger.wandb_logger import WanDBLogger
from logger.file_logger import FileLogger


from config.constants import RECOMMENDATION, NEGOTIATION, EMOTIONAL_SUPPORT, SL_RATIO, SUCCESS_RATE, AVG_TURN, FAIRNESS, \
    TOXICITY, ITEM_FREQ, USER_REWARD, PERSUATION, P4G_GOAL2DESCRIPTION, NEGOTIATION_GOAL2DESCRIPTION, ES_CONV_GOAL2DESCRIPTION, P4G_GOAL2DESCRIPTION

from utils.game import random_weights, save_conversation_for_human_evaluation


def calculate_gae(values, rewards, dones, next_values, gamma, lambd):
    # Calculate TD errors.
    deltas = rewards + gamma * next_values * (1 - dones) - values
    # Initialize gae.
    gaes = torch.empty_like(rewards)

    # Calculate gae recursively from behind.
    gaes[-1] = deltas[-1]
    for t in reversed(range(rewards.size(0) - 1)):
        gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]

    return gaes + values, (gaes - gaes.mean()) / (gaes.std() + 1e-8)


def release_memory(rank):
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    print(torch.cuda.memory_summary(rank))
    from pickle import dump
    snapshot = torch.cuda.memory._snapshot()
    if os.path.exists("snapshot.pickle"):
        if os.path.exists("snapshot.pickle.0"):
            os.remove("snapshot.pickle.0")
        os.rename("snapshot.pickle", "snapshot.pickle.0")
    dump(snapshot, open('snapshot.pickle', 'wb'))


class TRIPTrainer(Trainer):

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
        self.tokenizer = self.model.tokenizer

    def process_dataset(self, dataset):
        """
        method that process the given dataset and return processed data instances
        :return: processed data instances.
        """
        return dataset.train_instances, dataset.dev_instances, dataset.test_instances

    def construct_dataloaders(self, data_instances, batch_size, goal2id, shuffle=True, num_workers=1):
        """
        method that constructs dataloaders using given processed data instances
        :param data_instances: the processed data instances
        :param batch_size: number of batch size
        :param goal2id: a dictionary that map categorical goals to indexes
        :param shuffle: True if we shuffle the data set
        :param num_workers: number of workers used for loading the dataset
        :return: a instance of torch dataloader class
        """
        # recommendation scenario
        if self.game_config.name == RECOMMENDATION:
            torch_dataset = TRIPTorchDatasetForRecommendation(
                tokenizer=self.tokenizer,
                instances=data_instances,
                goal2id=goal2id,
                max_sequence_length=self.model_config.max_sequence_length,
                device=self.device,
                convert_example_to_feature=TRIPDataProcessorForRecommendation()
            )
        # negotiation scenario
        elif self.game_config.name == NEGOTIATION:
            torch_dataset = TRIPTorchDatasetForNegotiation(
                tokenizer=self.tokenizer,
                instances=data_instances,
                goal2id=goal2id,
                max_sequence_length=self.model_config.max_sequence_length,
                device=self.device,
                convert_example_to_feature=TRIPDataProcessorForNegotiation()
            )
        # emotional support conversation
        elif self.game_config.name == EMOTIONAL_SUPPORT:
            torch_dataset = TRIPTorchDatasetForEmotionalSupport(
                tokenizer=self.tokenizer,
                instances=data_instances,
                goal2id=goal2id,
                max_sequence_length=self.model_config.max_sequence_length,
                device=self.device,
                convert_example_to_feature=TRIPDataProcessorForEmotionalSupport()
            )
        # persuasion conversations
        elif self.game_config.name == PERSUATION:
            torch_dataset = TRIPTorchDatasetForPersuation(
                    tokenizer=self.tokenizer,
                    instances=data_instances,
                    goal2id=goal2id,
                    max_sequence_length=self.model_config.max_sequence_length,
                    device=self.device,
                    convert_example_to_feature=TRIPDataProcessorForPersuation()
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
        optimizer = AdamW(optimizer_grouped_parameters, lr=learning_rate)
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
        for step, batch in enumerate(data_loader):
            logits = self.model(batch)
            loss = criterion(logits, batch['labels']) / self.model_config.gradient_accumulation_steps
            self.accelerator.backward(loss)
            train_loss.append(float(loss))

            self.progress_bar.update(1)
            self.global_step += 1

            # optim step
            if step % self.model_config.gradient_accumulation_steps == 0 or step == len(data_loader) - 1:
                if self.model_config.max_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), self.model_config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if self.global_step >= max_train_steps:
                stop = True
                break

        # compute average train loss
        train_loss = np.mean(train_loss) * self.model_config.gradient_accumulation_steps
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

        dev_loss = np.mean(dev_loss) * self.model_config.gradient_accumulation_steps
        results = self.offline_evaluator.report()
        results['loss'] = dev_loss
        return results

    def train_sft(self, dataset, device=None):
        """
        method that train the model in an supervised tuning manner
        :param device: the device we use to train the model
        :return: None
        """

        # process dataset
        train_instances, dev_instances, test_instances = self.process_dataset(dataset)

        action_mapping = dataset.construct_action_mapping(
            # for single objective game we dont need to combine the goals and (bins/topics)
            combine=self.model_config.combined_action if not self.game_config.is_so_game else False
        )

        # create train, dev and test dataloaders
        train_loader = self.construct_dataloaders(train_instances,
                                                  batch_size=self.model_config.per_device_train_batch_size,
                                                  goal2id=action_mapping,
                                                  shuffle=True, num_workers=self.model_config.num_workers)

        dev_loader = self.construct_dataloaders(dev_instances, batch_size=self.model_config.per_device_eval_batch_size,
                                                goal2id=action_mapping,
                                                shuffle=False, num_workers=self.model_config.num_workers)

        best_loss = math.inf
        # create the optimizer
        optimizer = self.create_optimizer(self.model, self.model_config.learning_rate)

        # prepare the model
        self.model, optimizer, train_dataloader = self.accelerator.prepare(self.model, optimizer, train_loader)

        # compute the maximum number of training steps
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.model_config.gradient_accumulation_steps)
        max_train_steps = self.model_config.num_train_epochs * num_update_steps_per_epoch

        # create the learning rate scheduler
        lr_scheduler = self.create_scheduler(optimizer, self.model_config.num_warmup_steps, max_train_steps)

        # create the loss function
        self.criterion = self.create_criterion()

        # progress bar
        self.progress_bar = tqdm(range(max_train_steps), disable=not self.accelerator.is_local_main_process)

        # train the model
        self.model.to(device)
        for epoch in range(self.model_config.num_train_epochs):
            self.model.train()

            # reset the offline evaluator before each training epoch
            self.offline_evaluator.reset()

            # train the modelt
            train_loss, stop = self.train_epoch(data_loader=train_loader, optimizer=optimizer,
                                                lr_scheduler=lr_scheduler,
                                                criterion=self.criterion,
                                                max_train_steps=max_train_steps
                                                )
            # evaluate the performance
            results = self.eval_epoch(dev_loader, self.criterion)

            # logging the results
            for logger in self.loggers:
                logger.record(results, epoch + 1)

            # saving the model if needed
            # for generation, we use the loss as the saving criterion
            if results['loss'] < best_loss:
                loguru_logger.info("Performance improved. Saving the model .....")
                best_loss = results['loss']
                
                if self.game_config.name == RECOMMENDATION:
                    file_path = os.path.join(self.model_config.saved_dir, f"model_{self.model_config.domain}.pth")
                    
                elif self.game_config.name == NEGOTIATION:
                    file_path = os.path.join(self.model_config.saved_dir, f"model.pth")
                
                elif self.game_config.name == EMOTIONAL_SUPPORT:
                    file_path = os.path.join(self.model_config.saved_dir, f"model.pth")
                
                elif self.game_config.name == PERSUATION:
                    file_path = os.path.join(self.model_config.saved_dir, f"model.pth")

                self.save_model(file_path)

            if stop:
                loguru_logger.info("Training process is completed.")
                break

    def train_rlt(self, cases, dev_cases=None, device=None, simulators=None, dev_simulators=None, action_mapping=None):
        """
        method that train the model in a reinforcement learning manner
        :param cases: a list of situations, target_items, e.g....
        :param dev_cases: a list contains dev situations.
        :param device: the device that we use to train the model
        :param simulators: a set of simulators to train the rl agent
        :param dev_simulators: aa set of simulators used for model selection
        :param action_mapping: a dictionary that map (goa, topic) to index
        :return: None
        """
        # best metics
        best_metric = - math.inf
        # create the optimizer for rl training
        optimizer = self.create_optimizer(self.model, 
                                          self.model_config.rl_learning_rate)

        # using accelerator to prepare the model and optimizer
        # self.model, optimizer = self.accelerator.prepare(self.model, optimizer)
        self.model.to(self.device)
        
        # data for training generation
        data_for_generation_finetuning = []
        
        # loop for the number of epoch
        for train_step in range(1, self.model_config.num_train_rl_epochs + 1):

            # success rate and average number of conversation turns.
            SR, AvgT, total_reward = 0., 0., 0.

            # avg_sub_reward, avg_obj_reward, avg_turn_reward
            # avg_sub_reward, avg_obj_reward, avg_turn_reward = 0.0, 0.0, 0.0

            # a tensor which is used to store the loss
            loss = torch.tensor(0, dtype=torch.float, device=device)
            for i_episode in tqdm(range(self.model_config.sampled_times), desc='sampling'):

                # randomly sample one case
                case = np.random.choice(cases)

                # randomly sample persona information
                simulator = np.random.choice(simulators)

                loguru_logger.info('\n================New Episode:{}===================='.format(i_episode))

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

                # flag for checking if the target is mentioned during the conversation
                o_flag = False

                # multi objective game
                if not self.game_config.is_so_game:
                    if self.model_config.prioritized_objective == "uniform":
                        w = random_weights(self.model_config.n_objectives, dist="uniform")
                    
                    elif self.model_config.prioritized_objective != "uniform":
                        w = np.array(self.model_config.obj_to_weight[self.model_config.prioritized_objective.strip()])
                    
                    # evaluate using a given objective weight
                    elif self.model_config.objective_weight is not None:
                        w = np.array(self.model_config.objective_weight)
                        
                    # set the weight for the prioritized objective
                    # e.g: if the prioritized objective is sl_ratio then the corresponding weight is [0.9, 0.1, 0.1]
                    else:
                        pass
                    
                    self.model_config.objective_weight = w
                    loguru_logger.info(f"Objective Weight: [{w}]")
                    
                # single objective game
                else:
                    self.model_config.objective_weight = None
                
                all_states = []
                # interactive simulation
                
                for t in count():  # user  dialog
                    action, log_prob = self.predict(state, action_mapping)
                    
                    log_probs.append(log_prob)
                    state, reward, done, o_done = self.game.step(state, 
                                                                 action, 
                                                                 self.generation_method, 
                                                                 simulator)
                    
                    # check if target is mentioned during the conversation
                    if o_done == 1:
                        o_flag = True

                    # storing the reward
                    reward = torch.tensor([reward], device=device, dtype=torch.float)
                    rewards.append(reward)
                    epi_reward.append(reward)

                    # store the states for later use
                    all_states.append(copy.deepcopy(state))

                    # storing the log prob
                    log_probs.append(log_prob)
                    
                    # evaluate the outcome of the conversation
                    if done:
                        # successful case
                        # if the sub_reward is greater than epsilon
                        # and the target is mentioned in the conversation
                        if done == 1 and o_flag:
                            # increase the SR
                            SR += 1
                        AvgT += t + 1
                        conv_turn = len(state['dialogue_context'])
                        # total_reward += epi_reward
                        break

                # optimizing the policy model
                optimizer.zero_grad()
                
                # compute the loss function, e.g a proxy of the policy gradient
                newloss = self.compute_rl_policy_loss(rewards, 
                                                      log_probs)
                
                newloss.backward()
                optimizer.step()

                # log the results
                # this is the training results
                if newloss is not None:
                    loss += newloss

                    # multi objective game:
                    if not self.game_config.is_so_game:   
                        # construct the epi reward tensor
                        epi_reward = torch.cat(epi_reward, dim=0)

                        # objective-based epi reward
                        objective_based_reward = epi_reward.mean(dim=0)
                        
                        # scalarized reward
                        # computing the scalarized reward
                        # scalarized_reward = objective_based_reward.sum()
                        # applying weight-based scalarization
                        scalarized_reward = (torch.Tensor(self.model_config.objective_weight) * objective_based_reward.cpu()).sum()

                        # accumulate the rewards
                        # more 2 objectives
                        if len(objective_based_reward) > 2:
                            avg_sub_reward = objective_based_reward[0]
                            avg_obj_reward = objective_based_reward[1]
                            avg_turn_reward = objective_based_reward[2]
                        # only 2 objectives
                        else:
                            avg_sub_reward = objective_based_reward[0]
                            avg_obj_reward = objective_based_reward[1]
                            avg_turn_reward = -1

                        total_reward = scalarized_reward
                        if self.game_config.name == NEGOTIATION:
                            results = {
                                "loss": newloss,
                                "total_reward": total_reward,
                                SL_RATIO: avg_sub_reward,
                                FAIRNESS: avg_obj_reward,
                                "Avg_turn_reward": avg_turn_reward
                            }
                        elif self.game_config.name == RECOMMENDATION:
                            results = {
                                "loss": newloss,
                                "total_reward": total_reward,
                                USER_REWARD: avg_sub_reward,
                                ITEM_FREQ: avg_obj_reward,
                                "Avg_turn_reward": avg_turn_reward
                            }
                    # single objective game:
                    else:
                        epi_reward = torch.cat(epi_reward, dim=0)
                        # objective-based epi reward
                        
                        epi_reward = epi_reward.sum(dim=0)
                        total_reward = epi_reward.item()
                        
                        if self.game_config.name == NEGOTIATION:
                            results = {
                                "loss": newloss,
                                "total_reward": total_reward,
                                SL_RATIO: total_reward,
                                AVG_TURN: conv_turn
                            }
                        elif self.game_config.name == RECOMMENDATION:
                            results = {
                                "loss": newloss,
                                "total_reward": total_reward,
                                AVG_TURN: conv_turn
                            } 
                        elif self.game_config.name == EMOTIONAL_SUPPORT:
                            results = {
                                "loss": newloss,
                                "total_reward": total_reward,
                                AVG_TURN: conv_turn
                            }
                        elif self.game_config.name == PERSUATION:  
                            results = {
                                "loss": newloss,
                                "total_reward": total_reward,
                                AVG_TURN: conv_turn
                            }
                        # whether to use the stored states ??
                        # if the total reward is greater than zero
                        if total_reward > 0.0:
                            data_for_generation_finetuning.extend(all_states)
                            
                    # logging the results
                    # only using the file or terminal logger.
                    for logger in self.loggers:
                        if not isinstance(logger, WanDBLogger):
                            logger.record(results, train_step)

            # multi objective game
            if not self.game_config.is_so_game:
                # # evaluate the results on the dev set
                # # computationally
                # # for the other objectivest than the uniform, we save the best checkpoint on the dev set
                if self.model_config.prioritized_objective != "uniform":
                    if dev_cases is not None and train_step & self.model_config.eval_interval == 0:

                        # make sure the number of dev_cases is equal to the number of dev simulators
                        assert len(dev_cases) == len(dev_simulators)

                        # the rl model on the dev item set.
                        results = self.online_test(dev_cases, device, dev_simulators, action_mapping)

                        # logging the results
                        # also using wandb logger here.
                        for logger in self.loggers:
                            logger.record(results, train_step)

                        # saving the best checkpoiint
                        if results[self.model_config.prioritized_objective] > best_metric:
                            best_metric = results[self.model_config.prioritized_objective]
                            # saving the rl fine-tuning model
                            # save the rl pretrained model
                            loguru_logger.info("Saving the RL fine-tuned model .....")
                            if self.game_config.name == RECOMMENDATION:
                                # save the model checkpoint for the recommendation
                                file_path = os.path.join(self.model_config.saved_dir, f"rl_model_{self.model_config.prioritized_objective}_{self.model_config.domain}.pth")
                                
                            elif self.game_config.name == NEGOTIATION:
                                # save the model checkpoint for the negotiation
                                file_path = os.path.join(self.model_config.saved_dir, f"rl_model_{self.model_config.prioritized_objective}.pth") 
                            self.save_model(file_path)
            # single objective game
            else:
                if dev_cases is not None and train_step & self.model_config.eval_interval == 0:
                    # make sure the number of dev_cases is equal to the number of dev simulators
                    assert len(dev_cases) == len(dev_simulators)

                    # the rl model on the dev item set.
                    results = self.online_test(dev_cases, device, dev_simulators, action_mapping)
                    
                    metric_to_save = SL_RATIO                    
                    if self.game_config.name == EMOTIONAL_SUPPORT or self.game_config.name == RECOMMENDATION or self.game_config.name == PERSUATION:
                        metric_to_save = SUCCESS_RATE
                    
                    # saving the best checkpoiint
                    if results[metric_to_save] > best_metric:
                        best_metric = results[metric_to_save]
                        
                        # saving the rl fine-tuning model
                        # save the rl pretrained model
                        loguru_logger.info("Saving the RL fine-tuned model .....")
                        
                        # saving the model for recommendation
                        if self.game_config.name == RECOMMENDATION:
                            file_path = os.path.join(self.model_config.saved_dir, f"rl_model_{self.model_config.domain}.pth")
                        # saving the model for other scenarios.
                        else:
                            file_path = os.path.join(self.model_config.saved_dir, "rl_model.pth")
                
                        # save the model checkpoint           
                        self.save_model(file_path)

                    # logging the results
                    # also using wandb logger here.
                    for logger in self.loggers:
                        logger.record(results, train_step)
        # multi objective game
        if not self.game_config.is_so_game:
            # for the uniform model, we save the last checkpoint
            if self.model_config.prioritized_objective == "uniform":
                loguru_logger.info("Saving the RL fine-tuned model .....")
                if self.game_config.name == RECOMMENDATION:
                    file_path = os.path.join(self.model_config.saved_dir, f"rl_model_{self.model_config.prioritized_objective}_{self.model_config.domain}.pth")
                elif self.game_config.name == NEGOTIATION:
                    file_path = os.path.join(self.model_config.saved_dir, f"rl_model_{self.model_config.prioritized_objective}.pth")
                self.save_model(file_path)
        # for single objective game, we return the data for generation fine tuning
        else:
            return data_for_generation_finetuning

    def predict(self, instance, action_mapping=None, is_test=False):
        """
        method that predict the action given an input instance
        :param instance: the given input instance
        :param action_mapping: a dictionary that maps action to index
        :param is_test: True if it is inference time else False
        :return: an predicted action
        """
        # create the inverse action mapping
        if isinstance(action_mapping, tuple):
            # create the inverse action mapping
            inverse_action_mapping = {v: k for k, v in action_mapping[0].items()}
        else:    
            inverse_action_mapping = {v: k for k, v in action_mapping.items()}   
            
        # firstly we infering the user mental state 
        mental_states = self.model.infering_user_mental_states(instance['dialogue_context'], 
                                                               llm_pipeline = self.game_config.llm_pipeline,
                                                               terminators = self.game_config.terminators)
        instance['mental_states'] = mental_states

        # create the data loader
        data_loader = self.construct_dataloaders([instance],
                                                 batch_size=1,
                                                 goal2id=action_mapping,
                                                 shuffle=True, num_workers=self.model_config.num_workers)
        # predict the action
        for batch in data_loader:
            # predict the action
            # we also compute the reward using the next state
            logits = self.model(batch)
            action, log_prob = self.select_action(logits, is_test=is_test)
            action = inverse_action_mapping[action]

            if self.model_config.rewrite_action:
                
                if self.game_config.name == RECOMMENDATION:
                    action_description = DURECDIAL_GOAL2DESCRIPTION[action].format(instance['task_background']['target_topic'])
                
                if self.game_config.name == NEGOTIATION:
                    action_description = NEGOTIATION_GOAL2DESCRIPTION[action]
                    
                elif self.game_config.name == EMOTIONAL_SUPPORT:
                    action_description = ES_CONV_GOAL2DESCRIPTION[action]
                
                elif self.game_config.name == PERSUATION:
                    action_description = P4G_GOAL2DESCRIPTION[action]
                        
                rewritten_action = self.model.rewrite_action(
                        instance['dialogue_context'], 
                        action_description,
                        llm_pipeline=self.game_config.llm_pipeline,
                        terminators=self.game_config.terminators
                    )
                
                action = [action, rewritten_action]

            # return action and log prob
            return action, log_prob

    def select_action(self, logits, is_test=True):
        """
        method that select an action from the output logits
        :param logits: the logits output by a model
        :param is_test: True if it is inference time else false
        :return:action_id and log_prob
        """
        # convert logits to probabilities
        probs = nn.functional.softmax(logits, dim=1)
        m = Categorical(probs)

        # compute policy with offline policy model.
        if is_test:
            # greedy sampling
            action = logits.argmax()
            return action.item(), None
        else:
            # random sampling
            action = m.sample()
            log_prob = m.log_prob(action)
            print(action, log_prob)
            return action.item(), log_prob

    def compute_rl_policy_loss(self, saved_rewards, saved_log_probs):
        """
        method that compute the rl loss function, e.g a proxy of policy gradient
        :param saved_rewards: a list storing rewards observed during the current episode
        :param saved_log_probs: a list storing log probabilities of actions during the current episode
        :return: the rl loss function
        """
        R = 0
        policy_loss = []
        rewards = []

        # compute expected returns.
        for r in saved_rewards[::-1]:
            R = r + self.model_config.gamma * R
            rewards.insert(0, R)
            
        rewards = torch.cat(rewards, dim=0)
        
        # # Normalizing the expected returns.
        # if rewards.shape[0] > 1:
        #     rewards = (rewards - rewards.mean(dim=0)) / (rewards.std(dim=0) + self.model_config.epsilon)

        # compute - Q log p
        for log_prob, reward in zip(saved_log_probs, rewards):
            # first, we need to scalarize the expected returns
            # scalarizing the vector-valued return
            # assuming the weights of objectives are 1s
            # scalarized_return = reward.sum()
            # weights = [0.6, 0.3, 0.1]
            # computing scalarized returns
            # objective_weights = torch.Tensor(self.model_config.objective_weights)
            # uniformly sampled objective weights
            if not self.game_config.is_so_game:
                if self.model_config.uniform_weights:
                    objective_weights = random_weights(self.model_config.n_objectives)
                    objective_weights = torch.Tensor(objective_weights).to(self.device)
                # default objective weight = [1.0, 1.0]
                else:
                    objective_weights = torch.Tensor(self.model_config.objective_weight).to(self.device)

                assert len(objective_weights) == len(reward)
                scalarized_return = (objective_weights * reward).sum()
            # single objective game
            else:
                scalarized_return = reward
                
            # compute loss function which is a proxy of the policy gradient
            policy_loss.append(- log_prob * scalarized_return)

        policy_loss = torch.cat(policy_loss).sum()
        return policy_loss
    
    def test(self, dataset):
        """
        method that evaluate the performance of the model on the test set
        :param dataset: the dataset that we want to evaluate the model performance.
        :return: the results on the test setz
        """

        # create the action mapping
        action_mapping = dataset.construct_action_mapping(combine=self.model_config.combined_action)

        # create the data loader
        test_loader = self.construct_dataloaders(dataset.test_instances,
                                                 batch_size=self.model_config.per_device_eval_batch_size,
                                                 goal2id=action_mapping,
                                                 shuffle=False, num_workers=self.model_config.num_workers)

        # get the model performance on the test set.
        results = self.eval_epoch(test_loader, self.create_criterion())
        return results

    def online_test(self, cases, device=None, simulators=None, action_mapping=None):
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
        
        # read the meta prompt from file and set the meta prompt for the model config class
        if self.model_config.rewrite_action:
            self.model_config.read_meta_prompt()
            self.model.model_config = self.model_config
            
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
                action, log_prob = self.predict(state, 
                                                action_mapping, 
                                                is_test=True
                                                )
                
                log_probs.append(log_prob)
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

                # storing the log prob
                log_probs.append(log_prob)

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
                # persuation conversation
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
