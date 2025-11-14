import copy
import math
import os
import gc
import random
from itertools import count
import pickle as pkl

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.distributions import Categorical

from loguru import logger as loguru_logger
from transformers import get_linear_schedule_with_warmup

from baselines.DPDP.data_processor import DPDPDataProcessorForRecommendation, DPDPTorchDatasetForRecommendation, \
    DPDPTorchDatasetForNegotiation, DPDPDataProcessorForNegotiation, DPDPDataProcessorForEmotionalSupport, \
    DPDPTorchDatasetForEmotionalSupport

from base.trainer import Trainer
from logger.wandb_logger import WanDBLogger
from logger.file_logger import FileLogger


from config.constants import RECOMMENDATION, NEGOTIATION, EMOTIONAL_SUPPORT, SL_RATIO, SUCCESS_RATE, AVG_TURN, FAIRNESS, \
    TOXICITY, ITEM_FREQ, USER_REWARD

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


class DPDPTrainer(Trainer):

    def __init__(self, game_config, model_config, accelerator, game, model, offline_evaluator, online_evaluator,
                 loggers, generation_method=None):
        """
        constructor for class DPDP pipeline training
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

        # set the domain for the mcts config class.
        if self.game_config.name == RECOMMENDATION:
            self.model_config.mcts_config.set_params(
                {
                    'domain': self.model_config.domain
                }
            )
            print(self.model_config.mcts_config.domain)

    
    def preprocess_data_for_offline_rl(self, instances, goal2id, file_name):
        loguru_logger.info("Preprocessing the data for offline  reinforcment learning ....")
        loguru_logger.info(f"Numbr of instances: {len(instances)} ....")
        processed_instances = []
        for instance in tqdm(instances):
            
            if self.game_config.name == NEGOTIATION:
                data_processor = DPDPDataProcessorForNegotiation
                
            if self.game_config.name == RECOMMENDATION:
                data_processor = DPDPDataProcessorForRecommendation
                
            state, action, reward, next_state, done = data_processor()(tokenizer = self.tokenizer,
                                                                        instance = instance,                 
                                                                        action_to_id = goal2id,
                                                                        max_sequence_length=self.model_config.max_sequence_length,
                                                                        )

            # print(action, reward, done)
            processed_instance = copy.deepcopy(instance)
            processed_instance['processed_data'] =  [state, action, reward, next_state, done]
            processed_instances.append(processed_instance)
        
        loguru_logger.info("Saving the data for offline reinforcment learning ....")
        # save the processed instances to file
        # if not os.path.exists(os.path.join(self.model_config.saved_dir, file_name)):

        with open(os.path.join(self.model_config.saved_dir, file_name), "wb") as f:
            pkl.dump(processed_instances, f)       

    def process_tvqs(self, instances):
        processed_instances = []
        conv_ids = [x['conv_id'] for x in instances]
        max_ids = max(conv_ids)
                        
        for idx in range(max_ids):
            new_instances = [x for x in instances if x['conv_id'] == idx]
            rewards = [list(x['processed_data'][2]) for x in instances if x['conv_id'] == idx]
            target_qvs = []
            
            # no instances
            if len(rewards) == 0:
                continue
                
            for idx in range(len(rewards) - 2, -1, -1):
                for t in range(self.model_config.n_objectives):
                    rewards[idx][t] += self.model_config.gamma * rewards[idx + 1][t]
                
                max_r = max(rewards[idx]) + self.model_config.epsilon
                for t in range(len(rewards[idx])):
                    rewards[idx][t] = rewards[idx][t] / max_r   
                rewards[idx] = tuple(rewards[idx])

            for instance, reward in list(zip(new_instances, rewards)):
                instance['processed_data'][2] = reward
                processed_instances.append(instance)
                
        return processed_instances

            
    def load_processed_data_for_offline_rl(self, file_name):
        with open(os.path.join(self.model_config.saved_dir, file_name), "rb") as f:
            processed_instances = pkl.load(f)
        return processed_instances

    def process_dataset(self, dataset, action_mapping = None):
        """
        method that process the given dataset and return processed data instances
        :return: processed data instances.
        """
        
        # preprocessing data for offline reinforcment learning
        if self.model_config.preprocess_data_for_offline_rl:
            
            # if we have not preprocessed the data
            # negotiation scenario
            if self.game_config.name == NEGOTIATION:
                train_file_name = "train_processed_instances.pkl"
                dev_file_name = "dev_processed_instances.pkl"
                test_file_name = "test_processed_instances.pkl"
            # recommendation scenario
            # file_path + domain
            elif self.game_config.name == RECOMMENDATION:
                train_file_name = f"train_processed_instances_{self.model_config.domain}.pkl"
                dev_file_name = f"dev_processed_instances_{self.model_config.domain}.pkl"
                test_file_name = f"test_processed_instances_{self.model_config.domain}.pkl"

            if not os.path.exists(os.path.join(self.model_config.saved_dir, train_file_name)):
                # processing the data
                self.preprocess_data_for_offline_rl(dataset.train_instances, action_mapping, file_name=train_file_name)
                self.preprocess_data_for_offline_rl(dataset.dev_instances, action_mapping, file_name=dev_file_name)
                self.preprocess_data_for_offline_rl(dataset.test_instances, action_mapping, file_name=test_file_name)
                # then load the data
                dataset.train_instances = self.load_processed_data_for_offline_rl(file_name = train_file_name)
                dataset.dev_instances = self.load_processed_data_for_offline_rl(file_name = dev_file_name)
                dataset.test_instances = self.load_processed_data_for_offline_rl(file_name = test_file_name)   
            else:
                # only load the data
                loguru_logger.info("The data has been preprocessed for offline reinforcment learning. Loading the data ....")
                dataset.train_instances = self.load_processed_data_for_offline_rl(file_name = train_file_name)
                dataset.dev_instances = self.load_processed_data_for_offline_rl(file_name = dev_file_name)
                dataset.test_instances = self.load_processed_data_for_offline_rl(file_name = test_file_name)   

        # load the preprocessed data for offline RL
        elif self.model_config.load_processed_data_for_offline_rl:
            loguru_logger.info("Loading the preprocessed data for offline reinforcment learning.")
            dataset.train_instances = self.load_processed_data_for_offline_rl(file_name = train_file_name)
            dataset.dev_instances = self.load_processed_data_for_offline_rl(file_name = dev_file_name)
            dataset.test_instances = self.load_processed_data_for_offline_rl(file_name = test_file_name)   
            
        # processing the tvqs for offline RL
        # computing the discounted returns from the preprocess trajectories
        dataset.train_instances = self.process_tvqs(dataset.train_instances)
        dataset.dev_instances = self.process_tvqs(dataset.dev_instances)
        dataset.test_instances = self.process_tvqs(dataset.test_instances) 
                    
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
            torch_dataset = DPDPTorchDatasetForRecommendation(
                tokenizer=self.tokenizer,
                instances=data_instances,
                goal2id=goal2id,
                max_sequence_length=self.model_config.max_sequence_length,
                device=self.device,
                convert_example_to_feature=DPDPDataProcessorForRecommendation()
            )
        # negotiation scenario
        elif self.game_config.name == NEGOTIATION:
            torch_dataset = DPDPTorchDatasetForNegotiation(
                tokenizer=self.tokenizer,
                instances=data_instances,
                goal2id=goal2id,
                max_sequence_length=self.model_config.max_sequence_length,
                device=self.device,
                convert_example_to_feature=DPDPDataProcessorForNegotiation()
            )
        # emotional support conversation
        elif self.game_config.name == EMOTIONAL_SUPPORT:
            torch_dataset = DPDPTorchDatasetForEmotionalSupport(
                tokenizer=self.tokenizer,
                instances=data_instances,
                goal2id=goal2id,
                max_sequence_length=self.model_config.max_sequence_length,
                device=self.device,
                convert_example_to_feature=DPDPDataProcessorForEmotionalSupport()
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
            loss, actor_loss, critic_loss = self.model(batch)
            # loss = criterion(logits, batch['labels']) / self.model_config.gradient_accumulation_steps
            
            self.accelerator.backward(loss)
            train_loss.append(float(loss))

            self.progress_bar.update(1)
            self.global_step += 1

            loguru_logger.info(f"Pretraining Step {step},  Actor Loss: {actor_loss}, Critic Loss: {critic_loss}")

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
                    loss, _, _ = self.model(batch)
                    # loss = criterion(logits, batch['labels'])
                    # self.offline_evaluator.record(logits, batch['labels'])
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

        # construct the goal, topic mapping
        action_mapping = dataset.construct_action_mapping(combine=self.model_config.combined_action)


        # process dataset
        train_instances, dev_instances, test_instances = self.process_dataset(dataset, action_mapping)

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
                file_path = os.path.join(self.model_config.saved_dir, "model.pth")
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
        optimizer = self.create_optimizer(self.model, self.model_config.rl_learning_rate)

        # using accelerator to prepare the model and optimizer
        # self.model, optimizer = self.accelerator.prepare(self.model, optimizer)
        self.model.to(self.device)

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
                qa_values = []

                # flag for checking if the target is mentioned during the conversation
                o_flag = False

                if self.model_config.prioritized_objective == "uniform":
                    w = random_weights(self.model_config.n_objectives, dist = "uniform")
                
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

                # interactive simulation
                for t in count():  # user  dialog

                    # predict the action
                    action, log_prob, qa_value = self.predict(state, simulator, action_mapping)
                    log_probs.append(log_prob)
                    qa_values.append(qa_value)

                    # employing the action to observe the next state
                    # and the corresponding rewards
                    state, reward, done, o_done = self.game.step(state, action, self.generation_method, simulator)

                    # check if target is mentioned during the conversation
                    if o_done == 1:
                        o_flag = True

                    # storing the reward
                    reward = torch.tensor([reward], device=device, dtype=torch.float)
                    rewards.append(reward)
                    epi_reward.append(reward)

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
                        # total_reward += epi_reward
                        break

                # optimizing the policy model
                optimizer.zero_grad()
                
                # compute the loss function, e.g a proxy of the policy gradient
                newloss = self.compute_rl_policy_loss(rewards, log_probs, qa_values)
                newloss.backward()
                optimizer.step()

                # log the results
                # this is the training results
                if newloss is not None:
                    loss += newloss

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

                    # logging the results
                    # only using the file or terminal logger.
                    for logger in self.loggers:
                        if not isinstance(logger, WanDBLogger):
                            logger.record(results, train_step)

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

        # for the uniform model, we save the last checkpoint
        if self.model_config.prioritized_objective == "uniform":
            loguru_logger.info("Saving the RL fine-tuned model .....")
            if self.game_config.name == RECOMMENDATION:
                file_path = os.path.join(self.model_config.saved_dir, f"rl_model_{self.model_config.prioritized_objective}_{self.model_config.domain}.pth")
            elif self.game_config.name == NEGOTIATION:
                file_path = os.path.join(self.model_config.saved_dir, f"rl_model_{self.model_config.prioritized_objective}.pth")
            self.save_model(file_path)

    def predict(self, instance, simulator, action_mapping=None, is_test=False):
        """
        method that predict the action given an input instance
        :param instance: the given input instance
        :param action_mapping: a dictionary that maps action to index
        :param is_test: True if it is inference time else False
        :return: an predicted action
        """
        copied_action_mapping = copy.deepcopy(action_mapping)
        
        # create the inverse action mapping
        if isinstance(action_mapping, tuple):
            # create the inverse action mapping
            copied_action_mapping = copied_action_mapping[0]
        
        inverse_action_mapping = {v: k for k, v in copied_action_mapping.items()}     
        # predicting the action using policy planner.
        
        # create the data loader
        data_loader = self.construct_dataloaders([instance],
                                                 batch_size=1,
                                                 goal2id=action_mapping,
                                                 shuffle=True, 
                                                 num_workers=self.model_config.num_workers
                                                 )
        
        # predict the action acording to policy
        for batch in data_loader:
            
            # predict the action
            # we also compute the reward using the next state
            logits, qa_values = self.model(batch, is_test=True)
            
            # compute the log prob
            probs = nn.functional.softmax(logits, dim=1)
            m = Categorical(probs)
            
            # action, log_prob = self.select_action(logits, is_test=is_test)
            # action = inverse_action_mapping[action]
            # return action and log prob
        
        # game simulator
        game_simulator = copy.deepcopy(simulator)
        
        # no prior knowledge about the user profile
        # avoding information leakage.
        game_simulator.is_using_persona(False)
        
        # set the mcts generation model type
        # set the objective weight
        self.model.mcts.model_config.set_params(
            {
                'model_type': self.model_config.model_type,
                'objective_weight': self.model_config.objective_weight
            }
        )
        
        # inference time
        # action selecting according to https://github.com/cs-holder/DPDP/blob/main/ESConv%26CB/ppdpp/agent.py
        if is_test:
            topk_probs, _ = torch.topk(nn.functional.softmax(logits, dim=1), k=2)
            # if topk_probs[0][0] - topk_probs[0][1] > self.model_config.sub_value:
            # for this work, we do not consider the mcts planner part since we only consider the online dialogue simulations
            if True:
                action = logits.argmax().item()
                action = inverse_action_mapping[action]                          
            else:
                action = self.model.mcts(
                                    instance,
                                    game=self.game,
                                    generation_method=self.generation_method,
                                    user_simulator=game_simulator,
                                    action_mapping=copied_action_mapping
                                    ) 
                
            return action, None, None
        
        # training time
        else:
            # self-play MCTS training.
            # predicting the action using mcts planner
            mcts_action = self.model.mcts(
                                instance,
                                game=self.game,
                                generation_method=self.generation_method,
                                user_simulator=game_simulator,
                                action_mapping=copied_action_mapping
                                )
            
            # mcts_action_id
            mcts_action_id = torch.LongTensor([copied_action_mapping[mcts_action]]).to(self.device)
            
            # compute the log prob of action predicted by mcts.
            log_prob = m.log_prob(mcts_action_id)
            
            # qa values
            qa_values = qa_values.gather(1, mcts_action_id.unsqueeze(dim=-1)).squeeze(dim=-1)

            loguru_logger.info(f"[MCTS Action]: {mcts_action}")
                        
            # return mcts_action, log_prob
            return mcts_action, log_prob, qa_values        

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
            return action.item(), log_prob

    def compute_rl_policy_loss(self, saved_rewards, saved_log_probs, saved_qa_values):
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
        for log_prob, reward, qa_value in zip(saved_log_probs, rewards, saved_qa_values):
            # first, we need to scalarize the expected returns
            # scalarizing the vector-valued return
            # assuming the weights of objectives are 1s
            # scalarized_return = reward.sum()
            # weights = [0.6, 0.3, 0.1]
            # computing scalarized returns
            # objective_weights = torch.Tensor(self.model_config.objective_weights)
            # uniformly sampled objective weights
            if self.model_config.uniform_weights:
                objective_weights = random_weights(self.model_config.n_objectives)
                objective_weights = torch.Tensor(objective_weights).to(self.device)
            # default objective weight = [1.0, 1.0]
            else:
                objective_weights = torch.Tensor(self.model_config.objective_weight).to(self.device)

            assert len(objective_weights) == len(reward)

            # scalarizing
            scalarized_return = (objective_weights * reward).sum()
            
            # critic loss
            mse_loss = torch.nn.MSELoss(reduction='mean')
            critic_loss = mse_loss(qa_value, scalarized_return)
            
            # advantages
            td_delta = scalarized_return - qa_value
            
            # actor loss
            actor_loss = - log_prob * td_delta
            
            loss = actor_loss + self.model_config.critic_loss_w * critic_loss
            
            # compute loss function which is a proxy of the policy gradient
            policy_loss.append(loss)

        policy_loss = torch.cat(policy_loss).mean()
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

        # loss = torch.tensor(0, dtype=torch.float, device=device)

        # randomly sample persona information
        # simulator = np.random.choice(simulators)
        # select a particular simulator
        # and promote items to this simulator
        # simulator = simulators[0]
        convs = []

        # loop over the item set
        # make sure each item is associated with one user profile
        for idx, (case, simulator) in tqdm(enumerate(list(zip(cases[:20], simulators)))):

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

            # interactive simulation
            for t in count():  # user  dialog

                # predict the action
                action, log_prob, _ = self.predict(state, simulator, action_mapping, is_test=True)
                
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
            if self.game_config.name == NEGOTIATION:

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
        # compute the results using the evaluator
        results = self.online_evaluator.report()

        # log the results to terminal or file
        for logger in self.loggers:
            if not isinstance(logger, WanDBLogger):
                logger.record(results, "Testing")
            
            # save conversations for human evaluation
            if isinstance(logger, FileLogger):
                for idx, conv in enumerate(convs):
                    save_conv_path = os.path.join(logger.log_dir, f"conversation_{idx}.txt")
                    save_conversation_for_human_evaluation(save_conv_path, conv)    

        # return the results of the online evaluation
        return results
