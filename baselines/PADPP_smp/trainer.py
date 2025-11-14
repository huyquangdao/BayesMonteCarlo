import copy
import math
import os
import random
from itertools import count

from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.distributions import Categorical

from loguru import logger as loguru_logger
from transformers import get_linear_schedule_with_warmup

from baselines.PADPP_smp.data_processor import SetMaxPADPPDataProcessorForRecommendation, SetMaxPADPPTorchDataset, \
    SetMaxPADPPDataProcessorForNegotiation, SetMaxPADPPDataProcessorForEmotionalSupport

from base.trainer import Trainer
from logger.wandb_logger import WanDBLogger
from logger.terminal_logger import TerminalLogger

from utils.game import random_weights
from collections import deque

from config.constants import RECOMMENDATION, NEGOTIATION, EMOTIONAL_SUPPORT, SL_RATIO, SUCCESS_RATE, AVG_TURN, FAIRNESS, \
    TOXICITY, ITEM_FREQ, USER_REWARD


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def calculate_gae(rewards, dones, gamma):
    """
    compute
    :param values:
    :param rewards:
    :param dones:
    :param next_values:
    :param gamma:
    :param lambd:
    :return:
    """
    # discounted returns
    discounted_returns = torch.empty_like(rewards).to(rewards.device)
    R = 0
    for t in reversed(range(rewards.size(0))):
        # gaes[t] = deltas[t] + gamma * lambd * (1 - dones[t]) * gaes[t + 1]
        # computing the TD target
        if dones[t].item() == 1:
            R = 0
        R = rewards[t] + R * gamma
        discounted_returns[t] = R
    # # normalizing the gae
    # wrong
    # discounted_returns = (discounted_returns - discounted_returns.mean(dim=0).unsqueeze(0)) / (
    #         epsilon + discounted_returns.std(dim=0).unsqueeze(0))
    return discounted_returns


class SetMaxPADPPTrainer(Trainer):

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

        # global steps for training the actor-critic model and the preference model
        self.ppo_global_step = 1
        self.preference_global_step = 0

        # the corresponding progress bars
        self.ppo_progress_bar = None
        self.preference_progress_bar = None

        self.model = accelerator.unwrap_model(self.model)
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

        # the data processor for the  recommendation scenario
        if self.game_config.name == RECOMMENDATION:
            convert_example_to_feature = SetMaxPADPPDataProcessorForRecommendation
        # the data processor for the negotiation scenario
        elif self.game_config.name == NEGOTIATION:
            convert_example_to_feature = SetMaxPADPPDataProcessorForNegotiation
        # the data processor for the emotional support conversation
        elif self.game_config.name == EMOTIONAL_SUPPORT:
            convert_example_to_feature = SetMaxPADPPDataProcessorForEmotionalSupport
        else:
            raise Exception("Invalid scenario....")
        
        # construct the torch dataset
        torch_dataset = SetMaxPADPPTorchDataset(
            tokenizer=self.tokenizer,
            instances=data_instances,
            goal2id=goal2id,
            max_sequence_length=self.model_config.max_sequence_length,
            device=self.device,
            n_objectives=self.model_config.n_objectives,
            convert_example_to_feature=convert_example_to_feature()
        )
        
        # construct the data loader
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
        print("Number trainable params: ", count_parameters(model))
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
            logits, _ = self.model(batch)
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
                    logits, _ = self.model(batch)
                    loss = criterion(logits, batch['labels'])
                    self.offline_evaluator.record(logits, batch['labels'])
                    dev_loss.append(float(loss))

        dev_loss = np.mean(dev_loss) * self.model_config.gradient_accumulation_steps
        self.accelerator.wait_for_everyone()
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

        # construct the goal, topic mapping
        action_mapping = dataset.construct_action_mapping(combine=self.model_config.combined_action)

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

            # train the model
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
          
                # pretrained sft for recommendation
                if self.game_config.name == RECOMMENDATION:
                    file_path = os.path.join(self.model_config.saved_dir, f"model_{self.model_config.domain}.pth")
                # pretrained sft for negotiation
                elif self.game_config.name == NEGOTIATION:
                    file_path = os.path.join(self.model_config.saved_dir, f"model.pth")                   
                self.save_model(file_path)

            if stop:
                loguru_logger.info("Training process is completed.")
                break

    def train_preference(self, preference_instances, action_mapping, preference_optimizer, preference_scheduler,
                         device=None):
        """
        method that perform preference training step.
        :param preference_instances: the buffer containing data instances for training the preference model
        :param action_mapping: a dictionary that map goal, topic to ids
        :param preference_optimizer: an optimizer that is used to train the preference model
        :param preference_scheduler: the lr scheduler for the preference optimizer
        :param device: None
        :return:
        """
        # freezing parameters of the policy part
        # setting requires_grad to false
        self.model.manipulate_gradient_update(is_preference_block=False, flag=False)
        self.model.manipulate_gradient_update(is_preference_block=True, flag=True)

        # create train, dev and test dataloaders
        best_loss = math.inf

        # progress bar
        progress_bar = tqdm(range(self.model_config.num_train_preference_epochs),
                            disable=not self.accelerator.is_local_main_process)

        # train the model
        self.model.train()
        mean_preference_loss = []

        # loop for a number of training epochs:
        # e.g 10:
        prev_step = 0
        for epoch in range(self.model_config.num_train_preference_epochs):

            # randomly sample a batch of instances
            # batch_instances = random.sample(preference_instances, self.model_config.preference_batch_size)

            next_step = prev_step + self.model_config.preference_batch_size
            if next_step > len(preference_instances):
                prev_step = 0
                continue

            batch_instances = preference_instances[prev_step: next_step]
            prev_step = next_step

            # batch accumulated estimated reward
            batch_accumulated_estimated_reward = []
            batch_preference_weights = []
            batch_accumulated_returns = []

            # for each instance in the batch instances
            # for each trajectory in the current batch
            for instance in batch_instances:

                # get preference weight
                batch_preference_weights.append(torch.FloatTensor(instance[1]).to(self.device))

                # get the accumulate returns
                # monte-carlo return
                batch_accumulated_returns.append(instance[2])

                # get the accumulated reward score
                accumualated_estimated_reward = 0.0

                # loop over all state in the trajectory
                # computing the td-error loss
                for i, state in enumerate(instance[0]):

                    # create data loader to compute the estimated reward function
                    train_loader = self.construct_dataloaders([state],
                                                              batch_size=1,
                                                              shuffle=True,
                                                              goal2id=action_mapping,
                                                              num_workers=self.model_config.num_workers)

                    # computing the feature representation
                    # computing Phi(s_{t+1})
                    for batch in train_loader:
                        # computing the feature representation
                        estimated_reward = self.model.compute_features(batch)
                        # computing the accumulated estimated reward
                        # e.g phi(s') = \phi(s) + gamma ** i * estimated_reward
                        # an approximate version of V_{c}^{\pi}
                        accumualated_estimated_reward = accumualated_estimated_reward + (
                            self.model_config.gamma) ** i * estimated_reward

                batch_accumulated_estimated_reward.append(accumualated_estimated_reward)

            # optimizing the MSE loss between accumulated estimated reward and MC-sampled scalarized rewards
            batch_accumulated_returns = torch.Tensor(batch_accumulated_returns).to(self.device)

            # stacking the accumulated estimated reward tensor
            batch_accumulated_estimated_reward = torch.cat(batch_accumulated_estimated_reward, dim=0).squeeze(-1)
            preference_optimizer.zero_grad()

            # mask of reward
            mask_of_reward = (batch_accumulated_returns.sum(dim=-1) > 0).unsqueeze(-1)

            # compute the MSE loss
            # utilizing Monte-Carlo to estimate feature representation
            mc_loss = torch.nn.MSELoss(reduction='none')(batch_accumulated_estimated_reward, batch_accumulated_returns)
            loss1 = (mc_loss * mask_of_reward).mean()

            # computing the scalarized mc loss
            batch_preference_weights = torch.stack(batch_preference_weights)
            loss2 = torch.abs(
                (((batch_accumulated_estimated_reward - batch_accumulated_returns) * batch_preference_weights).sum(
                    dim=-1) * mask_of_reward)
            ).mean()

            # backward the loss function with accelerator
            loss = 0.9 * loss1 + 0.1 * loss2
            loss.backward()
            mean_preference_loss.append(loss.item())

            # update the progress bar
            progress_bar.update(1)
            results = {
                "loss": loss
            }

            # update the gradient and the lr scheduler
            preference_optimizer.step()
            preference_scheduler.step()

            # logging the results
            for logger in self.loggers:
                if not isinstance(logger, WanDBLogger):
                    logger.record(results, epoch + 1)

        # compute the mean preference loss for the current rl step
        mean_preference_loss = sum(mean_preference_loss) / len(mean_preference_loss)
        results = {
            "preference_loss": mean_preference_loss

        }

        # logging the results
        for logger in self.loggers:
            if isinstance(logger, WanDBLogger):
                logger.record(results, self.preference_global_step)

        self.preference_global_step += 1

    def train_ppo(self, ppo_buffer, action_mapping, actor_optimizer, actor_scheduler, critic_optimizer,
                  critic_scheduler):
        """
        method that perform the actor critic training part
        :param ppo_buffer: the buffer that stores data instances for actor-critic training
        :param action_mapping: a dictionary that maps goal, topic to ids
        :param actor_optimizer: the optimizer for training the actor model
        :param actor_scheduler: the scheduler for training the actor model
        :param critic_optimizer: the optimizer for training the critic model
        :param critic_scheduler: the scheduler for training the critic model
        :return:
        """
        # # freezing the preference block
        # # only update the policy part
        # self.model.manipulate_gradient_update(is_preference_block=True, flag=True)
        # self.model.manipulate_gradient_update(is_preference_block=False, flag=True)

        # # progress bar
        progress_bar = tqdm(range(self.model_config.num_train_ppo_epochs),
                            disable=not self.accelerator.is_local_main_process)

        # the target network
        self.target_model = copy.deepcopy(self.model)

        # loop over the number of ppo epochs
        mean_actor_loss = []
        mean_crtic_loss = []

        # define a variable capturing ids of previous batch data.
        for i in tqdm(range(1)):

            # otherwise we sample a batch of data from prev_step to next_step
            # to train the model
            batch_instances = ppo_buffer

            # states
            states = [x[0] for x in batch_instances]

            # action
            batch_act = [action_mapping[state['act']] for state in states]
            batch_act = torch.LongTensor(batch_act).to(self.device)

            # rewards
            batch_rewards = [x[2] for x in batch_instances]
            batch_rewards = torch.cat(batch_rewards, dim=0).to(self.device)

            values = [x[-3] for x in batch_instances]
            values = torch.cat(values, dim = 0)

            # batch log probabilities
            log_probs = [x[-2] for x in batch_instances]
            batch_log_probs = torch.cat(log_probs, dim=0)

            # get the done flags
            batch_done = [x[-1] for x in batch_instances]
            batch_done = torch.Tensor(batch_done).to(self.device)

            train_loader = self.construct_dataloaders(states,
                                                      batch_size=len(ppo_buffer),
                                                      shuffle=False,
                                                      goal2id=action_mapping,
                                                      num_workers=self.model_config.num_workers
                                                      )

            # loop over the train loader
            for batch in train_loader:
                # sample a batch of preference weights
                # sampled_preferences = random_weights(self.model_config.n_objectives, n=self.model_config.n_preferences)
                # # sampled_preferences = list(self.model_config.obj_to_weight.values())
                # # sampled_preferences = [x for x in sampled_preferences if x is not None]
                # # sampled_preferences = np.array(sampled_preferences)
                # sampled_preferences = torch.Tensor(sampled_preferences).to(self.device).requires_grad_(False)

                # bs, n_objectives
                sampled_preferences = batch['w']
                self.memory_buffer.extend(sampled_preferences)

                # computing feature representations
                # s, s', w
                state, _, w_embedding = self.model.compute_state_resp(batch, sampled_preferences)
                bs = state.size(0)

                # construct the feature vector
                feature = torch.cat([
                    state,
                    w_embedding
                ], dim=-1)

                # computing the logit using the actor network
                # Q(s,a,w)
                Q = self.model.compute_state_value(feature)
                Q = Q.view(Q.size(0), -1, self.model_config.n_objectives)

                action_size = Q.size(1)
                # shape = [bs, n_objectives]
                Q1 = Q.gather(1, batch_act.view(-1, 1, 1).expand(Q.size(0), 1, self.model_config.n_objectives)).view(-1,
                                                                                                                  self.model_config.n_objectives)
                
                # computing the discounted returns from the whole trajectory
                # optimizing the value network by optimizing the mse between value predictions and disocunted returns
                discounted_returns = calculate_gae(rewards=batch_rewards, 
                                                   dones=batch_done.unsqueeze(-1), 
                                                   gamma=self.model_config.gamma,
                                                   )
                             
                # # extend state, next state, w_embedding
                # # computing the convex envelope alignment
                with torch.no_grad():
                    if len(self.memory_buffer) < self.model_config.n_preferences:
                        mem_preferences = sampled_preferences
                    # sampled previous learned preferences from the memory
                    else:
                        mem_preferences = random.choices(self.memory_buffer, k=self.model_config.n_preferences)
                        mem_preferences = torch.stack(mem_preferences).to(self.device)

                    # sampled previous update preferences from a memory buffer
                    n_mem_preferences = len(mem_preferences)

                    # compute the objective embedidng for memory-basd preferences
                    _, _, w_embedding = self.model.compute_state_resp(batch, mem_preferences)
                
                    # expanding the sampled preferences
                    w_embedding = w_embedding.repeat(bs, 1).view(-1, w_embedding.size(-1))
                
                    # expanding the states
                    # s,w for all s,w' in W
                    state = state.repeat(1, n_mem_preferences).view(-1, state.size(-1))

                    expanded_features = torch.cat([
                        state,
                        w_embedding
                    ], dim=-1)

                    # compute target policy
                    target_policy = self.target_model.compute_policy(expanded_features)

                    # bs * n_preferences, n_objectives
                    # stop_gradient
                    # computing the state values for memoried preferences
                    # Q(s,a,w)
                    expanded_values = self.target_model.compute_state_value(expanded_features).detach()
                    expanded_values = expanded_values.view(expanded_values.size(0), -1, self.model_config.n_objectives)
                    

                    # tmp expanded features
                    tmp_expanded_values = self.model.compute_state_value(expanded_features).detach()
                    tmp_expanded_values = tmp_expanded_values.view(tmp_expanded_values.size(0), -1, self.model_config.n_objectives)


                    # bs * n_preferences, n_objectives
                    expanded_sampled_preferences = sampled_preferences.repeat(1, n_mem_preferences).view(-1,
                                                                                                         self.model_config.n_objectives)
                
                    
                    # V(s,w) = Q(s,\pi_{w}(s),w)
                    # shape = [bs * n_mem_preferences, n_objectives]
                    expanded_vs = expanded_values.gather(1, target_policy.view(-1, 1, 1).expand(expanded_values.size(0), 1, self.model_config.n_objectives)).view(-1,
                                                                                                                  self.model_config.n_objectives)
                    
                    expanded_tmp_vs = tmp_expanded_values.gather(1, target_policy.view(-1, 1, 1).expand(tmp_expanded_values.size(0), 1, self.model_config.n_objectives)).view(-1,
                                                                                                                  self.model_config.n_objectives)
                    
                    
                    # shape = [bs, n_mem_preferences]
                    scalarized_expanded_vs = torch.bmm(expanded_sampled_preferences.unsqueeze(1),
                                                           expanded_vs.unsqueeze(2)).view(bs, n_mem_preferences,
                                                                                              -1).squeeze(-1)
                    
                    # shape = [bs, n_mem_preferences]      
                    co_sim = torch.nn.functional.cosine_similarity(expanded_sampled_preferences, expanded_tmp_vs).view(bs, n_mem_preferences, -1).squeeze(-1)
                    
                    # idx = [bs]            
                    idx = (co_sim * scalarized_expanded_vs).max(1)[1]
                
                    #  stop gradient
                    gpi_values = expanded_vs.view(bs, n_mem_preferences, -1).gather(1,
                                                                                              idx.view(-1, 1, 1).expand(
                                                                                                  bs,
                                                                                                  1,
                                                                                                  self.model_config.n_objectives)).squeeze()
                    

                # scalarized state values using current sampled preferences
                w_values = torch.bmm(sampled_preferences.unsqueeze(1), 
                                     Q1.unsqueeze(2)).squeeze()

                # scalarized discounted returns
                w_gpi_values = torch.bmm(sampled_preferences.unsqueeze(1),
                                                gpi_values.unsqueeze(2)).squeeze()
                
                # scalarized discounted returns
                w_discounted_returns = torch.bmm(sampled_preferences.unsqueeze(1),
                                      discounted_returns.unsqueeze(2)).squeeze()
                
                                
                # computing the loss for the critic
                value_loss = (1 - self.model_config.alpha) * torch.nn.functional.mse_loss(w_values, w_discounted_returns, reduction = 'mean')

                # computing the gpi td error loss
                value_loss += self.model_config.alpha * torch.nn.functional.mse_loss(w_values, w_gpi_values, reduction='mean')

                # critic loss function is equal to the convential critic loss plus the convex envelope loss
                critic_loss = value_loss
                
                # advantages
                adv = w_discounted_returns.detach()

                # TODO: including the KL regularized term in the actor loss
                actor_loss = (- adv.detach() * batch_log_probs).mean()

                # update the parameters of actor and critic
                actor_optimizer.zero_grad()
                actor_loss.backward()

                actor_optimizer.step()
                # update the learning rate
                
                actor_scheduler.step()
                # update the target network
                self.target_model.load_state_dict(self.model.state_dict())

                # collect the mean actor and critic loss
                mean_actor_loss.append(actor_loss)
                mean_crtic_loss.append(critic_loss)
                progress_bar.update(1)

        # compute the mean actor and critic loss
        mean_actor_loss = sum(mean_actor_loss) / len(mean_actor_loss)
        mean_crtic_loss = sum(mean_crtic_loss) / len(mean_crtic_loss)

        results = {
            "actor_loss": mean_actor_loss,
            "critic_loss": mean_crtic_loss
        }

        # log the results to terminal or file
        for logger in self.loggers:
            logger.record(results, self.ppo_global_step)

        self.ppo_global_step += 1

    def train_rlt(self, cases, dev_cases=None, device=None, simulators=None, dev_simulators=None, action_mapping=None):
        """
        method that train the model in a reinforcement learning manner
        :param cases: a list of situations, target_items, e.g....
        :param device: the device that we use to train the model
        :param simulators: a set of simulators to train the rl agent
        :param action_mapping: a dictionary that map (goa, topic) to index
        :return: None
        """
        # self.model.manipulate_gradient_update(is_preference_block=True, flag=True)
        # self.model.manipulate_gradient_update(is_preference_block=False, flag=True)

        self.model = self.accelerator.unwrap_model(self.model)

        # compute the total actor and critic training step
        max_actor_critic_training_steps = self.model_config.num_train_rl_epochs * int(
            self.model_config.ppo_buffer_length // self.model_config.train_rl_batch_size)
        
        print("Actor Learning Rate: ", self.model_config.actor_learning_rate)
        
        # create the optimizer for actor and critic
        actor_optimizer = self.create_optimizer(self.model,
                                                self.model_config.actor_learning_rate)

        critic_optimizer = self.create_optimizer(self.model,
                                                 self.model_config.critic_learning_rate)

        # create the actor and critic lr schedulers
        actor_scheduler = self.create_scheduler(actor_optimizer,
                                                num_warmup_steps=self.model_config.actor_warmup_steps,
                                                max_train_steps=max_actor_critic_training_steps)

        critic_scheduler = self.create_scheduler(critic_optimizer,
                                                 num_warmup_steps=self.model_config.critic_warmup_steps,
                                                 max_train_steps=max_actor_critic_training_steps)

        # best metric
        best_metric = - math.inf
        # using accelerator to prepare the model and optimizer
        # self.model = self.accelerator.prepare(self.model)
        self.model.to(self.device)

        self.memory_buffer = deque(maxlen=self.model_config.preference_buffer_length)

        # loop for the number of epoch
        # number of training episode / n_episode each epoch
        for train_step in range(0, self.model_config.num_train_rl_epochs + 1):

            # collecting experiences
            # create a buffer to store trajectories results
            ppo_buffer = []
            self.model.train()

            # using the current policy model
            # this is the execution phase in the algorithm.
            for i_episode in tqdm(range(self.model_config.sampled_times), desc='sampling'):

                # randomly sample one case
                # sample 1 item
                case = np.random.choice(cases)
                # sample a preference vector using the trained preference params
                # this step is not differentiable
                w = random_weights(self.model_config.n_objectives)

                # objs = list(self.model_config.obj_to_weight.keys())
                # objs.remove("uniform")
                # choice = np.random.randint(low=0, high=len(objs))
                # w = self.model_config.obj_to_weight[objs[choice]]

                # randomly sample persona information
                # i.e a random user is sampled
                simulator = np.random.choice(simulators)
                loguru_logger.info('\n================New Episode:{}===================='.format(i_episode))

                # reset the game state
                # sampled initial state s(0)
                # construct a new game state based on the given case and the current simulator
                state = self.game.reset(case, simulator)

                # assign the preference weight vector
                state['w'] = w

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

                # more than 1 objectives, therefore the reward is a vector
                done = False

                # trajectories to store simulated interactions
                trajectory = []
                rewards = []

                loguru_logger.info(f"Objective Weight: [{w}]")

                # interactive simulations
                # executing step: c
                for t in count():  # user  dialog
                    # trajectory-level buffer

                    # old state
                    old_state = copy.deepcopy(state)

                    # predict the action using the sampled w and the trained model
                    # a ~ \pi(a|s,w)
                    action, log_prob, value, _ = self.predict(state, torch.FloatTensor(w).to(self.device), action_mapping,
                                                           is_computing_reward=False, use_gpi=False, is_test=False)

                    # employing the action to observe the next state
                    # and the corresponding rewards
                    state, reward, done, o_done = self.game.step(state, action, self.generation_method, simulator)

                    # storing the reward
                    # this is the reward obtain via Monte-Carlo sampling
                    reward = torch.tensor([reward], device=device, dtype=torch.float)
                    rewards.append(reward)

                    # collect information
                    # s, s', a, a', done to compute the TD error loss
                    old_state['next_state'] = copy.deepcopy(state)
                    old_state['act'] = action
                    old_state['done'] = 1 if done in (1, -1) else done

                    # storing the experiences to the ppo buffer
                    # each experience is (state, next_state, r, log_prob)
                    # if train_step >= 0:
                    # a failed case
                    # but for training if only consider 0, 1 for on-going or terminated conversation.
                    if done == -1:
                        done = 1

                    ppo_buffer.append((old_state, action, reward, value, log_prob, done))
                    if done:
                        break

            # update the ppo
            # starting from the second iterations
            if train_step >= 0:
                # make sure the numbef or dev items equal to the number of dev simulators
                assert len(dev_cases) == len(dev_simulators)

                # update the policy model
                loguru_logger.warning(f"Global epoch: {train_step}, Training the actor-critic model ....")
                self.train_ppo(ppo_buffer,
                               action_mapping,
                               actor_optimizer=actor_optimizer,
                               actor_scheduler=actor_scheduler,
                               critic_optimizer=critic_optimizer,
                               critic_scheduler=critic_scheduler
                               )

                # evaluate the results on the dev set
                # computationally
                if dev_cases is not None and (train_step + 1) % 20 == 0:
                    # uniform evaluation
                    if self.game_config.name == RECOMMENDATION:
                        loguru_logger.warning(f"Global epoch: {train_step}, Objective 1 Evaluation ....")
                        results = self.online_test(dev_cases, device, dev_simulators, action_mapping, stage='dev',
                                                   obj='user_reward')

                        # uniform evaluation
                        loguru_logger.warning(f"Global epoch: {train_step}, Objective 2 Evaluation ....")
                        _ = self.online_test(dev_cases, device, dev_simulators, action_mapping, stage='dev',
                                             obj='item_freq')

                    elif self.game_config.name == NEGOTIATION:
                        loguru_logger.warning(f"Global epoch: {train_step}, Objective 1 Evaluation ....")
                        results = self.online_test(dev_cases, device, dev_simulators, action_mapping, stage='dev',
                                                   obj='sl_ratio')

                        # uniform evaluation
                        loguru_logger.warning(f"Global epoch: {train_step}, Objective 2 Evaluation ....")
                        _ = self.online_test(dev_cases, device, dev_simulators, action_mapping, stage='dev',
                                             obj='fairness')

                    # logging the results
                    for logger in self.loggers:
                        logger.record(results, train_step)

                    # logging the results
                    for logger in self.loggers:
                        logger.record(_, train_step)

                    # saving the rl fine-tuning model
                    # save the rl pretrained model

                    loguru_logger.info("Saving the RL fine-tuned model .....")
                    file_path = os.path.join(self.model_config.saved_dir, "rl_model.pth")
                    self.save_model(file_path)

        loguru_logger.info("Saving the last checkpoint of the RL fine-tuned model .....")
        file_path = os.path.join(self.model_config.saved_dir, "rl_model_last.pth")
        self.save_model(file_path)

        # return data for preference training
        return None

    def predict(self, instance, w, action_mapping=None, is_test=False, is_computing_reward=False, use_gpi=True, n=10):
        """
        method that predict the action given an input instance
        :param instance: the given input instance
        :param action_mapping: a dictionary that maps action to index
        :param is_test: True if it is inference time else False
        :param is_computing_reward: True if we are computing the estimated reward function
        :return: an predicted action
        """
        # create the inverse action mapping
        inverse_action_mapping = {v: k for k, v in action_mapping.items()}

        # create the data loader
        data_loader = self.construct_dataloaders([instance],
                                                 batch_size=1,
                                                 goal2id=action_mapping,
                                                 shuffle=True,
                                                 num_workers=self.model_config.num_workers)

        # # prepare the dataloader and model using accelerator
        # data_Loader, self.model = self.accelerator.prepare(data_loader, self.model)

        # evaluation phase
        # computing the estimated reward
        if is_test:
            self.model.eval()
            # make sure no gradient pass through here.
            with torch.no_grad():
                # predict the action
                for batch in data_loader:
                    logits, value = self.model(batch)
                    action, log_prob = self.select_action(logits, is_test=True)
                    reward = None
                    action = inverse_action_mapping[action]
        else:
            # predict the action
            self.model.train()
            for batch in data_loader:
                logits, value = self.model(batch)
                action, log_prob = self.select_action(logits, is_test=is_test)
                reward = None
                print(action, log_prob)
                action = inverse_action_mapping[action]

        # return action and log prob
        return action, log_prob, value, reward
    

    def select_action(self, logits, is_test=True, eps=0.2):
        """
        method that select an action from the output logits
        :param logits: the logits output by a model
        :param is_test: True if it is inference time else false
        """
        # convert logits to probabilities
        probs = nn.functional.softmax(logits, dim=1)
        m = Categorical(probs)

        # compute policy with policy model.
        if is_test:
            # greedy sampling
            action = logits.argmax()
            return action.item(), None
        else:
            action = m.sample()
            log_prob = m.log_prob(action)
            return action.item(), log_prob

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
                                                 shuffle=False,
                                                 num_workers=self.model_config.num_workers)

        # get the model performance on the test set.
        results = self.eval_epoch(test_loader, self.create_criterion())
        return results

    def online_test(self, cases, device=None, simulators=None, action_mapping=None, stage='dev', obj='uniform'):
        """
        method that evaluate the rl-finetuned model on the test set
        :param cases: a list of situations, target_items, e.g....
        :param device: the device that we use to train the model
        :param simulators: a set of simulators to train the rl agent
        :param action_mapping: a dictionary that map (goa, topic) to index
        :return:
        """

        loguru_logger.warning(f"Online Testing on Target Item Set ......")
        loguru_logger.warning(f"Num Target Items: {len(cases)}, Num Simulators: {len(simulators)}")

        # success rate and average number of conversation turns.
        SR, total_reward = 0., 0.

        # avg_sub_reward, avg_obj_reward, avg_turn_reward
        # avg_sub_reward, avg_obj_reward, avg_turn_reward = 0.0, 0.0, 0.0

        # a tensor which is used to store the loss
        # select a particular simulator
        # and promote items to this simulator
        self.model.to(device)
        # simulator = simulators[0]

        # loop over the item set
        for idx, (case, simulator) in tqdm(enumerate(list(zip(cases, simulators)))):

            loguru_logger.info('\n================Item Num:{}===================='.format(idx))

            # IMPORTANT: the preference weight determine which aspects should be prioritied during inference
            # e.g: [0.7,0.2,0.1]: this preference favours the success rate
            # e.g: [0.2,0.7,0.1]: this preference favours other utitilies, such as fairness, item_freq or toxicity
            # e.g: [0.2, 0.2, 0.7]: this preference favours the avg turn metric.
            if stage == 'test':
                # specified objective weight
                if self.model_config.objective_weight is not None:
                    w = np.array(self.model_config.objective_weight)
                # specified objective
                else:
                    # uniform sampled weight
                    if self.model_config.prioritized_objective == "uniform":
                        w = random_weights(self.model_config.n_objectives, dist='uniform')
                    # evaluate using a given objective weight
                    # set the weight for the prioritized objective
                    # e.g: if the prioritized objective is sl_ratio then the corresponding weight is [1.0, 0.0, 0.0]
                    else:
                        w = np.array(self.model_config.obj_to_weight[self.model_config.prioritized_objective.strip()])

            # during development, we randomly sample a weight vector
            elif stage == 'dev':
                # unfirm sampling
                if obj == 'uniform':
                    w = random_weights(self.model_config.n_objectives, dist = 'uniform')
                else:
                    w = np.array(self.model_config.obj_to_weight[obj.strip()])

            # reset the game state
            # construct a new game state based on the given case and the current simulator
            state = self.game.reset(case, simulator)
            loguru_logger.info(f"Objective Weight: [{w}]")

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

            # flag for checking if the target is mentioned during the conversation
            o_flag = False

            # assign the preference weight vector
            state['w'] = w

            # a flag to check if the conversation is successful
            # for computing the success rate
            is_successful = False
            conv_turn = 0

            # interactive simulation
            # execution phase
            for t in count():  # user  dialog

                # predict the action
                action, log_prob, _, _ = self.predict(state, torch.FloatTensor(w).to(self.device), action_mapping,
                                                   is_computing_reward=False, use_gpi=False, is_test=True)

                # employing the action to observe the next state
                # and the corresponding rewards
                state, reward, done, o_done = self.game.step(state, action, self.generation_method, simulator)

                # storing the reward
                # this is reward obtained using monte-carlo sampling
                reward = torch.tensor([reward], device=device, dtype=torch.float)
                epi_reward.append(reward)

                # recommendation: check if target is mentioned during the conversation
                # negotiation: check if there is a deal between user and system
                # emotional support conversation: ......
                if o_done == 1:
                    o_flag = True

                # evaluate the outcome of the conversation
                if done:
                    # successful case
                    # if the llm_reward is greater than epsilon.
                    # recommendation: check if target is mentioned during the conversation.
                    # negotiation: check if there is a deal between user and system.
                    # emotional support conversation: check if the mental problem of the seeker is resolved.
                    if done == 1 and o_flag:
                        # increase the SR
                        SR += 1
                        is_successful = True
                    conv_turn = len(state['dialogue_context'])
                    # total_reward += epi_reward
                    break

            # construct the epi reward tensor
            # a tensor storing reward obtained during the conversation
            epi_reward = torch.cat(epi_reward, dim=0)

            # exception case, length of the conversation is 1
            if len(epi_reward.shape) == 1:
                epi_reward.unsqueeze(0)

            # objective-based epi reward
            objective_based_reward = epi_reward.sum(dim=0)

            # update the online evaluator
            # recommendation scenario
            if self.game_config.name == RECOMMENDATION:

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

                # for recommendation
                # the first objective is subjective reward
                user_reward = objective_based_reward[0].item()
                
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
                # for the negotiation
                # the first objective is sl_ratio
                # the second objective is fairness
                epi_reward = epi_reward.mean(dim=0)
                sl_ratio_reward = epi_reward[0].item()
                fairness_reward = epi_reward[1].item()
                turn_reward = epi_reward[-1].item()

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
            elif self.game_config.name == EMOTIONAL_SUPPORT:
                # the first objective is sl_ratio
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

        # return the results of the online evaluation
        return results
