import math
import os
from itertools import count

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from loguru import logger as loguru_logger
from transformers import get_linear_schedule_with_warmup

from baselines.COLOR.data_processor import COLORDataProcessorForRecommendation, COLORBridgeTorchDataset, \
    COLORPlanningTorchDataset

from base.trainer import Trainer
from logger.wandb_logger import WanDBLogger

from config.constants import RECOMMENDATION, NEGOTIATION, SUCCESS_RATE, AVG_TURN, ITEM_FREQ, USER_REWARD

from baselines.COLOR.eval_color import evaluate, evaluate_planning
from baselines.COLOR.utils import combine_tokens, get_eval_output, convert_example_to_feature_for_color_planning


class COLORTrainer(Trainer):

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

    def construct_dataloaders(self, data_instances, batch_size, stage='bridge', split='train', shuffle=True,
                              num_workers=1):
        """
        method that constructs dataloaders using given processed data instances
        :param data_instances: the processed data instances
        :param batch_size: number of batch size
        :param stage: the stage of the training process
        :param split: a string which indicates the data split
        :param shuffle: True if we shuffle the data set
        :param num_workers: number of workers used for loading the dataset
        :return: a instance of torch dataloader class
        """
        # recommendation scenario
        if self.game_config.name == RECOMMENDATION:
            # the training split
            if split == 'train':
                # the bridge training stage
                if stage == 'bridge':
                    torch_dataset = COLORBridgeTorchDataset(
                        stage=stage,
                        tokenizer=self.tokenizer,
                        instances=data_instances,
                        max_sequence_length=self.model_config.max_sequence_length,
                        device=self.device,
                        convert_example_to_feature=COLORDataProcessorForRecommendation(),
                        is_test=False
                    )
                # the planner training stage
                elif stage == 'planner':
                    torch_dataset = COLORPlanningTorchDataset(
                        stage=stage,
                        model=self.model,
                        latent_dim=self.model_config.latent_dim,
                        tokenizer=self.tokenizer,
                        instances=data_instances,
                        max_sequence_length=self.model_config.max_sequence_length,
                        device=self.device,
                        convert_example_to_feature=COLORDataProcessorForRecommendation(),
                        is_test=False
                    )
            # the splits are either the dev or test
            elif split in ['dev', 'test']:
                is_test = True
                # if it is the planner stage
                if stage == 'planner':
                    is_test = False
                torch_dataset = COLORPlanningTorchDataset(
                    stage=stage,
                    latent_dim=self.model_config.latent_dim,
                    model=self.model,
                    tokenizer=self.tokenizer,
                    instances=data_instances,
                    max_sequence_length=self.model_config.max_sequence_length,
                    device=self.device,
                    convert_example_to_feature=COLORDataProcessorForRecommendation(),
                    is_test=is_test
                )

            else:
                raise Exception("Something is wrong here ....")
        else:
            raise Exception("Something is wrong here ....")

        # construct the dataloader
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

    def train_epoch(self, stage, data_loader, optimizer, lr_scheduler, criterion, max_train_steps):
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
            # the recommendation scenario
            if self.game_config.name == RECOMMENDATION:
                # the brownian bridge training stage
                if stage == 'bridge':
                    model_output = self.model.plm.train_bridge(batch)
                    loss = model_output["contra_loss"]
                # the planner training stage
                elif stage == 'planner':
                    input_ids, input_masks = batch["input"]
                    decoder_input_all_ids, decoder_input_all_masks = batch["decoder_input_all"]
                    labels, _ = batch["label"]
                    transition_number_label = batch["transition_number"]
                    simulate_bridge_embed, simulate_bridge_mask = batch["simulate_bridge_embed"]
                    gold_bridge_embed, gold_bridge_mask = batch["gold_bridge_embed"]

                    if self.model_config.use_simulated:
                        model_output = self.model.plm(input_ids=input_ids, attention_mask=input_masks,
                                                      decoder_input_ids=decoder_input_all_ids,
                                                      decoder_attention_mask=decoder_input_all_masks, labels=labels,
                                                      bridge_embeds=simulate_bridge_embed,
                                                      bridge_mask=simulate_bridge_mask,
                                                      transition_number_label=transition_number_label)
                    else:
                        model_output = self.model.plm(input_ids=input_ids, attention_mask=input_masks,
                                                      decoder_input_ids=decoder_input_all_ids,
                                                      decoder_attention_mask=decoder_input_all_masks, labels=labels,
                                                      bridge_embeds=gold_bridge_embed, bridge_mask=gold_bridge_mask,
                                                      transition_number_label=transition_number_label)

                    lm_loss = model_output["lm_loss"]
                    if self.model_config.train_use_bridge:
                        trans_loss = model_output["trans_loss"]
                        if self.model_config.use_KLD:
                            kl_loss = model_output["kl_loss"]
                            loss = self.model_config.trans_alpha * trans_loss + self.model_config.gen_beta * lm_loss + self.model_config.kl_gamma * kl_loss
                        else:
                            loss = self.model_config.trans_alpha * trans_loss + self.model_config.gen_beta * lm_loss
                    else:
                        loss = lm_loss
            # the negotiation scenario
            # the emotional support conversation
            else:
                raise Exception('Something is wrong here ...')
            # loss = criterion(logits, batch['labels']) / self.model_config.gradient_accumulation_steps
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

    def eval_epoch(self, stage, data_loader, criterion):
        """
        method that evaluates the model on the validation set.
        :param data_loader:  the data loader used to evaluate the model
        :param criterion: the loss function
        :return: evaluation loss
        """
        self.model.eval()
        if stage == 'bridge':
            results = evaluate(self.model.plm, data_loader)
        elif stage == 'planner':
            results = evaluate_planning(self.model_config, self.model.plm, data_loader)
        return results

    def train_sft(self, stage, dataset, device=None):
        """
        method that train the model in an supervised tuning manner
        :param device: the device we use to train the model
        :return: None
        """

        # process dataset
        train_instances, dev_instances, test_instances = self.process_dataset(dataset)

        # create train, dev and test dataloaders
        train_loader = self.construct_dataloaders(train_instances,
                                                  stage=stage,
                                                  split='train',
                                                  batch_size=self.model_config.per_device_train_batch_size,
                                                  shuffle=True, num_workers=self.model_config.num_workers)

        dev_bs = self.model_config.per_device_eval_batch_size
        # for bridge training, the batch size for model evaluation should be 1
        if stage == 'bridge':
            dev_bs = 1

        dev_loader = self.construct_dataloaders(dev_instances,
                                                stage=stage,
                                                split='dev',
                                                batch_size=dev_bs,
                                                shuffle=False,
                                                num_workers=self.model_config.num_workers)

        # number of training epochs
        if stage == 'planner':
            n_epochs = self.model_config.num_planner_epochs
            lr = self.model_config.planner_learning_rate
        elif stage == 'bridge':
            n_epochs = self.model_config.num_bridge_epochs
            lr = self.model_config.bridge_learning_rate

        best_metric = -1.0 * math.inf
        # create the optimizer
        optimizer = self.create_optimizer(self.model, lr)

        # prepare the model
        self.model, optimizer, train_dataloader = self.accelerator.prepare(self.model, optimizer, train_loader)

        # compute the maximum number of training steps
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.model_config.gradient_accumulation_steps)
        max_train_steps = n_epochs * num_update_steps_per_epoch

        # create the learning rate scheduler
        lr_scheduler = self.create_scheduler(optimizer, self.model_config.num_warmup_steps, max_train_steps)

        # create the loss function
        self.criterion = self.create_criterion()

        # progress bar
        self.progress_bar = tqdm(range(max_train_steps), disable=not self.accelerator.is_local_main_process)

        # train the model
        self.model.to(device)
        for epoch in range(n_epochs):
            self.model.train()

            # reset the offline evaluator before each training epoch
            self.offline_evaluator.reset()

            # train the model
            train_loss, stop = self.train_epoch(stage=stage,
                                                data_loader=train_loader,
                                                optimizer=optimizer,
                                                lr_scheduler=lr_scheduler,
                                                criterion=self.criterion,
                                                max_train_steps=max_train_steps
                                                )
            # evaluate the performance on the current stage
            results = self.eval_epoch(stage, dev_loader, self.criterion)

            # logging the results
            for logger in self.loggers:
                logger.record(results, epoch + 1)

            # saving the model checkpoint
            # for bridge training, we use avg_similarity, for planning we use the accuracy
            if stage == 'bridge':
                metric = results['avg_similarity']
            elif stage == 'planner':
                metric = results['avg_trans_acc']

            # saving the model if needed
            # for generation, we use the loss as the saving criterion
            if metric > best_metric:
                loguru_logger.info("Performance improved. Saving the model .....")
                best_metric = metric
                # saving the checkpoint
                file_path = os.path.join(self.model_config.saved_dir, stage)
                if not os.path.exists(file_path):
                    os.mkdir(file_path)

                file_path = os.path.join(self.model_config.saved_dir, stage, "model.pth")
                self.save_model(file_path)

            if stop:
                loguru_logger.info("Training process is completed.")
                break

    def predict(self, instance, action_mapping=None, is_test=False):
        """
        method that predict the action given an input instance
        :param instance: the given input instance
        :param action_mapping: a dictionary that maps action to index
        :param is_test: True if it is inference time else False
        :return: an predicted action
        """
        feature = convert_example_to_feature_for_color_planning(self.tokenizer,
                                                                instance,
                                                                max_sequence_length=self.model_config.max_sequence_length,
                                                                is_test=is_test
                                                                )
        inputs = COLORPlanningTorchDataset.static_collate_fn([feature],
                                                             model=self.model.plm,
                                                             latent_dim=self.model_config.latent_dim,
                                                             device=self.device,
                                                             is_test=True)
        generated_output = self.model.plm.generate(inputs, self.tokenizer, args=self.model_config)
        sentences = combine_tokens(generated_output, self.tokenizer)

        goal, topic = get_eval_output(sentences[0])
        # post processing the action, topic

        goal = goal.replace('<unk>', '')
        topic = topic.replace('<unk>', '')

        return goal, topic

    def test(self, dataset):
        """
        method that evaluate the performance of the model on the test set
        :param dataset: the dataset that we want to evaluate the model performance.
        :return: the results on the test setz
        """
        # create the data loader
        test_loader = self.construct_dataloaders(dataset.test_instances,
                                                 stage='planner',
                                                 split='dev',
                                                 batch_size=self.model_config.per_device_eval_batch_size,
                                                 shuffle=False, num_workers=self.model_config.num_workers)

        # get the model performance on the test set.
        results = self.eval_epoch(stage='planner', data_loader=test_loader, criterion=self.create_criterion())
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
                                    action_mapping
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
