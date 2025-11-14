import math
import os
from itertools import count

from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader

from loguru import logger as loguru_logger
from transformers import get_linear_schedule_with_warmup

from baselines.UNIMIND.data_processor import UNIMINDDataProcessorForRecommendation, UnimindTorchDataset

from base.trainer import Trainer
from logger.wandb_logger import WanDBLogger

from config.constants import RECOMMENDATION, NEGOTIATION, SUCCESS_RATE, AVG_TURN, ITEM_FREQ, USER_REWARD


class UNIMINDTrainer(Trainer):

    def __init__(self, game_config, model_config, accelerator, game, model, offline_evaluator, online_evaluator,
                 loggers, generation_method=None):
        """
        constructor for class UNIMIND pipeline training
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

    def construct_dataloaders(self, tasks, data_instances, batch_size, shuffle=True, num_workers=1, is_test=False):
        """
        method that constructs dataloaders using given processed data instances
        :param data_instances: the processed data instances
        :param batch_size: number of batch size
        :param shuffle: True if we shuffle the data set
        :param num_workers: number of workers used for loading the dataset
        :param is_test: True if it is the inference time
        :return: a instance of torch dataloader class
        """
        # recommendation scenario
        if self.game_config.name == RECOMMENDATION:
            torch_dataset = UnimindTorchDataset(
                tokenizer=self.tokenizer,
                instances=data_instances,
                max_sequence_length=self.model_config.max_sequence_length,
                device=self.device,
                tasks=tasks,
                is_test=is_test,
                convert_example_to_feature=UNIMINDDataProcessorForRecommendation()
            )
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
            # the recommendation scenario
            if self.game_config.name == RECOMMENDATION:
                loss = self.model(batch)
            # the negotiation scenario
            # the emotional support conversation
            else:
                raise Exception("Something is wrong here")

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
                    # computing the loss function
                    loss = self.model(batch)
                    dev_loss.append(float(loss))

                    gen_seqs = self.accelerator.unwrap_model(self.model.plm).generate(
                        **batch['context'],
                        max_new_tokens=self.model_config.max_gen_length,
                        no_repeat_ngram_size=3
                    )
                    gen_resp_ids = []
                    for gen_seq in gen_seqs:
                        gen_seq = [token_id for token_id in gen_seq if token_id != self.model.tokenizer.pad_token_id]
                        gen_resp_ids.append(gen_seq)

                    label_resp_ids = []
                    for label_seq in batch['labels']:
                        label_seq = [token_id for token_id in label_seq if token_id != -100]
                        label_resp_ids.append(label_seq)

                    # decoding the prediction and labels
                    # convert sequences of ids to sequences of text
                    decoded_preds = self.model.tokenizer.batch_decode(gen_resp_ids, skip_special_tokens=True)
                    # for other kind of generation
                    # this block of code should be modified
                    decoded_preds = [decoded_pred.replace('</s>', '').replace('<s>', '') for decoded_pred in
                                     decoded_preds]
                    decoded_preds = [pred.strip() for pred in decoded_preds]

                    decoded_labels = self.model.tokenizer.batch_decode(label_resp_ids, skip_special_tokens=True)
                    decoded_labels = [decoded_label.replace('</s>', '').replace('<s>', '') for decoded_label in
                                      decoded_labels]
                    decoded_labels = [label.strip() for label in decoded_labels]

                    # update the evaluator
                    self.offline_evaluator.record(decoded_preds, decoded_labels)
                    dev_loss.append(float(loss))

        dev_loss = np.mean(dev_loss) * self.model_config.gradient_accumulation_steps
        results = self.offline_evaluator.report()
        results['loss'] = dev_loss
        return results

    def train_sft(self, dataset, epochs=5, tasks=None, device=None):
        """
        method that train the model in an supervised tuning manner
        :param device: the device we use to train the model
        :return: None
        """

        # process dataset
        train_instances, dev_instances, test_instances = self.process_dataset(dataset)

        # create train, dev and test dataloaders
        train_loader = self.construct_dataloaders(tasks=tasks,
                                                  data_instances=train_instances,
                                                  batch_size=self.model_config.per_device_train_batch_size,
                                                  shuffle=True, num_workers=self.model_config.num_workers)

        dev_loader = self.construct_dataloaders(tasks=tasks,
                                                data_instances=dev_instances,
                                                batch_size=self.model_config.per_device_eval_batch_size,
                                                shuffle=False,
                                                num_workers=self.model_config.num_workers)

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
        for epoch in range(epochs):
            self.model.train()

            # reset the offline evaluator before each training epoch
            self.offline_evaluator.reset()

            # train the model
            train_loss, stop = self.train_epoch(data_loader=train_loader,
                                                optimizer=optimizer,
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

                # save the model for each individual task.
                # pretraining step, we train the model for 3 different tasks
                if len(tasks) > 1:
                    file_path = os.path.join(self.model_config.saved_dir, "model.pth")
                    self.save_model(file_path)
                else:
                    # create a directory for the current task
                    file_path = os.path.join(self.model_config.saved_dir, tasks[0])
                    if not os.path.exists(file_path):
                        os.mkdir(file_path)
                    # creat a path to the model checkpoint
                    file_path = os.path.join(self.model_config.saved_dir, tasks[0], "model.pth")
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
        # create the data loader for goal prediction
        data_loader = self.construct_dataloaders(tasks=['goal'],
                                                 data_instances=[instance],
                                                 batch_size=1,
                                                 shuffle=True,
                                                 is_test=True,
                                                 num_workers=self.model_config.num_workers)

        # predict the goal first
        with torch.no_grad():
            for batch in data_loader:
                # generating the prediction
                gen_seqs = self.accelerator.unwrap_model(self.model.plm).generate(
                    **batch['context'],
                    max_new_tokens=self.model_config.max_gen_length,
                    no_repeat_ngram_size=3
                )
                gen_resp_ids = []
                for gen_seq in gen_seqs:
                    gen_seq = [token_id for token_id in gen_seq if token_id != self.model.tokenizer.pad_token_id]
                    gen_resp_ids.append(gen_seq)

                # decoding the prediction and labels
                # convert sequences of ids to sequences of text
                decoded_preds = self.model.tokenizer.batch_decode(gen_resp_ids, skip_special_tokens=True)
                # for other kind of generation
                # this block of code should be modified
                decoded_preds = [decoded_pred.replace('</s>', '').replace('<s>', '') for decoded_pred in
                                 decoded_preds]
                decoded_preds = [pred.strip() for pred in decoded_preds]

                # post-processing the prediction to get the goal
                goal = decoded_preds[0].split(':')[-1].strip()

        # update the state for predicting the topic
        instance['pred_goal'] = goal
        # create the data loader for goal prediction
        data_loader = self.construct_dataloaders(tasks=['topic'],
                                                 data_instances=[instance],
                                                 batch_size=1,
                                                 is_test=True,
                                                 shuffle=True,
                                                 num_workers=self.model_config.num_workers)

        # then we predict the topic
        with torch.no_grad():
            for batch in data_loader:
                # generating the prediction
                gen_seqs = self.accelerator.unwrap_model(self.model.plm).generate(
                    **batch['context'],
                    max_new_tokens=self.model_config.max_gen_length,
                    no_repeat_ngram_size=3
                )
                gen_resp_ids = []
                for gen_seq in gen_seqs:
                    gen_seq = [token_id for token_id in gen_seq if token_id != self.model.tokenizer.pad_token_id]
                    gen_resp_ids.append(gen_seq)

                # decoding the prediction and labels
                # convert sequences of ids to sequences of text
                decoded_preds = self.model.tokenizer.batch_decode(gen_resp_ids, skip_special_tokens=True)
                # for other kind of generation
                # this block of code should be modified
                decoded_preds = [decoded_pred.replace('</s>', '').replace('<s>', '') for decoded_pred in
                                 decoded_preds]
                decoded_preds = [pred.strip() for pred in decoded_preds]

                # post-processing the prediction to get the goal
                topic = decoded_preds[0].split(':')[-1].strip()

        action = (goal, topic)
        return action

    def test(self, tasks, dataset):
        """
        method that evaluate the performance of the model on the test set
        :param dataset: the dataset that we want to evaluate the model performance.
        :return: the results on the test setz
        """
        # create the data loader
        test_loader = self.construct_dataloaders(tasks=tasks,
                                                 data_instances=dataset.test_instances,
                                                 batch_size=self.model_config.per_device_eval_batch_size,
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

            # a flag to check if the conversation is successful
            # for computing the success rate
            is_successful = False
            conv_turn = 0

            # create two lists to store the rewards
            rewards = []

            # flag for checking if the target is mentioned during the conversation
            o_flag = False

            # interactive simulation
            for t in count():  # user  dialog

                # predict the action
                action = self.predict(state, action_mapping)
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
            else:
                raise Exception('Something is wrong here ...')

        # compute the results using the evaluator
        results = self.online_evaluator.report()

        # log the results to terminal or file
        for logger in self.loggers:
            if not isinstance(logger, WanDBLogger):
                logger.record(results, "Testing")

        # return the results of the online evaluation
        return results
