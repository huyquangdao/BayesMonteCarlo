import os
import math

from tqdm import tqdm
import numpy as np
import torch
from loguru import logger as loguru_logger
from transformers import get_linear_schedule_with_warmup

from base.trainer import Trainer


class BERTTrainer(Trainer):

    def create_criterion(self):
        """
        method that create the loss function to train the model
        :return: a torch.nn.CrossEntropyLoss object
        """
        return torch.nn.CrossEntropyLoss()

    def create_optimizer(self, model):
        """
        method that create the optimizer to train the model
        :param model: the given model that we wish to train
        :return: a torch.optim.Optimizer
        """
        modules = [model]
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for model in modules for n, p in model.named_parameters()
                           if not any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for model in modules for n, p in model.named_parameters()
                           if any(nd in n for nd in no_decay) and p.requires_grad],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.config.learning_rate)
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

    def train_epoch(self, model, data_loader, optimizer, lr_scheduler, criterion, max_train_steps):
        """
        method that trains the model on one epoch
        :param model: the given model that we wish to train
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
            logits = model(batch['context'])
            loss = criterion(logits, batch['labels']) / self.config.gradient_accumulation_steps
            self.accelerator.backward(loss)
            train_loss.append(float(loss))

            self.progress_bar.update(1)
            self.global_step += 1

            # optim step
            if step % self.config.gradient_accumulation_steps == 0 or step == len(data_loader) - 1:
                if self.config.max_grad_norm is not None:
                    self.accelerator.clip_grad_norm_(model.parameters(), self.config.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if self.global_step >= max_train_steps:
                stop = True
                break

        # compute average train loss
        train_loss = np.mean(train_loss) * self.config.gradient_accumulation_steps
        return train_loss, stop

    def eval_epoch(self, model, data_loader, criterion):
        """
        method that evaluates the model on the validation set.
        :param model:  the given model
        :param data_loader:  the data loader used to evaluate the model
        :param criterion: the loss function
        :return: evaluation loss
        """
        dev_loss = []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(data_loader, disable=not self.accelerator.is_local_main_process):
                with torch.no_grad():
                    logits = model(batch['context'])
                    loss = criterion(logits, batch['labels'])
                    self.offline_evaluator.record(logits, batch['labels'])
                    dev_loss.append(float(loss))

        dev_loss = np.mean(dev_loss) * self.config.gradient_accumulation_steps
        results = self.offline_evaluator.report()
        results['loss'] = dev_loss
        return results

    def train_sft(self, model, train_loader, dev_loader, device=None):
        """
        method that train the model
        :param model: the given that we wish to train
        :param train_loader: the train data loader
        :param dev_loader: the dev data loader
        :param device: the device we use to train the model
        :return: the trained model
        """

        best_loss = math.inf
        # create the optimizer
        self.optimizer = self.create_optimizer(model)
        model, optimizer, train_dataloader = self.accelerator.prepare(model, self.optimizer, train_loader)

        # compute the maximum number of training steps
        num_update_steps_per_epoch = math.ceil(len(train_dataloader) / self.config.gradient_accumulation_steps)
        max_train_steps = self.config.num_train_epochs * num_update_steps_per_epoch

        # create the learning rate scheduler
        lr_scheduler = self.create_scheduler(optimizer, self.config.num_warmup_steps, max_train_steps)

        # create the loss function
        self.criterion = self.create_criterion()

        # progress bar
        self.progress_bar = tqdm(range(max_train_steps), disable=not self.accelerator.is_local_main_process)

        # train the model
        model.to(device)
        for epoch in range(self.config.num_train_epochs):
            model.train()

            # reset the offline evaluator before each training epoch
            self.offline_evaluator.reset()

            # train the model
            train_loss, stop = self.train_epoch(model, data_loader=train_loader, optimizer=optimizer,
                                                lr_scheduler=lr_scheduler,
                                                criterion=self.criterion,
                                                max_train_steps=max_train_steps
                                                )
            # evaluate the performance
            results = self.eval_epoch(model, dev_loader, self.criterion)

            # logging the results
            for logger in self.loggers:
                logger.record(results, epoch + 1)

            # saving the model if needed
            # for generation, we use the loss as the saving criterion
            if results['loss'] < best_loss:
                loguru_logger.info("Performance improved. Saving the model .....")
                best_loss = results['loss']
                file_path = os.path.join(self.config.saved_dir, "model.pth")
                self.save_model(model, file_path)

            # saving the model if needed
            if stop:
                print("Training process is completed.")
                break

        return model
