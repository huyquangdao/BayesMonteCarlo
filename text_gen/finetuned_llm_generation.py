import math
import os
from collections import defaultdict

from tqdm import tqdm
import numpy as np

import torch
import copy
import torch.nn as nn
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM, BitsAndBytesConfig

from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup

from peft import (
    LoraConfig,
    prepare_model_for_kbit_training,
    get_peft_model,
    PeftModel
)

from base.text_gen import LLMGeneration
from base.pipeline import Pipeline
from base.trainer import Trainer
from base.torch_dataset import BaseTorchDataset
from base.data_processor import DataProcessorForGeneration

from config.constants import RECOMMENDATION, NEGOTIATION, EMOTIONAL_SUPPORT
from base.model import Model

from loguru import logger as loguru_logger

from config.config import GenerationConfig, ModelConfig
from logger.file_logger import FileLogger
from logger.wandb_logger import WanDBLogger

from utils.generation import convert_example_to_feature_for_finetuned_llm_generation_negotiation, convert_example_to_feature_for_finetuned_llm_generation_recommendation, \
    convert_example_to_feature_for_finetuned_llm_generation_emotional_support, convert_example_to_feature_for_finetuned_llm_generation_persuation

class FinetunedLLMGenerationConfig(GenerationConfig, ModelConfig):
    """
    Finetuned LLM Generation config class
    """
    
    tokenizer = 'Qwen/Qwen2.5-1.5B-Instruct'
    plm = 'Qwen/Qwen2.5-1.5B-Instruct'

    # params 
    no_repeat_ngram_size = 3
    max_target_length = 50
    max_sequence_length = 1024
    
    q_lora: bool = True
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing = True
    
    # lora configurations for the generation adapter.
    gen_lora_target_modules: list = ["gate_proj", "down_proj", "up_proj"]
    gen_lora_r: int = 8
    gen_lora_alpha: int = 16
    gen_lora_dropout: float = 0.05
    gen_lora_weight_path: str = ""
    gen_lora_bias: str = "none"
    pass


class FinetunedLLMDataProcessorForGeneration(DataProcessorForGeneration):

    def __call__(self, *args, **kwargs):
        """
        function that convert instances to data features for the generation task
        :param args:  arguments
        :param kwargs:  keywords arguments
        :return:
        """
        # recommendation scenario
        if self.game_config.name == RECOMMENDATION:
            input_ids, attention_masks, labels = convert_example_to_feature_for_finetuned_llm_generation_recommendation(*args,
                                                                                    **kwargs)
        # negotiation scenario
        elif self.game_config.name == NEGOTIATION:
            input_ids, attention_masks, labels = convert_example_to_feature_for_finetuned_llm_generation_negotiation(*args,
                                                                                               **kwargs)
        # emotional support scenario
        elif self.game_config.name == EMOTIONAL_SUPPORT:
            input_ids, attention_masks, labels = convert_example_to_feature_for_finetuned_llm_generation_emotional_support(*args, 
                                                                                                     **kwargs)
        elif self.game_config.name == PERSUATION:
            input_ids, attention_masks, labels = convert_example_to_feature_for_finetuned_llm_generation_persuation(*args, 
                                                                                                     **kwargs)
        # invalid scenario
        else:
            raise Exception("Invalid Scenario !!")
        return input_ids, attention_masks, labels


class FinetunedLLMTorchDatasetForGeneration(BaseTorchDataset):
    """
    BART Torch Dataset for generation class
    """
    
    def __init__(self, tokenizer, instances, goal2id=None, max_sequence_length=512, padding='max_length',
                 pad_to_multiple_of=True, device=None, convert_example_to_feature=None, max_target_length=50,
                 is_test=False, is_gen=False, n_objectives=3, is_preference=False, is_so_game = True, model_path = ""):
        self.tokenizer = tokenizer
        self.instances = instances
        self.goal2id = goal2id
        self.max_sequence_length = max_sequence_length
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.device = device
        self.convert_example_to_feature = convert_example_to_feature
        self.max_target_length = max_target_length
        self.is_test = is_test
        self.is_gen = is_gen
        self.is_preference = is_preference
        self.n_objectives = n_objectives
        self.model_path = model_path
        self.is_so_game = is_so_game
        self.instances = self.preprocess_data(instances, convert_example_to_feature)
    
    def preprocess_data(self, instances, convert_example_to_feature):
        """
        method that preprocess an data instances
        @param instances: an instance from the data
        @return: a processed instance which consists of input_ids and labels
        """
        processed_instances = []
        for instance in tqdm(instances):
            # data processor for policy training
            gen_input_ids, gen_attention_mask, gen_label_ids = convert_example_to_feature(self.tokenizer, instance,
                                                                                                                                            self.max_sequence_length,
                                                                                                                                            self.goal2id,
                                                                                                                                            model_path = self.model_path,
                                                                                                                                            is_test = self.is_test
                                                                                                                                            )
            new_instance = {
                
                # input for generation
                "gen_input_ids": gen_input_ids,
                "gen_attention_mask": gen_attention_mask,
                "gen_label_ids": gen_label_ids,
            }
            
            processed_instances.append(new_instance)
        return processed_instances

    def collate_fn(self, batch):
        """
        collate function for RTCP model in the recommendation scenario
        :param batch:
        :return:
        """
        gen_input_ids = defaultdict(list)
        gen_attention_masks = defaultdict(list)
        gen_label_ids = defaultdict(list)
        
        for instance in batch:
            gen_input_ids['input_ids'].append(instance['gen_input_ids'])
            gen_attention_masks['input_ids'].append(instance['gen_attention_mask'])
            gen_label_ids['input_ids'].append(instance['gen_label_ids'])
        
        batch_gen = self.__collate(gen_input_ids, gen_attention_masks, gen_label_ids)
        return batch_gen
    
    def __collate(self, input_ids, attention_masks, label_ids):
        input_ids = torch.cat(input_ids['input_ids'], dim = 0)
        attention_masks = torch.cat(attention_masks['input_ids'], dim = 0)
        label_ids = torch.cat(label_ids['input_ids'], dim = 0)        
        new_batch = {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
            "labels": label_ids,
        }
        return new_batch


class FinetunedLLMModelForGeneration(Model):

    def __init__(self, model_config, **kwargs):
        """
        constructor for Class BERT-based policy model
        :param model_config: the model configuration class
        :param kwargs: other keywords arguments
        """
        super().__init__(model_config, **kwargs)
        
        # create the tokenizer and the backbone pretrained language model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.tokenizer,
                                                       )

        self.compute_dtype = (
                torch.float16
                if self.model_config.fp16
                else (torch.bfloat16 if self.model_config.bf16 else torch.float32)
        )
                
        # load the base model
        self.plm = AutoModelForCausalLM.from_pretrained(self.model_config.plm,
                                                        torch_dtype=self.compute_dtype,
                                                        quantization_config=BitsAndBytesConfig(
                                                            load_in_4bit=True,
                                                            bnb_4bit_use_double_quant=True,
                                                            bnb_4bit_quant_type="nf4",
                                                            bnb_4bit_compute_dtype=self.compute_dtype,
                                                        )
                                                        if self.model_config.q_lora
                                                        # if True
                                                        else None,
                                                        trust_remote_code=True
                                                        )
        # the lora config for generation
        gen_lora_config = LoraConfig(
            r=model_config.gen_lora_r,
            
            lora_alpha=model_config.gen_lora_alpha,
            target_modules= model_config.gen_lora_target_modules,
                        
            lora_dropout=model_config.gen_lora_dropout,
            bias=model_config.gen_lora_bias,
            task_type="CAUSAL_LM",
        )
        
        # create the peft model
        # planning is the default adapter
        self.plm = get_peft_model(self.plm, gen_lora_config)
        self.plm.print_trainable_parameters()
                    
    def forward(self, batch):
        outputs = self.plm(**batch)
        loss = outputs.loss
        logits = outputs.logits
        return loss, logits        
    
    def load_lora_model(self, file_path, task):
        # load the base model
        self.plm = AutoModelForCausalLM.from_pretrained(self.model_config.plm,
                                                        torch_dtype=self.compute_dtype,
                                                        quantization_config=BitsAndBytesConfig(
                                                            load_in_4bit=True,
                                                            bnb_4bit_use_double_quant=True,
                                                            bnb_4bit_quant_type="nf4",
                                                            bnb_4bit_compute_dtype=self.compute_dtype,
                                                        )
                                                        if self.model_config.q_lora
                                                        else None,
                                                        trust_remote_code=True, 
                                                        cache_dir=self.model_config.cached_dir
                                                        )
        
        # print(file_path)
        # assert 1 == 0
        
        self.plm = PeftModel.from_pretrained(self.plm, file_path, adapter_name = task)


class FinetunedLLMTrainerForGeneration(Trainer):

    def __init__(self, dataset_config, game_config, model_config, accelerator, game, model, offline_evaluator,
                 online_evaluator,
                 loggers):
        """
        constructor for class BartTrainerForGeneration
        :param dataset_config: the configuration of the dataset
        :param game_config: the game configuration
        :param model_config: the model configuration
        :param accelerator: the accelerator
        :param game: the game
        :param model: the model
        :param offline_evaluator: an instance of class offline evaluator
        :param online_evaluator: an instance of class online evaluator
        :param loggers: a set of loggers.
        """
        super().__init__(game_config, model_config, accelerator, game, model, offline_evaluator, online_evaluator,
                         loggers)
        self.dataset_config = dataset_config

    def save_model(self, file_path, task = 'default'):
        """
        method that saves the model to a particular file path
        :return: None
        """
        # save the peft model
        self.model.plm.save_pretrained(file_path, 
                                       save_embedding_layers = False, 
                                       select_adapters = task)

    def load_model(self, file_path, task): 
        # load the adapter from the file path
        self.model.load_lora_model(file_path, task) 
        
    def process_dataset(self, dataset):
        """
        method that process the given dataset and return processed data instances
        :param dataset: an instance of the Dataset class
        :return: processed data instances.
        """
        return dataset.train_instances, dataset.dev_instances, dataset.test_instances

    def construct_dataloaders(self, data_instances, batch_size, shuffle=True, num_workers=1, is_test=False):
        """
        method that constructs dataloaders using given processed data instances
        :param data_instances: the processed data instances
        :param batch_size: number of batch size
        :param shuffle: True if we shuffle the data set
        :param num_workers: number of workers used for loading the dataset
        :param is_test: True if we're in inference time, else False
        :return: a instance of torch dataloader class
        """

        # construct the torch dataset
        torch_dataset = FinetunedLLMTorchDatasetForGeneration(
            tokenizer=self.model.tokenizer,
            instances=data_instances,
            max_sequence_length=self.model_config.max_sequence_length,
            device=self.device,
            convert_example_to_feature=FinetunedLLMDataProcessorForGeneration(self.game_config, self.dataset_config),

            # for generation task
            is_gen=True,
            is_test=is_test,
            model_path = self.model.plm.config._name_or_path,
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

    def create_optimizer(self):
        """
        method that create the optimizer to train the model
        :return: a torch.optim.Optimizer
        """
        modules = [self.model]
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
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.model_config.learning_rate)
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
        self.model.plm.cuda()
        
        for step, batch in enumerate(data_loader):
            
            loss, logits = self.model(batch)
            loguru_logger.info(f"training loss: {loss}")

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

    def eval_epoch(self, data_loader, criterion, return_responses=False):
        """
        method that evaluates the model on the validation set.
        :param data_loader:  the data loader used to evaluate the model
        :param criterion: the loss function
        :param: return_responses: True if we with to save the generated responses
        :return: evaluation loss
        """
        self.model.eval()
        # development loss
        dev_loss = []
        with torch.no_grad():
            count = 0
            data_loader, self.model = self.accelerator.prepare(data_loader, self.model)
            for batch in tqdm(data_loader, disable=not self.accelerator.is_local_main_process):
                with torch.no_grad():
                    # # set the task adapter
                    # predict the action token
                    generated_tokens = self.model.plm.generate(
                        **batch,
                        max_new_tokens= 30,
                        temperature=0.000001
                    )
                    
                    outputs = [self.model.tokenizer.decode(ids[len(input_ids):], skip_special_tokens=True) for ids, input_ids in list(zip(generated_tokens, batch['input_ids']))]
                    contexts = [self.model.tokenizer.decode(ids, skip_special_tokens=True) for ids in batch['input_ids']]       
                    labels = [self.model.tokenizer.decode(ids, skip_special_tokens=True).replace(context, "") for ids, context in list(zip(batch['labels'], contexts))]
                
                    print("preds: ", outputs)
                    print("labels: ", labels)
                    self.offline_evaluator.record(outputs, labels)

        results = self.offline_evaluator.report()
                    
        # saving the generated responses and labels
        if return_responses:
            response_records = []
            for pred_res, label_res in list(zip(self.offline_evaluator.preds, self.offline_evaluator.labels)):
                response_records.append({
                    "pred": pred_res,
                    "label": label_res
                })
            return results, response_records
        
        # only return the values fo the metrics
        return results, None

    def train_sft(self, dataset, device=None):
        """
        method that train the model
        :param dataset: the dataset that we wish to train the model
        :param device: the device we use to train the model
        :return: the trained model
        """

        best_value = - math.inf

        # process dataset
        train_instances, dev_instances, test_instances = self.process_dataset(dataset)

        # create train, dev and test dataloaders
        train_loader = self.construct_dataloaders(train_instances,
                                                  batch_size=self.model_config.per_device_train_batch_size,
                                                  shuffle=True, 
                                                  num_workers=self.model_config.num_workers,
                                                  is_test=False)

        dev_loader = self.construct_dataloaders(dev_instances, 
                                                batch_size=self.model_config.per_device_eval_batch_size,
                                                shuffle=False, 
                                                num_workers=self.model_config.num_workers, 
                                                is_test=True)
        
        # create the optimizer
        self.optimizer = self.create_optimizer()
        self.model, optimizer, train_loader = self.accelerator.prepare(self.model, self.optimizer, train_loader)
        
        # compute the maximum number of training steps
        num_update_steps_per_epoch = math.ceil(len(train_loader) / self.model_config.gradient_accumulation_steps)
        max_train_steps = self.model_config.num_train_epochs * num_update_steps_per_epoch

        # create the learning rate scheduler
        lr_scheduler = self.create_scheduler(optimizer, self.model_config.num_warmup_steps, max_train_steps)

        # create the loss function
        self.criterion = self.create_criterion()

        # progress bar
        self.progress_bar = tqdm(range(max_train_steps), disable=not self.accelerator.is_local_main_process)

        # train the model
        for epoch in range(self.model_config.num_train_epochs):
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
            results, response_records = self.eval_epoch(dev_loader, self.criterion, return_responses=True)

            # logging the results
            for logger in self.loggers:
                logger.record(results, epoch + 1)

                # saving the generated response
                if isinstance(logger, FileLogger):
                    logger.save_responses(response_records, self.model_config.log_dir,
                                          file_name=f"{logger.model_name}-{logger.local_time}-{epoch + 1}-Dev.txt")


            # saving the model if needed
            # for generation, we use the loss as the saving criterion
            metric_to_save = 'BleuN'
            current_value = results[metric_to_save]
            
            if isinstance(current_value, dict):
                current_value = list(current_value.values())[0]
            
            if current_value > best_value:
                loguru_logger.info("Performance improved. Saving the model .....")
                best_value = current_value
                
                # path for the sft pretrained model      
                # file_path = os.path.join(self.model_config.saved_dir, self.game_config.name)                
                self.save_model(self.model_config.saved_dir)

            # terminal if needed
            if stop:
                print("Training process is completed.")
                break

    def predict(self, instance):
        """
        method that predict the output response given an particular input instance
        :param instance: the input instance
        :return: a generated response in form of text
        """

        # construct the data loader
        # set is test to True for inference.
        data_loader = self.construct_dataloaders([instance], 
                                                 batch_size=1, 
                                                 is_test=True,
                                                 num_workers=self.model_config.num_workers, 
                                                 shuffle=False
                                                 )
        # self.model.to(self.device)
        data_loader, self.model = self.accelerator.prepare(data_loader, self.model)
        self.model.eval()
        
        # predict the response
        for batch in tqdm(data_loader, disable=not self.accelerator.is_local_main_process):
            generated_tokens = self.model.plm.generate(
                **batch,
                max_new_tokens = 30,
                temperature=0.000001
            )
            outputs = [self.model.tokenizer.decode(ids[len(input_ids):], skip_special_tokens=True) for ids, input_ids in list(zip(generated_tokens, batch['input_ids']))]
            return outputs[0]

    def test(self, dataset):
        """
        method that evaluate the model performance on the test set,
        :param dataset: the given dataset that we want to evaluate the performance
        :return: the results of metrics and generated responses
        """
        # construct the data loader
        test_loader = self.construct_dataloaders(dataset.test_instances,
                                                 batch_size=self.model_config.per_device_eval_batch_size,
                                                 shuffle=False, 
                                                 num_workers=self.model_config.num_workers,
                                                 is_test=True
                                                 )

        # evaluate the perfomance.
        results, response_records = self.eval_epoch(test_loader, self.criterion, return_responses=True)
        return results, response_records


class FinetunedLLMPipelineForGeneration(Pipeline):
    
    def load_pretrained_model(self):
        # saved_model_path = os.path.join(self.model_config.saved_dir, self.game_config.name)
        self.trainer.load_model(self.model_config.saved_dir, 'default')

    def inference(self, instance):
        """
        method that predict the output response given an particular input instance
        :param instance: the given input instance
        :return: a generated response in the format of text
        """
        generated_response = self.trainer.predict(instance)
        return generated_response

    def run_sft(self):
        """
        This method run the whole pipeline for model training, selection and evaluation
        :return: the results of the current run.
        """

        # train and select the best model
        self.trainer.train_sft(self.dataset, self.device)

        # evaluate the performance on the test set
        # reset the offline evaluator
        self.trainer.offline_evaluator.reset()

        # load the pretrained model
        self.load_pretrained_model()

        # compute the performance on test set and generated responses
        results, response_records = self.trainer.test(self.dataset)

        # logging the test results
        for logger in self.trainer.loggers:

            # log the results to file or terminal
            # currently it is not available for Wandb Logger
            if not isinstance(logger, WanDBLogger):
                logger.record(results, "Test Set")

            # saving the generated response
            if isinstance(logger, FileLogger):
                logger.save_responses(response_records, self.model_config.log_dir,
                                      file_name=f"{logger.model_name}-{logger.local_time}-Test.txt")

        # return the results of the current run
        return results

    def execute(self):
        """
        method that execute the training pipeline
        :return:
        """
        loguru_logger.info("Fine tuning the generation model on the background dataset.....")
        sft_result = self.run_sft()
        return sft_result


class FinetunedLLMGeneration(LLMGeneration):

    def __init__(self, generation_config, pipeline, is_test=False):
        """
        constructor for class Bart Generation
        :param generation_config: an instance of the generation config
        :param pipeline: a particular pipeline
        :param is_test: True if we are using the generation class at inference time
        """
        super().__init__()
        self.generation_config = generation_config
        self.pipeline = pipeline
        self.is_test = is_test

        # creating the log dir for the scenario if it is not existed
        if not os.path.exists(self.pipeline.game_config.log_dir):
            os.mkdir(self.pipeline.game_config.log_dir)

        # utilizing this class at inference time
        if is_test:
            loguru_logger.info("Testing phase. Loading model checkpoint ......")
            # load the pretrained generation model
            self.pipeline.load_pretrained_model()

    def generate_response(self, instance, **kwargs):
        """
        function that generate the response
        :return:
        """
        generated_response = self.pipeline.inference(instance)
        return generated_response

    def prepare(self):
        """
        method that prepare the text generation model
        :return: the results of fine-tuning the generation model
        """
        results = self.pipeline.execute()
        return results
