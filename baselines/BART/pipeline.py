from torch.utils.data import DataLoader

from base.pipeline import Pipeline
from baselines.BART.data_processor import BARTDataProcessorForRecommendation, BARTTorchDatasetForRecommendation


class BARTPipelineForRecommendation(Pipeline):

    def process_dataset(self, dataset):
        """
        method that process the given dataset and return processed data instances
        :param dataset: an instance of the Dataset class
        :return: processed data instances.
        """
        return dataset.train_instances, dataset.dev_instances, dataset.test_instances

    def construct_dataloaders(self, data_instances, batch_size, shuffle=True, num_workers=1):
        """
        method that constructs dataloaders using given processed data instances
        :param data_instances: the processed data instances
        :param batch_size: number of batch size
        :param shuffle: True if we shuffle the data set
        :param num_workers: number of workers used for loading the dataset
        :return: a instance of torch dataloader class
        """
        torch_dataset = BARTTorchDatasetForRecommendation(
            tokenizer=self.model.tokenizer,
            instances=data_instances,
            goal2id=self.config.goal2id,
            max_sequence_length=self.config.max_sequence_length,
            device=self.device,
            convert_example_to_feature=BARTDataProcessorForRecommendation()
        )

        dataloader = DataLoader(
            torch_dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=torch_dataset.collate_fn,
        )
        return dataloader

    def run_sft(self, dataset, offline_evaluator=None, loggers=None):
        """
        This method run the whole pipeline for model training, selection and evaluation
        :param dataset: an instance of Dataset class
        :param offline_evaluator: a instance of class Offline evaluator
        :param loggers: a set of loggers
        :return: the results of the current run.
        """
        # process dataset
        train_instances, dev_instances, test_instances = self.process_dataset(dataset)

        # create train, dev and test dataloaders
        train_loader = self.construct_dataloaders(train_instances, batch_size=self.config.per_device_train_batch_size,
                                                  shuffle=True, num_workers=self.config.num_workers)

        dev_loader = self.construct_dataloaders(dev_instances, batch_size=self.config.per_device_eval_batch_size,
                                                shuffle=False, num_workers=self.config.num_workers)

        test_loader = self.construct_dataloaders(test_instances, batch_size=self.config.per_device_eval_batch_size,
                                                 shuffle=False, num_workers=self.config.num_workers)

        # train and select the best model
        self.trainer.train(self.model, train_loader, dev_loader, offline_evaluator, loggers, self.device)

        results = self.trainer.eval_epoch()

        # return the results of the current run
        return results
