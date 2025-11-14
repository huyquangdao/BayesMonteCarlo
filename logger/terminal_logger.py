from loguru import logger
from base.logger import Logger


class TerminalLogger(Logger):

    def __init__(self, game_config, dataset_config, model_config, local_time, random_seed, model_name, **kwargs):
        """
        constructor for class Terminal LOgger
        :param game_config:
        :param dataset_config:
        :param model_config:
        :param local_time:
        :param random_seed:
        :param model_name:
        """
        super().__init__(**kwargs)
        self.scenario_config = game_config
        self.dataset_config = dataset_config
        self.model_config = model_config
        self.local_time = local_time
        self.model_name = model_name
        self.logger = logger
        self.random_seed = random_seed

        # logging the information of the scenario
        self.logger.warning("Scenario Configurations")
        for k, v in self.scenario_config.get_params():
            self.logger.info(f"[{k}]: {v}")

        # logging the information of the dataset
        self.logger.warning("Dataset Configurations")
        for k, v in self.dataset_config.get_params():
            self.logger.info(f"[{k}]: {v}")

        self.logger.warning("Model Configurations")
        # logng the model configurations to the terminal
        for k, v in self.model_config.get_params():
            self.logger.info(f"[{k}]: {v}")

        # logging information of the current run
        self.logger.warning("Runtime Information")
        self.logger.info(f"[Local Time]: {local_time}, [RandomSeed]: {random_seed} ,[Model Name]: {model_name}")

    def record(self, results, step=None):
        """
        method that records the results at a particular step
        :param results: the results at this step
        :param step: the current step
        :return: None
        """
        self.logger.info(f"[Random Seed]: {self.random_seed} , [Model Name]: {self.model_name}, [Step]: {step}")
        for k, v in results.items():
            self.logger.info(f"[Metric]: {k}, [Values]: {v}")
