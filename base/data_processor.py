from abc import ABC, abstractmethod


class DataProcessor(ABC):

    @abstractmethod
    def __call__(self, *args, **kwargs):
        raise NotImplementedError("This method must be implemented for every model")


class DataProcessorForRecommendation(DataProcessor):
    pass


class DataProcessorForNegotiation(DataProcessor):
    pass


class DataProcessorForEmotionalSupport(DataProcessor):
    pass


class DataProcessorForPersuation(DataProcessor):
    pass


class DataProcessorForPreferenceEstimation(DataProcessor):
    pass


class DataProcessorForGeneration(DataProcessor):

    def __init__(self, game_config, dataset_config):
        """
        constructor for class Data processor for generation
        :param game_config: an instance of the scenario config class
        :param dataset_config: an instance of the dataset config class
        """
        super().__init__()
        self.game_config = game_config
        self.dataset_config = dataset_config
