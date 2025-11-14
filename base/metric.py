from abc import ABC, abstractmethod


class Metric(ABC):

    def __init__(self, params=None):
        """
        constructor for abstract class metric
        :param params: set of parameters
        """
        if params is not None:
            for k, v in params:
                setattr(self, k, v)

    @abstractmethod
    def compute(self):
        """
        method that computes values of the metric
        :return: None
        """
        raise NotImplementedError("This method must be implemented")
