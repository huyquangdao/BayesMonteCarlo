from abc import ABC, abstractmethod


class Logger(ABC):

    def __init__(self, **kwargs):
        """
        constructor for abstract class Logger
        """
        pass

    @abstractmethod
    def record(self, results, steps=None):
        """
        method that record the results at a particular step
        :param results: the current results, in form of a dictionary
        :param steps:
        :return:
        """
        raise NotImplementedError("This method must be implemented")
