from abc import ABC, abstractmethod


class Evaluator(ABC):

    def __init__(self):
        """
        Constructor for class Evaluator
        """
        pass

    @abstractmethod
    def record(self):
        """
        Method that record values of metrics
        :return: None
        """
        raise NotImplementedError("Must implement")

    @abstractmethod
    def report(self):
        """
        Method that reports the values of metrics
        :return:
        """
        raise NotImplementedError("Must implement")

    @abstractmethod
    def reset(self):
        """
        Method that reset the values of metrics
        :return:
        """
        raise NotImplementedError("Must implement")
