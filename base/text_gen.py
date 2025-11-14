from abc import ABC, abstractmethod


class ResponseGeneration(ABC):

    def __init__(self):
        """
        constructor for abstract class text generation
        """
        pass

    @abstractmethod
    def generate_response(self, instance):
        """
        function that generates the response
        :return: None
        """
        raise NotImplementedError("This method must be implemented")


class PLMGeneration(ResponseGeneration):
    """
    Abstract class text generation model for the recommendation scenario
    """

    @abstractmethod
    def prepare(self):
        """
        method that prepares the text generation model including training, model selection
        :return: None
        """
        raise NotImplementedError("This method must be implemented")


class LLMGeneration(ResponseGeneration):
    """
    Abstract class text generation model for the negotiation scenario
    """

    def __init__(self):
        super().__init__()
