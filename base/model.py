import torch.nn as nn


class Model(nn.Module):

    def __init__(self, model_config, **kwargs):
        """
        constructor for class model which inherits the Module Class from Torch
        :param model_config: the model configurations
        :param kwargs: other keywords arguments
        """
        super().__init__()
        self.model_config = model_config

    def forward(self, batch):
        """
        the forward method
        :param batch: the input batch
        :return: outputs of the model
        """
        raise NotImplementedError("This method must be implemented for every model")
