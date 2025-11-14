from base.model import Model

class StandardPromptModel(Model):

    def __init__(self, model_config, **kwargs):
        """
        constructor for ProCOT dialogue policy
        :param model_config: the configuration of the model
        :param kwargs: other keywords arguments
        """
        super().__init__(model_config, **kwargs)
        self.model_config = model_config
        self.temperature = self.model_config.temperature
        self.max_gen_tokens = self.model_config.max_gen_tokens
        self.prompt = self.model_config.prompt

    def forward(self, inputs):
        """
        method that predicts the action using the Standard Prompting model
        :param inputs: the input of the model
        :return:
        """
        # just return the "Standard" action
        return self.prompt
