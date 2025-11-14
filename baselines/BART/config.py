from config.config import ModelConfig


class BARTConfig(ModelConfig):
    tokenizer = ''
    plm = ''
    lm_size = 768
    combined_action = True

    def __init__(self, params):
        """
        constructor for class Bart config
        :param params: a dictionary that contains parameter names and their values
        """
        super().__init__()
        for k, v in params.items():
            setattr(self, k, v)
