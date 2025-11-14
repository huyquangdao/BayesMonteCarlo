from config.config import ModelConfig


class BERTConfig(ModelConfig):
    tokenizer = 'bert-large-case'
    plm = 'bert-large-case'
    lm_size = 768
    hidden_size = 128
    combined_action = True
    run_sft = True

    def __init__(self, params):
        """
        constructor for class Bert config
        :param params: a dictionary that contains parameters and their values
        """
        super().__init__()
        for k, v in params.items():
            setattr(self, k, v)
