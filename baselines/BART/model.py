from transformers import AutoTokenizer, AutoModel

from base.model import Model


class BARTModel(Model):

    def __init__(self, model_config, **kwargs):
        """
        constructor for class BART Model
        :param model_config: an instance of the model config class
        :param kwargs:
        """
        super().__init__(model_config, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.tokenizer,
                                                       cache_dir=self.model_config.cached_dir)
        self.plm = AutoModel.from_pretrained(self.model_config.plm, cache_dir=self.model_config.cached_dir)

    def __call__(self, *args, **kwargs):
        pass
