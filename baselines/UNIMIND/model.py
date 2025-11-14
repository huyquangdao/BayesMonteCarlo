from transformers import AutoTokenizer, BartForConditionalGeneration
from base.model import Model


class UNIMINDModel(Model):

    def __init__(self, model_config, **kwargs):
        """
        constructor for class UNIMIND policy model
        :param model_config: an instance of the model config class
        :param kwargs:
        """
        super().__init__(model_config, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.tokenizer,
                                                       cache_dir=self.model_config.cached_dir)
        self.plm = BartForConditionalGeneration.from_pretrained(self.model_config.plm,
                                                                cache_dir=self.model_config.cached_dir)

    def forward(self, batch):
        """
        forward function of the Unimind policy model
        :param batch: the current batch of data
        :return: the training loss
        """
        loss = self.plm(**batch['context'], labels=batch['labels'], return_dict=True)['loss']
        return loss
