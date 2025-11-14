import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

from base.model import Model


class BERTModel(Model):

    def __init__(self, model_config, **kwargs):
        """
        constructor for Class BERT-based policy model
        :param model_config: the model configuration class
        :param kwargs: other keywords arguments
        """
        super().__init__(model_config, **kwargs)
        # create the tokenizer and the backbone pretrained language model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config.tokenizer,
                                                       cache_dir=self.model_config.cached_dir)
        self.plm = AutoModel.from_pretrained(self.model_config.plm, cache_dir=self.model_config.cached_dir)

        # prepend special tokens to the vocabulary and resize the embedding matrix of the PLM
        self.tokenizer.add_special_tokens(self.model_config.special_tokens_dict)
        self.plm.resize_token_embeddings(len(self.tokenizer))

        # other parameters.
        self.n_goals = self.model_config.n_goals
        self.n_topics = self.model_config.n_topics

        # if we predict both goal, topic at a time
        if self.model_config.combined_action:
            self.n_classses = self.n_goals * self.n_topics

        self.hidden_size = self.model_config.hidden_size
        self.lm_size = self.model_config.lm_size
        self.drop_out = nn.Dropout(p=self.model_config.dropout)
        self.proj_layer = nn.Linear(self.lm_size, self.hidden_size)
        self.out_layer = nn.Linear(self.hidden_size, self.n_classses)

    def forward(self, batch):
        """
        Forward function
        :param batch: a batched tensor data
        :return:
        """
        cls_token = self.plm(**batch)[0][:, 0, :]
        hidden = torch.relu(self.proj_layer(cls_token))
        hidden = self.drop_out(hidden)
        logits = self.out_layer(hidden)
        return logits
