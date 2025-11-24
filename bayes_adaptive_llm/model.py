"""
Lightweight Bayes-Adaptive LLM model skeleton.
Matches the interface of TRIPModel so the trainer/pipeline can be reused,
and exposes a placeholder hook for MCTS-based preference scoring.
"""

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

from base.model import Model


class BayesAdaptiveLLMModel(Model):
    """
    Classification-style policy head on top of a PLM.
    """

    def __init__(self, model_config, **kwargs):
        super().__init__(model_config, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.tokenizer,
            cache_dir=self.model_config.cached_dir,
        )
        self.plm = AutoModel.from_pretrained(
            self.model_config.plm,
            cache_dir=self.model_config.cached_dir,
        )
        self.config = self.plm.config

        # extend vocabulary with task-specific tokens
        self.tokenizer.add_special_tokens(self.model_config.special_tokens_dict)
        self.plm.resize_token_embeddings(len(self.tokenizer))

        self.n_classes = self._infer_num_actions()
        self.drop_out = nn.Dropout(p=getattr(self.model_config, "dropout", 0.1))
        self.out_layer = nn.Linear(self.model_config.lm_size, self.n_classes)

    def _infer_num_actions(self) -> int:
        n_goals = getattr(self.model_config, "n_goals", 1)
        n_topics = getattr(self.model_config, "n_topics", 1)
        return n_goals * n_topics if self.model_config.combined_action else n_goals

    def forward(self, batch: Dict[str, torch.Tensor]):
        """
        Encode context with the PLM, take the [CLS] token and project to action logits.
        """
        cls_token = self.plm(**batch["context"]).last_hidden_state[:, 0, :]
        cls_token = self.drop_out(cls_token)
        logits = self.out_layer(cls_token)
        return logits

    # Expose embedding accessors expected by TRL trainers.
    def get_input_embeddings(self):
        return self.plm.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.plm.set_input_embeddings(new_embeddings)

    # Expose gradient checkpointing toggles expected by HF trainers.
    def gradient_checkpointing_enable(self, **kwargs):
        if hasattr(self.plm, "gradient_checkpointing_enable"):
            return self.plm.gradient_checkpointing_enable(**kwargs)
        return None

    def gradient_checkpointing_disable(self):
        if hasattr(self.plm, "gradient_checkpointing_disable"):
            return self.plm.gradient_checkpointing_disable()
        return None
