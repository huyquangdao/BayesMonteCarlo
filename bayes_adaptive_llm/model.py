"""
Lightweight Bayes-Adaptive LLM model skeleton.
Matches the interface of TRIPModel so the trainer/pipeline can be reused,
and exposes a placeholder hook for MCTS-based preference scoring.
"""

from typing import Any, Dict, Iterable, List, Sequence, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

from base.model import Model
import gc

gc.collect()
torch.cuda.empty_cache()

class BayesAdaptiveLLMModel(Model):

    def __init__(self, model_config, **kwargs):
        super().__init__(model_config, **kwargs)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_config.tokenizer,
            cache_dir=self.model_config.cached_dir,
            torch_dtype=torch.bfloat16 if getattr(self.model_config, "bf16", False) else None,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
        )
        self.plm = AutoModelForCausalLM.from_pretrained(
            self.model_config.plm,
            cache_dir=self.model_config.cached_dir,
            torch_dtype=torch.bfloat16 if getattr(self.model_config, "bf16", False) else None,
            device_map=None,
        )

        # extend vocabulary with task-specific tokens
        if not getattr(self.model_config, "run_online_eval", False):
            self.tokenizer.add_special_tokens(self.model_config.special_tokens_dict)
            self.plm.resize_token_embeddings(len(self.tokenizer), mean_resizing=False)

        if getattr(self.tokenizer, "chat_template", None) is None:
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                "[SYSTEM] {{ message['content'] }}\n"
                "{% elif message['role'] == 'user' %}"
                "[USER] {{ message['content'] }}\n"
                "{% elif message['role'] == 'assistant' %}"
                "[ASSISTANT] {{ message['content'] }}\n"
                "{% endif %}"
                "{% endfor %}"
            )
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
    
    def post_processing_response(self, response: str):
        return response.replace('assistant', '').strip()
    
    def generate_text(self, prompt: str, max_new_tokens: int = 50, **gen_kwargs) -> str:
        self.plm.eval()
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.plm.device)
        output_ids = self.plm.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=gen_kwargs.get("do_sample", True),
            temperature=gen_kwargs.get("temperature", 0.7),
            top_p=gen_kwargs.get("top_p", 0.9),
            eos_token_id=self.tokenizer.eos_token_id,
        )
        response = self.tokenizer.decode(output_ids[0][len(inputs.input_ids[0]):], skip_special_tokens=True)
        return self.post_processing_response(response)

    def score_candidates(self,
                         dialogue_context: Sequence[Dict[str, Any]],
                         candidates: Sequence[str],
                         **kwargs) -> torch.Tensor:
        """
        Placeholder hook for future MCTS preference scoring.
        Given a dialogue context and multiple candidate actions/responses,
        return a tensor of scores so an MCTS loop can pick the highest-valued sample.
        """
        raise NotImplementedError("Candidate scoring for MCTS has not been implemented yet.")
