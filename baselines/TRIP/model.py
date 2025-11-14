import torch
import torch.nn as nn
from torch.distributions import Categorical
from transformers import AutoTokenizer, AutoModel

from base.model import Model
from utils.prompt import call_llm
from config.constants import LLAMA3, QWEN, CHATGPT

from baselines.TRIP.config import TRIPConfig, TRIPConfigForRecommendation, TRIPConfigForNegotiation, TRIPConfigForEmotionalSupport, TRIPConfigForPersuation

class TRIPUserModel(Model):

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
        self.cot_prompt = self.model_config.cot_prompt

    def forward(self, inputs, **kwargs):
        """
        method that predicts the action using the ProCOT model
        :param inputs: the input of the model
        :return:
        """
        # constructing the prompt for ProCOT
        dialogue = ''
        for utt in inputs:
            dialogue += f"{utt['role']}: {utt['content']} "
        prompt = [
            {"role": "system", "content": self.prompt},
            {"role": "user", "content": self.cot_prompt.format(dialogue)}
        ]
        # calling the llm to predict the action
        responses = call_llm(prompt, temperature=self.temperature, max_token=50,
                             model_type=self.model_config.model_type,
                             n = 1,
                             **kwargs
                             )
        
        return responses[0].strip().split(":")[-1]

class TRIPModel(Model):

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

        # if we predict both goal, topic at a
        # only applicable for the recommendation scenario
        if self.model_config.combined_action:
            # other parameters.
            # for recommendation, for other scenarios, new code might be needed.
            self.n_classses = self.model_config.n_goals * self.model_config.n_topics
        # other scenarios such as negotiation and emotional support conversation
        else:
            self.n_classses = self.model_config.n_goals

        self.drop_out = nn.Dropout(p=self.model_config.dropout)
        self.out_layer = nn.Linear(self.model_config.lm_size, self.n_classses)
        self.user_model = TRIPUserModel(self.model_config, **kwargs)

    def forward(self, batch):
        """
        Forward function
        :param batch: a batched tensor data
        :return:
        """
        cls_token = self.plm(**batch['context'])[0][:, 0, :]
        # appluing the dropout label
        cls_token = self.drop_out(cls_token)
        # producing the outputs
        logits = self.out_layer(cls_token)
        return logits

    def infering_user_mental_states(self, dialogue_context, **kwargs):
        return self.user_model(dialogue_context, **kwargs)


    def rewrite_action(self, inputs, action, **kwargs):
        dialogue = ''
        for utt in inputs:
            if isinstance(self.model_config, TRIPConfigForRecommendation):
                if utt['role'] == "user":
                    role = "User"
                else:
                    role = "Recommender"
            elif isinstance(self.model_config, TRIPConfigForNegotiation):
                if utt['role'] == "user":
                    role = "Seller"
                else:
                    role = "Buyer"
            elif isinstance(self.model_config, TRIPConfigForEmotionalSupport):
                if utt['role'] == "user":
                    role = "Therapist"
                else:
                    role = "Patient"
            elif isinstance(self.model_config, TRIPConfigForPersuation):
                if utt['role'] == "user":
                    role = "Persuadee"
                else:
                    role = "Persuadee"
            dialogue += f"{role}: {utt['content']} "
                            
        meta_prompt = self.model_config.meta_prompt                    
        prompt = [
            {"role": "system", "content": self.model_config.rewrite_prompt},
            {"role": "user", "content": self.model_config.rewrite_prompt_cot.format(meta_prompt, dialogue, action)}
        ]
        
        print("action: ", action)
        
        # calling the llm to predict the action
        responses = call_llm(prompt, temperature=0.6, 
                             max_token= 50,
                             model_type= self.model_config.model_type,
                             **kwargs
                             )
        
        # print(action)
        return responses[0].split(":")[-1].replace("\"", "").replace(".", "").strip()
    