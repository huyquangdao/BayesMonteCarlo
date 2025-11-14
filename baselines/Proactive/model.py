import re

from base.model import Model
from utils.prompt import call_llm
from baselines.Proactive.config import ProactiveConfigForRecommendation, ProactiveConfigForNegotiation, ProactiveConfigForEmotionalSupport, ProactiveConfigForPersuation


class ProactiveModel(Model):

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
            if isinstance(self.model_config, ProactiveConfigForRecommendation):
                if utt['role'] == "user":
                    role = "User"
                else:
                    role = "Recommender"
            elif isinstance(self.model_config, ProactiveConfigForNegotiation):
                if utt['role'] == "user":
                    role = "Seller"
                else:
                    role = "Buyer"
            elif isinstance(self.model_config, ProactiveConfigForEmotionalSupport):
                if utt['role'] == "user":
                    role = "Therapist"
                else:
                    role = "Patient"
            elif isinstance(self.model_config, ProactiveConfigForPersuation):
                if utt['role'] == "user":
                    role = "Persuadee"
                else:
                    role = "Persuadee"

            dialogue += f"{role}: {utt['content']} "
    
        # prompt for negotiation, emotional support and persuation
        if not isinstance(self.model_config, ProactiveConfigForRecommendation):                
            prompt = [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": self.cot_prompt.format(dialogue)}
            ]
        # prompt for recommendation
        else:
            target_item = kwargs["target_item"]
            goals = kwargs["goals"]
            domain = kwargs["domain"]
            goal_string = ""
            for goal in goals:
                goal_string += f"\"{goal}\", "
            
            if domain == "poi":
                domain = "Point of Interest"
            
            prompt = [
                {"role": "system", "content": self.prompt.format(domain, target_item, goal_string)},
                {"role": "user", "content": self.cot_prompt.format(dialogue)}
            ]
            
            del kwargs["target_item"]
            del kwargs["goals"]
            del kwargs["domain"]
             
        
        # print(prompt)
        # assert 1 == 0
                                            
        # calling the llm to predict the action
        responses = call_llm(prompt, 
                             temperature=0.7, 
                             max_token=self.max_gen_tokens,
                             model_type=self.model_config.model_type,
                             **kwargs
                             )
        
        # print(prompt)
        # assert 1 == 0
                
        strategy = responses[0].split(":")[-1].replace("\"", "").replace(".", "").strip()
        # print("Strategy: ", strategy.split("\n")[0])
        # print("Proposed Price: ", proposed_price)
        
        return strategy.split("\n")[0]

    def rewrite_action(self, inputs, action, **kwargs):
        dialogue = ''
        for utt in inputs:
            if isinstance(self.model_config, ProactiveConfigForRecommendation):
                if utt['role'] == "user":
                    role = "User"
                else:
                    role = "Recommender"
            elif isinstance(self.model_config, ProactiveConfigForNegotiation):
                if utt['role'] == "user":
                    role = "Seller"
                else:
                    role = "Buyer"
            elif isinstance(self.model_config, ProactiveConfigForEmotionalSupport):
                if utt['role'] == "user":
                    role = "Therapist"
                else:
                    role = "Patient"
            elif isinstance(self.model_config, ProactiveConfigForPersuation):
                if utt['role'] == "user":
                    role = "Persuadee"
                else:
                    role = "Persuadee"
            dialogue += f"{role}: {utt['content']} "
            
        prompt = [
            {"role": "system", "content": self.model_config.rewrite_prompt},
            {"role": "user", "content": self.model_config.rewrite_prompt_cot.format(dialogue, action)}
        ]
                
        # calling the llm to predict the action
        responses = call_llm(prompt, 
                             temperature=0.7, 
                             max_token= 50,
                             model_type=self.model_config.model_type,
                             **kwargs
                             )
        
        # print(responses)
        # assert 1 == 0
        return responses[0].split(":")[-1].replace("\"", "").replace(".", "").strip()
