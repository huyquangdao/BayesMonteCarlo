from base.model import Model
from utils.prompt import call_llm
from baselines.AnE.config import AnEConfigForRecommendation


class AnEModel(Model):

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
        self.q1_prompt = self.model_config.q1_prompt
        self.q2_prompt = self.model_config.q2_prompt
        self.q3_prompt = self.model_config.q3_prompt

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
            
        # answering question 1
        # prompt negotiation, emotional support and persuasion
        if not isinstance(self.model_config, AnEConfigForRecommendation):
            system_prompt = self.prompt
        # prompt for recommendation
        else:
            target_item = kwargs["target_item"]
            domain = kwargs["domain"]
            system_prompt = self.prompt.format(domain, target_item)
            del kwargs["target_item"]
            del kwargs["domain"]
        
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.q1_prompt.format(dialogue)},
        ]
                
        q1_response = \
            call_llm(prompt, temperature=0.001, 
                     max_token=self.max_gen_tokens, 
                     model_type=self.model_config.model_type, **kwargs)[0]

        # answering question 2
        prompt = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self.q1_prompt.format(dialogue)},
            {"role": "assistant", "content": q1_response},
            {"role": "user", "content": self.q2_prompt.format(dialogue)},
        ]
        q2_response = \
            call_llm(prompt, temperature=0.001, max_token=self.max_gen_tokens, 
                     model_type=self.model_config.model_type, **kwargs)[0]

        # answering question 3
        prompt = [
            {"role": "system", "content":  system_prompt},
            {"role": "user", "content": self.q1_prompt.format(dialogue)},
            {"role": "assistant", "content": q1_response},
            {"role": "user", "content": self.q2_prompt.format(dialogue)},
            {"role": "assistant", "content": q2_response},
            {"role": "user", "content": self.q3_prompt.format(dialogue)},
        ]
        q3_response = \
            call_llm(prompt, temperature=0.001, 
                     max_token=self.max_gen_tokens, 
                     model_type=self.model_config.model_type,
                     **kwargs)[0]

        # calling the llm to predict the action
        return q3_response
