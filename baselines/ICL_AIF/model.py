from base.model import Model
from utils.prompt import call_llm
from baselines.ICL_AIF.config import ICLAIFConfigForRecommendation

class ICLAIFModel(Model):

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
        
        # prompt for negotiation, emotional support and persuation
        if not isinstance(self.model_config, ICLAIFConfigForRecommendation):
            prompt = [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": self.cot_prompt.format(dialogue)}
            ]
        # for recommendation scenario
        else:
            target_item = kwargs["target_item"]
            domain = kwargs["domain"]
            prompt = [
                {"role": "system", "content": self.prompt.format(domain, target_item)},
                {"role": "user", "content": self.cot_prompt.format(dialogue)}
            ]   
            del kwargs["target_item"]
            del kwargs["domain"]
        
        # calling the llm to predict the action
        responses = call_llm(prompt, 
                             temperature=0.6, 
                             max_token=self.max_gen_tokens,
                             model_type=self.model_config.model_type,
                             **kwargs
                             )
        
        return responses[0].split(":")[-1].replace("\"", "").replace(".", "").strip()
