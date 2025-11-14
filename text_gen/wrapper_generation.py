import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from base.text_gen import LLMGeneration
from config.config import GenerationConfig
from config.constants import RECOMMENDATION, NEGOTIATION, EMOTIONAL_SUPPORT

class WrapperConfigForGeneration(GenerationConfig):
    """
    the configuration class for the vicuna response generation
    """
    pass

class WrapperGeneration(LLMGeneration):
    
    def __init__(self, generation_config, param_a, param_b):
        pass
    
    def set_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
    
    def set_accelerator(self, accelerator):
        self.accelerator = accelerator
    
    def set_generation_model(self, generation_model):
        """
        method that set the generation model
        :param generation_model: the generation model
        :return: None
        """
        self.generation_model = generation_model
    
    def set_dataloader_config(self, dataloader_config):
        self.dataloader_config = dataloader_config
    
    def set_generation_config(self, generation_config):
        """
        method that set the generation configuration
        :param generation_config: the configuration of the generation method
        :return: None
        """
        self.generation_config = generation_config
    
    def set_dataloader_construction(self, dataloader_construction):
        """
        method that set the dataloader
        :param dataloader: the dataloader
        :return: None
        """
        self.dataloader_construction = dataloader_construction
    
    def generate_response(self, instance, task = 'generation', **kwargs):
        """
        method that generated a generated response using vicuna model
        :param instance: the current state of the conversation
        :return: a text response
        """
        # construct the dataloader for generation
        data_loader = self.dataloader_construction(
            [instance],
            **self.dataloader_config
        )        
        
        # generate the response
        with torch.no_grad():
            data_loader, self.generation_model = self.accelerator.prepare(data_loader, self.generation_model)
            count = 0
            for batch in data_loader:
                with torch.no_grad():
                    # predict the action token
                    generated_tokens = self.generation_model.generate(
                        **batch[f"batch_{task}"],
                        max_new_tokens = 30
                    )
                    response = [self.tokenizer.decode(ids[len(input_ids):], skip_special_tokens=True) for ids, input_ids in list(zip(generated_tokens, batch[f"batch_{task}"]['input_ids']))]
                    response = response[0]
                    print("response: ", response)
        return response
