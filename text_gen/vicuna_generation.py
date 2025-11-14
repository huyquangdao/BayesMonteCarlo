import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from base.text_gen import LLMGeneration
from config.config import GenerationConfig
from utils.generation import construct_prompt_for_vicuna_response_generation_negotiation, \
    construct_prompt_for_vicuna_response_generation_recommendation, \
    construct_prompt_for_vicuna_response_generation_emotional_support

from config.constants import RECOMMENDATION, NEGOTIATION, EMOTIONAL_SUPPORT


class VicunaGenerationConfig(GenerationConfig):
    """
    the configuration class for the vicuna response generation
    """
    device = 'cuda'
    temperature = 0.7
    num_gpus = 1
    max_gpu_memory = 1
    load_8bit = False
    cpu_offloading = False
    debug = False
    early_stopping = False
    prompt = "This should be changed"


class VicunaGeneration(LLMGeneration):

    def __init__(self, generation_config, pipeline=None, is_test=False):
        """
        constructor for class Vicuna Response Generation
        :param generation_config: the configuration of the Vicuna generation method
        :param pipeline: the pipeline, for LLM the pipeline should be None.
        :param is_test: True if we're at testing time.
        """
        super().__init__()
        self.generation_config = generation_config
        self.pipeline = pipeline
        self.is_test = is_test

        # initializing the vicuna tokenizer and vicuna model
        self.vicuna_model = AutoModelForCausalLM.from_pretrained(
            self.generation_config.model_path,
            device_map="auto",
            offload_folder="offload",
            torch_dtype=torch.float16
        )

        self.vicuna_tokenizer = AutoTokenizer.from_pretrained(
            self.generation_config.model_path
        )

    def generate_response(self, instance):
        """
        method that generated a generated response using vicuna model
        :param instance: the current state of the conversation
        :return: a text response
        """
        # construct the prompt for the vicuna model
        # for the negotiation scenario
        if self.generation_config.scenario_name == NEGOTIATION:
            prompt = construct_prompt_for_vicuna_response_generation_negotiation(instance,
                                                                                 self.generation_config.prompt)
        # construct the prompt for the recommendation scenario
        elif self.generation_config.scenario_name == RECOMMENDATION:
            prompt = construct_prompt_for_vicuna_response_generation_recommendation(instance,
                                                                                    self.generation_config.prompt)
        # construct the prompt for the emotional support conversation
        elif self.generation_config.scenario_name == EMOTIONAL_SUPPORT:
            prompt = construct_prompt_for_vicuna_response_generation_emotional_support(instance,
                                                                                       self.generation_config.prompt)
        else:
            raise Exception('Invalid Scenario ....')
        # construct the input ids
        input_ids = self.vicuna_tokenizer([prompt]).input_ids
        # generate the response

        # using the vicuna model
        output_ids = self.vicuna_model.generate(
            torch.as_tensor(input_ids).cuda(),
            max_new_tokens=self.generation_config.max_gen_length,
            temperature=self.generation_config.temperature,
            early_stopping=self.generation_config.early_stopping
        )
        output_ids = output_ids[0][len(input_ids[0]):]

        # de-tokenize the output ids
        response = self.vicuna_tokenizer.decode(output_ids, skip_special_tokens=True,
                                                spaces_between_special_tokens=False)
        return response
