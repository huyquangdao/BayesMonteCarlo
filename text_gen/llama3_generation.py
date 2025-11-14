import time

from base.text_gen import LLMGeneration
from config.config import GenerationConfig
# reuse the function designed for chatgpt
from utils.generation import construct_prompt_for_chat_gpt_response_generation_negotiation, \
    construct_prompt_for_chat_gpt_response_generation_emotional_support, \
    construct_prompt_for_chat_gpt_response_generation_recommendation, construct_prompt_for_chat_gpt_response_generation_persuation
from utils.prompt import call_llm

from config.constants import EMOTIONAL_SUPPORT, RECOMMENDATION, NEGOTIATION, PERSUATION


class Llama3ConfigForGeneration(GenerationConfig):
    # the prompt used for
    prompt = "This is the prompt and subjected to be changed"
    temperature = 0.1
    model_type = "llama3"


class Llama3Generation(LLMGeneration):

    def __init__(self, generation_config, pipeline=None, is_test=False):
        """
        constructor for class Chatgpt generation
        :param generation_config: the configuration of the generation method
        :param pipeline: pipeline used to prepare the generation method, for chatgpt, we do not need any pipeline
        :param is_test: True if we are using the generation method at inference time
        """
        super().__init__()
        self.generation_config = generation_config
        self.pipeline = pipeline
        self.is_test = is_test

    def generate_response(self, instance, **kwargs):
        """
        method that generates the response using chatgpt.
        :param instance: the current state of the conversation
        :return:
        """
        dialogue_context = instance['dialogue_context']
        # the recommendation scenario
        if self.generation_config.scenario_name == RECOMMENDATION:
            messages, goal_description = construct_prompt_for_chat_gpt_response_generation_recommendation(instance,
                                                                                                          self.generation_config.prompt)
        # the negotiation scenario
        elif self.generation_config.scenario_name == NEGOTIATION:
            messages, goal_description = construct_prompt_for_chat_gpt_response_generation_negotiation(instance,
                                                                                                       self.generation_config.prompt)
        # the emotional support conversation
        elif self.generation_config.scenario_name == EMOTIONAL_SUPPORT:
            messages, goal_description = construct_prompt_for_chat_gpt_response_generation_emotional_support(instance,
                                                                                                             self.generation_config.prompt)
        # the emotional support conversation
        elif self.generation_config.scenario_name == PERSUATION:
            messages, goal_description = construct_prompt_for_chat_gpt_response_generation_persuation(instance,
                                                                                                    self.generation_config.prompt
                                                                                                    )
        else:
            raise Exception("Invalid Scenario ...")

        messages.extend(dialogue_context)
        
        # Incorporating strategy description at the later of the prompt improve the alignment
        # between the predicted dialogue strategy and the generated response.
        messages.append(
            {'role': 'user', 'content': f"{goal_description}. "
                                        'Please reply with only one short and succinct sentence.'}
        )

        # calling the llm for response generation
        t = time.time()
        response = call_llm(messages, 
                            n=1,
                            temperature=0.7,
                            max_token=self.generation_config.max_gen_length,
                            model_type= self.generation_config.model_type,
                            **kwargs
                            )

        print("Response Generation Time: ", time.time() - t)
        # returning the response
        return response[0]