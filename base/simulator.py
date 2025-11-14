from abc import ABC

from config.constants import USER, ASSISTANT, CHATGPT, LLAMA3
from utils.prompt import call_llm


class Simulator(ABC):
    temperature = 1.1
    max_description_tokens = 200
    max_gen_token = 32
    model_type = LLAMA3

    def __init__(self, persona_type, decision_making_style, use_persona=False):
        """
        constructor for class Simulator
        :param persona_type: type of the persona
        :param decision_making_style: type of decision making
        """
        self.use_persona = use_persona
        self.persona_type = persona_type
        self.decision_making_style = decision_making_style

    def reformat_dialogue_context(self, dialogue_context):
        """
        method that reformats the given dialogue context
        :param dialogue_context: the given dialogue context, which is in the format of list of dictionaries
        :return: a reformated dialogue context.
        """
        messages = []
        for utt in dialogue_context:
            # switching the role
            if utt['role'] == USER:
                messages.append({"role": ASSISTANT, "content": utt['content']})
            else:
                messages.append({'role': USER, "content": utt['content']})
        return messages

    def respond(self, state):
        """
        method that return a generated response given the dialogue context and persona information
        :param state: the current state of the game
        :return:  None
        """
        raise NotImplementedError("This method must be implemented")

    def generate_persona_description(self):
        """
        method that generate a persona description for the simulator
        :return: a generated persona description
        """
        prompt = f"""
        You need to incorporate the following persona attributes and generate a cohesive persona description.
        You need to ensure the description is easy to understand.        
        ***************************
        Big-Five Personality: {self.persona_type}
        Decision-Making Style: {self.decision_making_style}
        """
        messages = [
            {"role": "system", "content": prompt}
        ]
        output = call_llm(messages, n=1, temperature=self.temperature, max_token=self.max_description_tokens)
        return output[0]

    def set_model_type(self, model_type):
        """
        set the model type can be either llama3 or chatgpt
        :param model_type:
        :return:
        """
        self.model_type = model_type

    def is_using_persona(self, flag):
        """
        set the flag indicating if we're using the persona for the user.
        :param flag:
        :return:
        """
        self.use_persona = flag
