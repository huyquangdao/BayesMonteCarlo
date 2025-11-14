from base.simulator import Simulator
from utils.prompt import call_llm
from config.constants import DURECDIAL, INSPIRED


class RecommendationSimulator(Simulator):

    def __init__(self, user_profile, use_persona=False):
        """
        constructor for class Recommendation Simulator
        :param user_profile: the profile of the user, in the format of a dictionary
        """
        # generating the profile description
        self.user_profile = user_profile
        self.use_persona = use_persona
        # self.user_profile_description = ''
        self.user_profile_description = self.generate_persona_description(user_profile)

    def respond(self, state, dataset='durecdial', **kwargs):
        """
        method that generate a response given the user profile description and the dialogue context
        :param state: the current dialogue context in the format of list of dictionary
        :return: a generated response in the format of string
        """
        dialogue_context = state['dialogue_context']

        # simulator for the durecdial dataset
        if dataset == DURECDIAL:
            domain = state['task_background']['target_goal']
            if 'POI' in domain:
                domain = 'Restaurant recommendation'

        # simulator for the inspired dataset
        elif dataset == INSPIRED:
            domain = 'Movie recommendation'

        # if using the user persona
        if self.use_persona:
            prompt = f"""
            Now enter the role-playing mode. In the following conversation, you will play as a User in a
            recommendation game. You are looking for a {domain}.
            Your persona: {self.user_profile_description}
            1. Your utterances and preferences need to strictly follow your persona. Varying your wording
            and avoid repeating yourself verbatim!
            2. You can decide to change your preferences flexibly based on your persona and the conversation.
            Please reply with only one short and succinct sentence.
            """
        # not using the user persona
        else:
            prompt = f"""
            Now enter the role-playing mode. In the following conversation, you will play as a user in a
            recommendation game. You are looking for a {domain}.
            You must follow the instructions below during chat.
            1. You are the user who is being convinced by a recommender.
            2. The recommender is trying to recommend a specific item to you. 
            Please reply with only one short and succinct sentence.
            """

        # construct the system instruction prompt
        messages = [
            {"role": "system", "content": prompt}
        ]

        # reformating and prepending the dialogue context to the current prompt
        messages.extend(self.reformat_dialogue_context(dialogue_context))
        messages.append(
            {'role': 'user', 'content': 'Please reply with only one short and succinct sentence.'}
        )
        
        # calling the llm for response generation
        response = call_llm(messages,
                            n=1, 
                            temperature=0.6, 
                            max_token=self.max_gen_token, 
                            model_type=self.model_type,
                            **kwargs
                            )
        return response[0]

    def convert_profile_to_string(self, user_profile):
        """
        method that convert the user profile dictionary to a string format
        :return: a string format of the user profile
        """
        user_profile_description = ""
        for k, v in user_profile.items():
            if "Name" in k or "Accepted" in k or "Rejected" in k:
                user_profile_description += f"{k}: {v} \n"
        return user_profile_description

    def generate_persona_description(self, user_profile):
        """
        method that generate a profile description given the user profile dictionary
        :return: an user profile description
        """
        prompt = f"""
        You need to incorporate the following user profile and generate a cohesive profile description.
        You need to ensure the description is easy to understand.
        ********
        {self.convert_profile_to_string(user_profile)}
        ********
        """
        messages = [
            {"role": "system", "content": prompt}
        ]
        output = call_llm(messages, n=1, temperature=self.temperature, max_token=self.max_description_tokens,
                          model_type=self.model_type)
        # return the user persona description
        return output[0]
