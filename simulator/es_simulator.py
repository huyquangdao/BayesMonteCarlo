import time

from base.simulator import Simulator
from utils.prompt import call_llm


class EmotionalSupportSimulator(Simulator):

    def __init__(self, user_profile, use_persona=False):
        """
        constructor for class emotional support simulator
        :param user_profile:
        """
        self.use_persona = use_persona
        self.user_profile_description = self.generate_persona_description(user_profile)

    def respond(self, state, **kwargs):
        """
        method that generates the simulated user response from the seeker in emotional support conversation
        :param state: the current state of the conversation
        :return:
        """
        dialogue_context = state['dialogue_context']
        problem_type = state['task_background']['problem_type']
        emotion_type = state['task_background']['emotion_type']
        situation = state['task_background']['situation']

        # using the persona description for simulator
        if self.use_persona:
            prompt = f"""
                Now enter the role-playing mode. In the following conversation, you will play as a patient in a emotional support
                 game.
                Your persona: {self.user_profile_description}
                You must follow the instructions below during chat.
                1. Your utterances and emotional attitude need to strictly follow your persona. Varying your wording
                and avoid repeating yourself verbatim!
                2. You can decide to change your emotional attitude flexibly based on your persona and the conversation.
                Please reply with only one short and succinct sentence. Following is the conversation history:
                """
        # do not use the persona description
        else:
            prompt = f"""
                Now enter the role-playing mode. In the following conversation, you will play
                as a patient in a counselling conversation with a therapist.
                """

        # construct the system instruction prompt
        messages = [
            {"role": "system", "content": prompt},
            {"role": "user", "content": f"""
             You are the patient who is looking for the help from the therapist, because you have the emotional issue about {emotion_type} regarding
            {problem_type}. Please reply with only one short and succinct sentence. Now tell me your issue.
             """}
        ]

        # reformating and prepending the dialogue context to the current prompt
        messages.extend(self.reformat_dialogue_context(dialogue_context))

        t = time.time()
        
        # calling the llm for response generation
        response = call_llm(messages, 
                            n=1, 
                            temperature=0.7, 
                            max_token=self.max_gen_token, 
                            model_type=self.model_type,
                            **kwargs
                            )
        
        print("Simulator Generation Time: ", time.time() - t)
        # print(response)
        return response[0]

    def generate_persona_description(self, user_profile):
        """
        method that generate a persona description given the situation of the patient (problem_type, emotion_type, situation)
        :return: an user profile description
        """
        prompt = f"""
        You need to incorporate the following information and generate a cohesive persona description.
        You need to ensure the description is easy to understand.
        ********
        Problem: {user_profile['problem_type']}
        Situation: {user_profile['situation']}
        Emotion: {user_profile['emotion_type']}
        ********
        """
        messages = [
            {"role": "system", "content": prompt}
        ]
        output = call_llm(messages, n=1,
                          temperature=self.temperature,
                          max_token=self.max_description_tokens,
                          model_type=self.model_type
                          )
        
        # return the user persona description
        return output[0]
