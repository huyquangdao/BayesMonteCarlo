import time

from base.simulator import Simulator
from utils.prompt import call_llm
from config.constants import CHATGPT


class PersuationSimulator(Simulator):

    def __init__(self, user_profile, use_persona=False):
        """
        constructor for class persuation simulator
        :param user_profile: a tuple of big5 persona and decision making style
        """
        self.use_persona = use_persona
        # generating the profile description
        self.user_profile_description = self.generate_persona_description(user_profile)

    def respond(self, state, **kwargs):
        """
        method that generates the user response for the negotiation scenario
        :param state: the current state of the conversation
        :return: the generated response by the user.
        """
        dialogue_context = state['dialogue_context']
        # if we employ the persona for the user simulator
        # print(self.user_profile_description)
        if self.use_persona:
            prompt = f"""
            Now enter the role-playing mode. In the following conversation, you will play as a Persuadee in a
            persuasion game.
            Your persona: {self.user_profile_description}.
            You must follow the instructions below during chat.
            1. Your utterances need to strictly follow your persona. Varying your wording and avoid repeating
            yourself verbatim. Consider your values, emotions, priorities and practical constraints (e.g., time, finances, trust).
            2. Pretend you have little knowledge about the Save the Children charity. You have little willingness
            for donation at the beginning of conversation.
            Do not remain permanently undecided if your main questions and doubts have been reasonably addressed.
            3. Your willingness to donate depends on:
            - how well the Persuader’s arguments match your values and priorities,
            - how clear and trustworthy the information is,
            - whether donating feels feasible and appropriate for you.
            If you genuinely feel convinced and able to help, you may decide to donate an amount that makes sense for you.

            4. Over the course of the conversation, you should think carefully and respond in a balanced way:
            - Sometimes you may ask for more information or express doubts.
            - Sometimes you may acknowledge good points or move closer to a decision.
            - Eventually, you should choose a clear stance (either donate, postpone, or refuse) that is consistent with your persona.

            Your Response Strategy:
            1. "Donate": show your willingness to donate.
            2. "Source Derogation": attacks or doubts the organisation’s credibility.
            3. "Counter Argument": argues that the responsibility is not on them or refutes a previous statement.
            4. "Personal Choice": Attempts to saves face by asserting their personal preference such as their choice
            of charity and their choice of donation.
            5. "Information Inquiry": Ask for factual information about the organisation for clarification or as an
            attempt to stall.
            6. "Self Pity": Provides a self-centred reason for not being willing to donate at the moment.
            7. "Hesitance": Attempts to stall the conversation by either stating they would donate later or is
            currently unsure about donating.
            8. "Self-assertion": Explicitly refuses to donate without even providing a personal reason.
            9. "Others": Please respond naturally when no specific persuasion strategy applies.
            Very important:
            - First, decide which strategy best describes your intention according to the rules above.
            - Then, write ONE short sentence that clearly shows that strategy.
            - The chosen Strategy MUST be consistent with the meaning of your sentence.
            - Prefer strategies 1–8 whenever possible. Use "Others" only as a last resort when none of the other 8 strategies apply.
            - Do NOT output labels like "Donate:", "Hesitance:" or any other strategy name in your response; just speak naturally as the Persuadee.

            You are the Persuadee who is being persuaded by a Persuader.
            The conversation history is as bellow:
            """
        # we ignore the persona for the user simulator
        else:
            prompt = f"""
            Now enter the role-playing mode. In the following conversation, you will play as a Persuadee in a
            persuasion game.
   
            The conversation history is as bellow:
            """
        # construct the system instruction prompt
        messages = [
            {"role": "system", "content": prompt}
        ]

        # reformating and prepending the dialogue context to the current prompt
        messages.extend(self.reformat_dialogue_context(dialogue_context))
        # debug: print the constructed prompt
        # messages.append(
        #     {'role': 'user', 'content': f"""
        #         Reply with exactly one short sentence formatted as "[Strategy] - [Response]".
        #         Strategy must be one of: Donate, Source Derogation, Counter Argument, Personal Choice,
        #         Information Inquiry, Self Pity, Hesitance, Self-assertion, Others.
        #         Do not add any extra text before or after the format.
        #     """
        #      }
        # )
        
        # print the prompt used for simulator response
        # self.log_prompt(messages, prefix="PG_USER_SIM_PROMPT")
        
        # print(messages)
                        
        # messages.extend(dialogue_context)
        t = time.time()

        # calling the llm for response generation
        response = call_llm(messages, 
                            n=1, 
                            temperature=0.2, 
                            max_token=self.max_gen_token, 
                            model_type=self.model_type,
                            **kwargs
                            )
                        
        # print("Simulator Generation Time: ", time.time() - t)
        return response[0]

    def generate_persona_description(self, user_profile):
        """
        method that generate a persona description given the big5 personality and decision making style (persona, decision_type)
        :return: an user profile description
        """
        assert len(user_profile) == 2
        prompt = f"""
        You need to incorporate the following user profile and generate a cohesive persona description.
        You need to ensure the description is easy to understand.
        ********
        Big-Five Personality: {user_profile[0]}
        Decision-Making Style: {user_profile[1]}
        ********
        """
        messages = [
            {"role": "system", "content": prompt}
        ]
        # generate the user profiles with chat gpt
        output = call_llm(messages, n=1, 
                          temperature=self.temperature, 
                          max_token=self.max_description_tokens,
                          model_type=CHATGPT)
        
        # return the user persona description
        return output[0]
