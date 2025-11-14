from config.config import ModelConfig
from config.constants import rec_special_tokens_dict, neg_special_tokens_dict, es_special_tokens_dict


class GDPZeroConfig(ModelConfig):
    """
    ProCot is a prompt based method
    therefore it only requires a prompt for planning
    """
    prompt = ""
    max_gen_tokens = 24
    max_realizations = 3
    rollouts = 3
    combined_action = False
    Q_0 = 0.0
    cpuct = 1.0
    temperature = 0.0

    def __init__(self, params):
        """
        constructor for class RTCP config
        :param params: a dictionary that contains parameters and their values
        """
        super().__init__()
        for k, v in params.items():
            setattr(self, k, v)


class GDPZeroConfigForRecommendation(GDPZeroConfig):
    """
    class RTCP config for the recommendation scenario
    """
    objective_weight = [0.5, 0.5]
    prompt = ""
    special_tokens_dict = rec_special_tokens_dict

    domain2strategies = {
        "all": """
        "Greetings". "Ask about weather". "Play music". "Music recommendation".
        "Question & Answer". "Chat about stars". "Music on demand". "Movie recommendation". 
        "Say goodbye". "Ask about date". "Ask questions", "POI recommendation", "Food recommendation".
        """,
        "movie": """
        "Question & Answer". "Movie recommendation". "Say goodbye". "Ask about date". "Chat about stars". 
        "Music recommendation". "Ask questions". "Greetings".
        """,
        "music": """
        "Movie recommendation". "Music on demand". "Ask about weather". "Ask questions". "Music recommendation". 
        "Ask about date". "Play music". "Chat about stars". "Question & Answer". "Say goodbye". "Greetings".
        """,
        "food": """
        "Food recommendation". "Say goodbye". "Greetings". "Ask about weather".
        """,
        "poi": """
        "Food recommendation". "Ask about weather". "Greetings". "Say goodbye". "POI recommendation".
        """
    }

    # general task instruction
    prompt = """
    Assume you are the recommender system. Given the conversation history, in order to reach a successful recommendation,
    please select the most appropriate dialogue strategy.
    """
    # chain of thought prompting
    cot_prompt = """
    You can only reply by selecting one of the following dialogue strategy to reach
    the goal: {}
    The following is the conversation history: {}
    Question: Which one is the most appropriate dialogue strategy ? Answer:
    """
    special_tokens_dict = neg_special_tokens_dict
    action_mapping = {
        "Greetings": "Greetings",
        "Ask about weather": "Ask about weather",
        "Play music": "Play music",
        "Music recommendation": "Music recommendation",
        "Question & Answer": "Q&A",
        "Chat about stars": "Chat about stars",
        "Music on demand": "Music on demand",
        "Movie recommendation": "Movie recommendation",
        "Say goodbye": "Say goodbye",
        "Ask about date": "Ask about date",
        "Ask questions": "Ask questions",
        "POI recommendation": "POI recommendation",
        "Food recommendation": "Food recommendation"
    }
    pass

class GDPZeroConfigForNegotiation(GDPZeroConfig):
    """
    class RTCP config for the negotiation scenario
    """
    objective_weight = [0.5, 0.5, 0.5]
    n_topics = 5

    # general task instruction
    prompt = """
    Assume you are the buyer. Given the conversation history, in order to reach a
    better deal with the seller, please select the most appropriate dialogue strategy 
    and an appropriate price.
    """
    # chain of thought prompting
    cot_prompt = """
    You can only reply by selecting one of the following dialogue strategy to reach
    the goal: "Greetings". "Ask a question". "Answer a question". "Propose the first price".
    "Propose a counter price". "Use comparatives". "Confirm information". "Affirm
    confirmation". "Deny confirmation". "Agree with the proposal". "Disagree with
    the proposal".
    The following is the conversation history: {}
    Question: Which one is the most appropriate dialogue strategy and price ? Answer:
    """
    special_tokens_dict = neg_special_tokens_dict
    action_mapping = {
        "Greetings": "greet",
        "Ask a question": "inquire",
        "Answer a question": "inform",
        "Propose the first price": "propose",
        "Propose a counter price": "counter",
        "Use comparatives": "counter-noprice",
        "Confirm information": "confirm",
        "Affirm confirmation": "affirm",
        "Deny confirmation": "deny",
        "Agree with the proposal": "agree",
        "Disagree with the proposal": "disagree"
    }
    pass


class GDPZeroConfigForEmotionalSupport(GDPZeroConfig):
    """
    class RTCP config for the emotional support conversation
    """
    # general task instruction
    prompt = """
    Assume you are the therapist. Given the conversation history, in order to help
    the patient reduce their emotional distress and help them understand and work
    through the challenges, please select the most appropriate dialogue strategy.
    """
    # chain of thought prompting
    cot_prompt = """
    You can only reply by selecting one of the following dialogue strategy to reach
    the goal: "Question". "Self-disclosure". "Affirmation and Reassurance". "Providing
    Suggestions". "Reflection of feelings". "Information". "Restatement or Paraphrasing".
    The following is the conversation history: {}
    Question: Which one is the most appropriate dialogue strategy? Answer:
    """
    special_tokens_dict = es_special_tokens_dict
    action_mapping = {
        "Question": "Question",
        "Self-disclosure": "Self-disclosure",
        "Affirmation and Reassurance": "Affirmation and Reassurance",
        "Providing Suggestions": "Providing Suggestions",
        "Reflection of feelings": "Reflection of feelings",
        "Information": "Information",
        "Restatement or Paraphrasing": "Restatement or Paraphrasing",
    }
    pass
