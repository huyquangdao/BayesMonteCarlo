from config.config import ModelConfig
from config.constants import rec_special_tokens_dict, neg_special_tokens_dict, es_special_tokens_dict, pg_special_tokens_dict, rec_special_tokens_dict


class AnEConfig(ModelConfig):
    """
    ProCot is a prompt based method
    therefore it only requires a prompt for planning
    """
    prompt = ""
    max_gen_tokens = 24
    model_type = "chatgpt"

    def __init__(self, params):
        """
        constructor for class ProCot config
        :param params: a dictionary that contains parameters and their values
        """
        super().__init__()
        for k, v in params.items():
            setattr(self, k, v)


class AnEConfigForRecommendation(AnEConfig):
    """
    class ANE config for the recommendation scenario
    """
    special_tokens_dict = rec_special_tokens_dict
    
    # general task instruction
    prompt = """
    Assume you are a recommender who is trying to recommend a {} to an user. 
    Your goal is to proactively lead the conversation with the user towards the target: *{}*.
    Given the conversation history, answer the following questions. Please answer with only one
    short and succinct sentence.
    """
    
    # AnE prompting
    q1_prompt = """
    The following is the conversation history: {}
    Question: How did the user feel? Answer:
    """
    q2_prompt = """
    The following is the conversation history: {}
    Question: Why did the user feel that way? Answer:
    """
    q3_prompt = """
    The following is the conversation history: {}
    Question: What should the recommender do? Answer:
    """
    pass


class AnEConfigForNegotiation(AnEConfig):
    """
    class ANE config for the negotiation scenario
    """
    special_tokens_dict = neg_special_tokens_dict
    # general task instruction
    prompt = """
    Assume you are the bargain expert to reach a better deal with the seller. Given
    the conversation history, answer the question. Please answer with only one
    short and succinct sentence.
    """
    # AnE prompting
    q1_prompt = """
    The following is the conversation history: {}
    Question: How did the seller feel? Answer:
    """
    q2_prompt = """
    The following is the conversation history: {}
    Question: Why did the seller feel that way? Answer:
    """
    q3_prompt = """
    The following is the conversation history: {}
    Question: What should the buyer do? Answer:
    """
    pass


class AnEConfigForEmotionalSupport(AnEConfig):
    """
    class RTCP config for the emotional support conversation
    """
    special_tokens_dict = es_special_tokens_dict
    # general task instruction
    prompt = """
    Assume you are a therapist expert to help the patient reduce their emotional
    distress and help them understand and work through the challenges. Given the
    conversation history, answer the question. Please answer with only one short
    and succinct sentence.
    """
    # AnE prompting
    q1_prompt = """
    The following is the conversation history: {}
    Question: How did the patient feel? Answer:
    """
    q2_prompt = """
    The following is the conversation history: {}
    Question: Why did the patient feel that way? Answer:
    """
    q3_prompt = """
    The following is the conversation history: {}
    Question: What should the therapist do? Answer:
    """
    pass


class AnEConfigForPersuation(AnEConfig):
    """
    class RTCP config for the emotional support conversation
    """
    special_tokens_dict = pg_special_tokens_dict
    # general task instruction
    prompt = """
    Assume you are the persuader who is trying to convince the persuadee to
    donate for charity. Given the conversation history, answer the question. 
    Please answer with only one short and succinct sentence.
    """
    # AnE prompting
    q1_prompt = """
    The following is the conversation history: {}
    Question: How did the persuadee feel? Answer:
    """
    q2_prompt = """
    The following is the conversation history: {}
    Question: Why did the persuadee feel that way? Answer:
    """
    q3_prompt = """
    The following is the conversation history: {}
    Question: What should the persuader do? Answer:
    """
    pass
