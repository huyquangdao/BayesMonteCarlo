from config.config import ModelConfig
from config.constants import neg_special_tokens_dict, es_special_tokens_dict, pg_special_tokens_dict, rec_special_tokens_dict


class ICLAIFConfig(ModelConfig):
    """
    ProCot is a prompt based method
    therefore it only requires a prompt for planning
    """
    prompt = ""
    max_gen_tokens = 24

    def __init__(self, params):
        """
        constructor for class ProCot config
        :param params: a dictionary that contains parameters and their values
        """
        super().__init__()
        for k, v in params.items():
            setattr(self, k, v)


class ICLAIFConfigForRecommendation(ICLAIFConfig):
    """
    class ICLAIF config for the recommendation scenario
    """
    special_tokens_dict = rec_special_tokens_dict
    
    # general task instruction
    prompt = """
    Now enter the role-playing mode. In the following conversation, you will play
    as a coach in a recommendation game. There will be a recommender who is trying to recommend a {} to the user.
    The recommender's goal is to proactively lead the conversation with the user towards the target: *{}* 
    Your task is to read the conversation between the user and the recommender, 
    then provide suggestions to the recommender about how to recommend the item to the user.
    """
    
    # chain of thought prompting
    cot_prompt = """
    Read the following conversation between the recommender and the user, then give
    three suggestions to the recommender about how to recommend the item to the user.
    Each suggestion should be only one short and succinct sentence.
    The following is the conversation: {}
    Question: What are your suggestions? Answer:
    """
    action_mapping = ''
    pass


class ICLAIFConfigForNegotiation(ICLAIFConfig):
    """
    class RTCP config for the negotiation scenario
    """
    special_tokens_dict = neg_special_tokens_dict
    # general task instruction
    prompt = """
    Now enter the role-playing mode. In the following conversation, you will play
    as a coach in a bargain game. There will be a buyer and a seller bargaining
    about a product price. Your task is to read the conversation between the buyer
    and the seller, then provide suggestions to the buyer about how to buy the
    product with a lower price.
    """
    # chain of thought prompting
    cot_prompt = """
    Read the following conversation between the buyer and the seller, then give
    three suggestions to the buyer about how to buy the product with a lower price.
    Each suggestion should be only one short and succinct sentence.
    The following is the conversation: {}
    Question: What are your suggestions? Answer:
    """
    action_mapping = ''
    pass


class ICLAIFConfigForEmotionalSupport(ICLAIFConfig):
    """
    class RTCP config for the emotional support conversation
    """
    special_tokens_dict = es_special_tokens_dict
    # general task instruction
    prompt = """
    Now enter the role-playing mode. In the following conversation, you will play
    as a coach in a counselling game. There will be a therapist and a patient talking
    about some emotional issues. Your task is to read the conversation between the
    therapist and the patient, then provide suggestions to the therapist about how to
    help the patient reduce their emotional distress and help them understand and
    work through the challenges.
    """
    # chain of thought prompting
    cot_prompt = """
    Read the following conversation between the therapist and the patient, then give
    three suggestions to the therapist about how to help the patient reduce their emotional distress 
    and help them understand and work through the challenges.
    Each suggestion should be only one short and succinct sentence.
    The following is the conversation: {}
    Question: What are your suggestions? Answer:
    """
    action_mapping = ''
    pass



class ICLAIFConfigForPersuation(ICLAIFConfig):
    """
    class ICLAIF config for persuation conversation
    """
    special_tokens_dict = pg_special_tokens_dict
    
    # general task instruction
    prompt = """
    Now enter the role-playing mode. In the following conversation, you will play
    as a coach in a charity donation game. There will be a persuader and a persuadee talking
    about charity donation. Your task is to read the conversation between the
    persuader and the persuadee, then provide suggestions to the persuader about how to
    convince the persuadee to donate for charity.
    """
    
    # chain of thought prompting
    cot_prompt = """
    Read the following conversation between the persuader and the persuadee, then give
    three suggestions to the therapist about how to convince the persuadee to donate for charity.
    Each suggestion should be only one short and succinct sentence.
    The following is the conversation: {}
    Question: What are your suggestions? Answer:
    """
    
    action_mapping = ''
    pass
