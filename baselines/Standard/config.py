from config.config import ModelConfig
from config.constants import neg_special_tokens_dict, es_special_tokens_dict


class StandardPromptConfig(ModelConfig):
    """
    ProCot is a prompt based method
    therefore it only requires a prompt for planning
    """
    prompt = "Standard"
    max_gen_tokens = 24
    action_mapping = {
        "Standard": "Standard"
    }

    def __init__(self, params):
        """
        constructor for class ProCot config
        :param params: a dictionary that contains parameters and their values
        """
        super().__init__()
        for k, v in params.items():
            setattr(self, k, v)


class StandardPromptConfigForNegotiation(StandardPromptConfig):
    """
    class RTCP config for the negotiation scenario
    """
    special_tokens_dict = neg_special_tokens_dict
    pass


class StandardPromptConfigForEmotionalSupport(StandardPromptConfig):
    """
    class RTCP config for the emotional support conversation
    """
    special_tokens_dict = es_special_tokens_dict
    pass
