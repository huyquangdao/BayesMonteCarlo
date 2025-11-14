from config.config import ModelConfig
from config.constants import rec_special_tokens_dict, neg_special_tokens_dict, es_special_tokens_dict


class RTCPConfig(ModelConfig):
    tokenizer = 'bert-large-cased'
    plm = 'bert-large-cased'
    lm_size = 1024
    combined_action = False
    run_sft = True
    run_offline_eval = True

    # other configurations
    n_layers = 6
    n_heads = 8
    fc_hidden_size = 128
    ffn_size = 128
    attention_dropout = 0.2
    relu_dropout = 0.2

    def __init__(self, params):
        """
        constructor for class RTCP config
        :param params: a dictionary that contains parameters and their values
        """
        super().__init__()
        for k, v in params.items():
            setattr(self, k, v)


class RTCPConfigForRecommendation(RTCPConfig):
    """
    class RTCP config for the recommendation scenario
    """
    special_tokens_dict = rec_special_tokens_dict
    pass


class RTCPConfigForNegotiation(RTCPConfig):
    """
    class RTCP config for the negotiation scenario
    """
    special_tokens_dict = neg_special_tokens_dict
    pass


class RTCPConfigForEmotionalSupport(RTCPConfig):
    """
    class RTCP config for the emotional support conversation
    """
    special_tokens_dict = es_special_tokens_dict
    pass
