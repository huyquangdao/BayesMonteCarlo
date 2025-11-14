from config.config import ModelConfig
from config.constants import rec_special_tokens_dict


class UNIMINDConfig(ModelConfig):
    # general configurations of UNIMIND
    tokenizer = 'facebook/bart-base'
    plm = 'facebook/bart-base'
    lm_size = 768
    run_sft = True
    run_offline_eval = True
    do_pretrain = True
    do_finetune = True
    max_target_length = 100
    max_gen_length = 50
    num_finetune_epochs = 2
    tasks = ['goal', 'topic', 'response']

    def __init__(self, params):
        """
        constructor for class RTCP config
        :param params: a dictionary that contains parameters and their values
        """
        super().__init__()
        for k, v in params.items():
            setattr(self, k, v)


class UNIMINDConfigForRecommendation(UNIMINDConfig):
    # general configurations of UNIMIND for the recommendation scenario
    special_tokens_dict = rec_special_tokens_dict
    pass
