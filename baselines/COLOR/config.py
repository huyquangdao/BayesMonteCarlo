from config.config import ModelConfig
from config.constants import rec_special_tokens_dict


class COLORConfig(ModelConfig):
    # general configurations of UNIMIND
    tokenizer = 'facebook/bart-base'
    plm = 'facebook/bart-base'
    lm_size = 768
    run_sft = True
    run_offline_eval = True
    max_target_length = 100
    max_gen_length = 50
    max_grad_norm = 5

    # parameters for training the brownian bridge
    max_transition_number = 11
    latent_dim = 16
    use_simulated = True
    freeze_plm = True
    eval_brownian_bridge = True
    bridge_learning_rate = 2e-4

    # additional parameters ofr training the color planner.
    train_use_bridge = True
    trans_alpha = 0.1
    gen_beta = 1.0
    kl_gamma = 1.0
    planner_learning_rate = 2e-5

    # inference parameters
    infer_use_bridge = True
    dataset = 'durecdial'
    max_dec_len = 30
    min_length = 1
    repetition_penalty = 1.0
    diversity_penalty = 0.0
    no_repeat_ngram_size = 0
    bad_words_ids = None
    remove_invalid_values = False

    #
    use_transform = False
    use_KLD = False

    def __init__(self, params):
        """
        constructor for class RTCP config
        :param params: a dictionary that contains parameters and their values
        """
        super().__init__()
        for k, v in params.items():
            setattr(self, k, v)


class COLORConfigForRecommendation(COLORConfig):
    # general configurations of UNIMIND for the recommendation scenario
    special_tokens_dict = rec_special_tokens_dict
    pass
