from config.config import ModelConfig
from config.constants import rec_special_tokens_dict, neg_special_tokens_dict, es_special_tokens_dict
from baselines.GDP_Zero.config import GDPZeroConfigForNegotiation, GDPZeroConfigForRecommendation

class DPDPConfig(ModelConfig):
    # general parameters
    tokenizer = 'roberta-large'
    plm = 'roberta-large'
    lm_size = 1024
    combined_action = True
    run_sft = True
    run_rlt = True
    run_offline_eval = True
    run_online_eval = True
    sampled_times = 10
    gamma = 0.999
    epsilon = 1.0

    num_train_rl_epochs = 100
    rl_learning_rate = 5e-6
    uniform_weights = False

    # parameters for offline reinforcement learning
    hidden_size = 128
    eval_interval = 1

    # parameters for mcts self-play training
    entropy_bound = 0.5
    sub_value = 0.0
    success_base = 0
    lmbda = 0.0
    critic_loss_w = 1.0
    sub_value = 0.5

    # only set to True once.
    preprocess_data_for_offline_rl = True

    # Alway set to True.
    load_processed_data_for_offline_rl = True

    def __init__(self, params):
        """
        constructor for class Bert config
        :param params: a dictionary that contains parameters and their values
        """
        super().__init__()
        for k, v in params.items():
            setattr(self, k, v)


class DPDPConfigForRecommendation(DPDPConfig):
    combined_action = False
    special_tokens_dict = rec_special_tokens_dict
    objective_weight = [1.0, 1.0]
    mcts_config = GDPZeroConfigForRecommendation({})
    """
    Class PPDPP configuration for recommendation scenario.
    """
    obj_to_weight = {
        "uniform": [0.5, 0.5],
        "user_reward": [1.0, 0.0],
        "item_freq": [0.0, 1.0],
    }
    pass


class DPDPConfigForNegotiation(DPDPConfig):
    combined_action = True
    special_tokens_dict = neg_special_tokens_dict
    n_topics = 5
    mcts_config = GDPZeroConfigForNegotiation({})
    """
    class PPDPP configuration for negotiation scenario.
    """
    # objective_weights = [1.0, 1.0]
    # objective weight for price_gain, fairness, sr
    objective_weight = [1.0, 0, 0]
    obj_to_weight = {
        "uniform": [0.5, 0.5, 0.5],
        "sl_ratio": [1.0, 0.0, 0.0],
        "fairness": [0.0, 1.0, 0.0],
        "sr": [0.0, 0.0, 1.0]
    }
    pass


class DPDPConfigForEmotionalSupport(DPDPConfig):
    combined_action = False
    special_tokens_dict = es_special_tokens_dict
    """
    class PPDPP configuration for emotional support scenario
    """
    pass
