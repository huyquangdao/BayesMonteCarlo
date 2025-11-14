import math

from config.config import ModelConfig
from config.constants import rec_special_tokens_dict, neg_special_tokens_dict, es_special_tokens_dict


class MinDistPADPPConfig(ModelConfig):
    # common configurations
    tokenizer = 'roberta-large'
    plm = 'roberta-large'
    lm_size = 1024
    combined_action = True
    run_sft = True
    run_rlt = True
    run_offline_eval = True
    run_online_eval = True
    sampled_times = 10
    gamma = 0.99
    epsilon = 1.0
    num_train_rl_epochs = 100

    # preference and actor-critic training config
    # parameters for training the preference model
    preference_learning_rate = 5e-4
    num_train_preference_epochs = 3
    preference_warmup_steps = 200
    freeze_plm = True
    objective_embedding_size = 6
    reward_hidden_size = 64
    mlp_hidden_size = 128

    # parameters for training the actor critic model
    lambd = 0.97
    clip_eps = 0.2
    coef_ent = 0.2
    max_grad_norm = 5
    actor_learning_rate = 5e-4
    critic_learning_rate = 5e-1
    actor_warmup_steps = 500
    critic_warmup_steps = 500
    n_warmup_epochs = 5

    # number of sampled preferences
    n_preferences = 128
    objective_weight = None
    
    # hyper parameters for controlling the gpi step
    alpha = 0.7
    task_eps = 0.1
    use_gpi = True

    # preference and ppo buffer length
    preference_buffer_length = 512
    ppo_buffer_length = 1000

    def __init__(self, params):
        """
        constructor for class Bert config
        :param params: a dictionary that contains parameters and their values
        """
        super().__init__()
        for k, v in params.items():
            setattr(self, k, v)


class MinDistPADPPConfigForRecommendation(MinDistPADPPConfig):
    """
    MODPL configuration for the recommendation scenario
    """
    combined_action = False
    special_tokens_dict = rec_special_tokens_dict
    learning_rate = 5e-5
    actor_learning_rate = 5e-4
    # objective to weight mapping dict
    obj_to_weight = {
        "uniform": None,
        "user_reward": [1.0, 0.0],
        "item_freq": [0.0, 1.0],
    }
    pass


class MinDistPADPPConfigForNegotiation(MinDistPADPPConfig):
    """
    MODPL configuration for the negotiation scenario
    """
    combined_action = True
    special_tokens_dict = neg_special_tokens_dict
    n_topics = 5
    actor_learning_rate = 5e-4
    # objective to weight mapping dict
    obj_to_weight = {
        "uniform": None,
        "sl_ratio": [1.0, 0.0, 0.0],
        "fairness": [0.0, 1.0, 0.0],
        "sr": [0.0, 0.0, 1.0]
    }
    pass


# class ContextualMODPLConfigForNegotiation(ContextualMODPLConfig):
#     """
#     MODPL configuration for the negotiation scenario
#     """
#     combined_action = True
#     special_tokens_dict = neg_special_tokens_dict
#     n_topics = 5

#     # objective to weight mapping dict
#     obj_to_weight = {
#         "uniform": None,
#         "sl_ratio": [1.0, 0.0],
#         "fairness": [0.0, 1.0],
#         "mid": [0.5, 0.5]
#     }
#     pass



class MinDistPADPPConfigForEmotionalSupport(MinDistPADPPConfig):
    """
    MODPL configuration for the emotional support scenario
    """
    combined_action = False
    special_tokens_dict = es_special_tokens_dict

    # objective to weight mapping dict
    obj_to_weight = {
        "uniform": None,
        "user_reward": [1.0, 0.0, 0.0],
        "toxicity": [0.0, 1.0, 0.0],
        "avg_turn": [0.0, 0.0, 1.0],
    }
    pass
