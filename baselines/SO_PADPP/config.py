import math

from config.config import ModelConfig
from config.constants import rec_special_tokens_dict, neg_special_tokens_dict, es_special_tokens_dict, pg_special_tokens_dict


class SOPADPPConfig(ModelConfig):
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
    freeze_plm = False
    reward_hidden_size = 64
    mlp_hidden_size = 128
    actor_warmup_steps = 500

    # parameters for training the actor critic model
    lambd = 0.97
    clip_eps = 0.2
    coef_ent = 0.2
    max_grad_norm = 5
    actor_learning_rate = 5e-4
    n_warmup_epochs = 5

    # number of sampled preferences
    n_preferences = 64
    objective_weight = None
    eval_interval = 10
    
    # hyper parameters for controlling the gpi step
    alpha = 0.8
    task_eps = 0.1
    use_gpi = True

    # preference and ppo buffer length
    ppo_buffer_length = 1000
    train_rl_batch_size = 32

    def __init__(self, params):
        """
        constructor for class Bert config
        :param params: a dictionary that contains parameters and their values
        """
        super().__init__()
        for k, v in params.items():
            setattr(self, k, v)


class SOPADPPConfigForRecommendation(SOPADPPConfig):
    """
    MODPL configuration for the recommendation scenario
    """
    combined_action = False
    special_tokens_dict = rec_special_tokens_dict
    learning_rate = 5e-5
    actor_learning_rate = 5e-4
    pass


class SOPADPPConfigForNegotiation(SOPADPPConfig):
    """
    MODPL configuration for the negotiation scenario
    """
    combined_action = False
    special_tokens_dict = neg_special_tokens_dict
    n_topics = 5
    actor_learning_rate = 5e-3
    pass


class SOPADPPConfigForEmotionalSupport(SOPADPPConfig):
    """
    MODPL configuration for the emotional support scenario
    """
    combined_action = False
    special_tokens_dict = es_special_tokens_dict
    pass


class SOPADPPConfigForPersuation(SOPADPPConfig):
    """
    MODPL configuration for the persuation scenario
    """
    combined_action = False
    special_tokens_dict = pg_special_tokens_dict
    pass
