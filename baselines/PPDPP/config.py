from config.config import ModelConfig
from config.constants import rec_special_tokens_dict, neg_special_tokens_dict, es_special_tokens_dict, pg_special_tokens_dict


class PPDPPConfig(ModelConfig):
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
    num_train_rl_epochs = 10
    rl_learning_rate = 1e-5
    uniform_weights = False
    eval_interval = 10
    
    # meta prompt path
    # need to be initialized later
    meta_prompt_path = ""
    meta_prompt = ""

    def __init__(self, params):
        """
        constructor for class Bert config
        :param params: a dictionary that contains parameters and their values
        """
        super().__init__()
        for k, v in params.items():
            setattr(self, k, v)
    
    def read_meta_prompt(self):
        with open(self.meta_prompt_path, "r") as f:
            self.meta_prompt = f.read().strip()


class PPDPPConfigForRecommendation(PPDPPConfig):
    combined_action = False
    special_tokens_dict = rec_special_tokens_dict
    objective_weight = [1.0, 0]
    """
    Class PPDPP configuration for recommendation scenario.
    """
    obj_to_weight = {
        "uniform": None,
        "user_reward": [1.0, 0.0],
        "item_freq": [0.0, 1.0],
    }
    
    # chain of thought prompting
    rewrite_prompt = """
    Assume you are the expert analyst. Given the conversation history and a naive action instruction, 
    in order to recommend an target item to the user successfully, please only add neccesary information to the following action instruction. 
    Do not modify the action instruction entirely !!
    You answer should be in the following format: "Answer: X"
    """
    
    # filling the meta prompt into this template
    rewrite_prompt_cot = """
    {}
    Do not modify the action instruction entirely !!
    Do not modify the action instruction entirely !!
    The following is the conversation history: {}
    Here is the naive action instruction: **{}**.
    Question: What is the rewritten action instruction ? Answer:
    """
    pass

# class PPDPPConfigForRecommendation(PPDPPConfig):
#     combined_action = False
#     special_tokens_dict = rec_special_tokens_dict
#     objective_weight = [1.0, 0]
#     """
#     Class PPDPP configuration for recommendation scenario.
#     """
#     obj_to_weight = {
#         "uniform": None,
#         "user_reward": [1.0, 0.0, 0.0],
#         "item_freq": [0.0, 1.0, 1.0],
#         "avg_turn": [0.0, 0.0, 1.0],
#     }
#     pass

class PPDPPConfigForNegotiation(PPDPPConfig):
    combined_action = True
    special_tokens_dict = neg_special_tokens_dict
    n_topics = 5
    """
    class PPDPP configuration for negotiation scenario.
    """
    # objective_weights = [1.0, 1.0]
    # objective weight for price_gain, fairness, sr
    objective_weight = [1.0, 0, 0]
    obj_to_weight = {
        "uniform": None,
        "sl_ratio": [1.0, 0.0, 0.0],
        "fairness": [0.0, 1.0, 0.0],
        "sr": [0.0, 0.0, 1.0]
    }
    
    # chain of thought prompting
    rewrite_prompt = """
    Assume you are the expert analyst. Given the conversation history and a naive action instruction, 
    in order to reach a better deal with the seller, please revise the following action instruction appropriately.
    Do not modify the action instruction entirely.
    You answer should be in the following format: "Answer: X"
    """
    
    rewrite_prompt_cot = """
    {}
    Do not modify the action instruction entirely.
    The following is the conversation history: {}
    Here is the naive action instruction: {}
    Question: What is the rewritten action instruction ? Answer:
    """
    pass


class PPDPPConfigForEmotionalSupport(PPDPPConfig):
    combined_action = False
    special_tokens_dict = es_special_tokens_dict
    
    # chain of thought prompting
    rewrite_prompt = """
    Assume you are the expert analyst. Given the conversation history and a naive action instruction, 
    in order to in order to help the patient reduce their emotional distress and help them understand and work
    through the challenges, please revise the following action instruction appropriately.
    Do not modify the action instruction entirely.
    You answer should be in the following format: "Answer: X"
    """
    
    rewrite_prompt_cot = """
    {}
    Do not modify the action instruction entirely.
    The following is the conversation history: {}
    Here is the naive action instruction: {}
    Question: What are the rewritten action instruction ? Answer:
    """
    
    """
    class PPDPP configuration for emotional support scenario
    """
    pass


class PPDPPConfigForPersuation(PPDPPConfig):
    combined_action = False
    special_tokens_dict = pg_special_tokens_dict
    
    # chain of thought prompting
    rewrite_prompt = """
    Assume you are the expert analyst. Given the conversation history and a naive action instruction, 
    in order to convince the persuadee to donate for charity, please revise the following naive action instruction appropriately.
    Do not modify the action instruction entirely.
    You answer should be in the following format: "Answer: X"
    """
    
    rewrite_prompt_cot = """
    {}
    Do not modify the action instruction entirely.
    The following is the conversation history: {}
    Here is the naive action instruction: {}
    Question: What are the rewritten action instruction ? Answer:
    """
    
    """
    class PPDPP configuration for persuation conversation
    """
    pass


class PreferencePPDPPConfig(PPDPPConfig):
    preference_learning_rate = 1e-4
    preference_batch_size = 2
    num_train_preference_epochs = 3
    num_preference_warmup_steps = 200
    freeze_preference_backbone = True
    n_alternative_iterations = 5
    objective_embedding_size = 6
    reward_hidden_size = 64
    lambd = 0.97
    clip_eps = 0.2
    coef_ent = 0.2
    max_grad_norm = 5
    actor_learning_rate = 1e-5
    critic_learning_rate = 1e-4
