"""
Bayes-Adaptive LLM configuration skeleton.
This mirrors the structure of `baselines/TRIP/config.py` so we can plug in
Bayesian preference search and DPO/MCTS training without refactoring other code.
"""

from config.config import ModelConfig
from config.constants import rec_special_tokens_dict, neg_special_tokens_dict, es_special_tokens_dict, \
    pg_special_tokens_dict


class BayesAdaptiveConfig(ModelConfig):
    """
    Base config for the Bayes-Adaptive LLM pipeline.
    Defaults closely follow TRIP so trainers/pipelines can be reused.
    """

    # backbone
    tokenizer = "roberta-large"
    plm = "roberta-large"
    lm_size = 1024
    combined_action = True
    dropout = 0.1

    # training flow toggles
    run_sft = True
    run_preference_search = False  # generate preference pairs via MCTS loop
    run_dpo = False
    run_offline_eval = False
    run_online_eval = False

    # optimisation
    batch_size = 4
    gradient_accumulation = 1
    learning_rate = 5e-5
    warmup_ratio = 0.05
    weight_decay = 0.01
    num_train_epochs = 3
    logging_steps = 10
    save_total_limit = 2
    max_grad_norm = 1.0

    # inference / prompting
    temperature = 0.7
    max_gen_tokens = 256
    max_sequence_length = 512

    # DPO-related knobs
    dpo_beta = 0.1
    max_length = None
    max_prompt_length = None
    fp16 = False
    bf16 = False
    gradient_checkpointing = False
    reference_model = None
    output_dir = None


    top_k_preferences = 1  

    def __init__(self, params):
        super().__init__()
        for k, v in params.items():
            setattr(self, k, v)


class BayesAdaptiveConfigForRecommendation(BayesAdaptiveConfig):
    combined_action = False
    special_tokens_dict = rec_special_tokens_dict
    # recommendation: predict goal only
    n_goals = 14
    n_topics = 1


class BayesAdaptiveConfigForNegotiation(BayesAdaptiveConfig):
    combined_action = True
    special_tokens_dict = neg_special_tokens_dict
    n_goals = 14
    n_topics = 5


class BayesAdaptiveConfigForEmotionalSupport(BayesAdaptiveConfig):
    combined_action = False
    special_tokens_dict = es_special_tokens_dict
    n_goals = 14
    n_topics = 1


class BayesAdaptiveConfigForPersuation(BayesAdaptiveConfig):
    # MCTS loop for preference pair generation
    mcts_num_evaluate = 1
    num_mcts_sims = 15
    max_realizations = 3
    max_turns = 12
    combined_action = False
    special_tokens_dict = pg_special_tokens_dict
    temperature = 0.000001
    # prompt for user-aware strategic planning for persuation
    prompt = """
    Context:
    - Save the Children is a global charity that provides safety, nutrition, education, and emergency relief for vulnerable children.
    - Donations of any size (as little as $1 or $2) can meaningfully improve children’s lives in developing regions and crisis zones.

    Role:
    - You are the Persuadee. The Persuader is trying to convince you to donate to Save the Children.

    Guidelines:
    1. Evaluate each request objectively and ask for clarification when details are unclear.
    2. Think about how the Persuader’s message resonates with your values and priorities before deciding what feels right for you.
    3. Respond politely, using complete sentences that add substance to the conversation (never empty or meaningless).
    4. Always respond in the format `[dialog_act] utterance`, where `dialog_act` is one of the allowed persuadee acts.
    5. Choose the dialog act that best reflects your genuine reaction; take action `[donate]` only when sufficiently convinced.
    """
    cot_prompt = """
    The following is the conversation history: {}
    Question: What are the mental states and future actions of the persuadee.? Answer:
    """
    
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
    class TRIP configuration for emotional support scenario
    """
    pass
