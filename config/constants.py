USER_TOKEN = "[USER]"
SYSTEM_TOKEN = "[SYSTEM]"
KNOW_TOKEN = "[KNOW]"
PATH_TOKEN = "[PATH]"
SEP_TOKEN = "[SEP]"
PROFILE_TOKEN = "[PROFILE]"
CONTEXT_TOKEN = "[CONTEXT]"
GOAL_TOKEN = "[GOAL]"
TARGET = "[TARGET]"
TOPIC_TOKEN = "[TOPIC]"
PAD_TOKEN = "<pad>"
IGNORE_INDEX = -100

# special tokens for negotiation
BUYER_TOKEN = "[BUYER]"
SELLER_TOKEN = "[SELLER]"

# special tokens for emotional support
SEEKER_TOKEN = "[SEEKER]"
SUPPORTER_TOKEN = "[SUPPORTER]"

# special tokens for charity persuation
PERSUADER_TOKEN = "[PERSUADER]"
PERSUADEE_TOKEN = "[PERSUADEE]"

rec_special_tokens_dict = {
    'additional_special_tokens': [USER_TOKEN, SYSTEM_TOKEN, KNOW_TOKEN, PATH_TOKEN, SEP_TOKEN, PROFILE_TOKEN,
                                  CONTEXT_TOKEN, GOAL_TOKEN, TARGET],
}

neg_special_tokens_dict = {
    'additional_special_tokens': [SELLER_TOKEN, BUYER_TOKEN, PATH_TOKEN, SEP_TOKEN, CONTEXT_TOKEN, GOAL_TOKEN],
}

es_special_tokens_dict = {
    'additional_special_tokens': [SEEKER_TOKEN, SUPPORTER_TOKEN, PATH_TOKEN, SEP_TOKEN, CONTEXT_TOKEN, GOAL_TOKEN],
}

pg_special_tokens_dict = {
    'additional_special_tokens': [PERSUADER_TOKEN, PERSUADEE_TOKEN, PATH_TOKEN, SEP_TOKEN, CONTEXT_TOKEN, GOAL_TOKEN],
}

DURECDIAL_TARGET_GOALS = [
    "Movie recommendation",
    "Food recommendation",
    "Music recommendation",
    "POI recommendation",
]

DURECDIALGOALS = {
    'Ask about weather',
    'Play music',
    'Q&A',
    'Music on demand',
    'Movie recommendation',
    'Chat about stars',
    'Say goodbye',
    'Music recommendation',
    'Ask about date',
    'Ask questions',
    'Greetings',
    'POI recommendation',
    'Food recommendation',
}

P4G_GOALS = [
    "personal-story",
	"credibility-appeal",
	"emotion-appeal",
    "proposition-of-donation",
	"donation-information",
	"foot-in-the-door",
	"logical-appeal",
	"self-modeling",
	"task-related-inquiry",
	"source-related-inquiry",
	"personal-related-inquiry",
	"greeting",
]

BIG5_PERSONALITY = [
    "openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"
]
DECISION_MAKING_STYLE = [
    "directive", "analytical", "conceptual", "behavioral"
]

USER = 'user'
ASSISTANT = "assistant"

# scenario names
RECOMMENDATION = 'recommendation'
NEGOTIATION = 'negotiation'
EMOTIONAL_SUPPORT = 'emotional_support'
PERSUATION = 'persuation'

# logger names
TERMINAL_LOGGER = 'terminal'
FILE_LOGGER = 'file'
WANDB_LOGGER = 'wandb'


# datasets for recommendation
DURECDIAL = 'durecdial'
INSPIRED = 'inspired'

# datasets for negotiation
CRAIGSLIST_BARGAIN = 'craigslist_bargain'

# dataset for emotional support
ES_CONV = "es_conv"

# dataset for persuation
PERSUATION4GOOD = 'p4g'

# configuration of datasets
DURECDIAL_CONFIG_PATH = 'config/datasets/durecdial.yaml'
INSPIRED_CONFIG_PATH = 'config/datasets/inspired.yaml'
CRAIGSLIST_BARGAIN_CONFIG_PATH = 'config/datasets/craigslist_bargain.yaml'
ES_CONV_CONFIG_PATH = 'config/datasets/es_conv.yaml'
PERSUATION4GOOD_CONFIG_PATH = 'config/datasets/p4g.yaml'

MULTI_OBJECTIVE_RECOMMENDATION_CONFIG_PATH = 'config/scenario/multi_objective/recommendation.yaml'
MULTI_OBJECTIVE_NEGOTIATION_CONFIG_PATH = 'config/scenario/multi_objective/negotiation.yaml'
MULTI_OBJECTIVE_EMOTIONAL_SUPPORT_CONFIG_PATH = "config/scenario/multi_objective/emotional_support.yaml"
MULTI_OBJECTIVE_PERSUATION_CONFIG_PATH = "config/scenario/multi_objective/persuation.yaml"

SINGLE_OBJECTIVE_RECOMMENDATION_CONFIG_PATH = 'config/scenario/single_objective/recommendation.yaml'
SINGLE_OBJECTIVE_NEGOTIATION_CONFIG_PATH = 'config/scenario/single_objective/negotiation.yaml'
SINGLE_OBJECTIVE_EMOTIONAL_SUPPORT_CONFIG_PATH = "config/scenario/single_objective/emotional_support.yaml"
SINGLE_OBJECTIVE_PERSUATION_CONFIG_PATH = "config/scenario/single_objective/persuation.yaml"

# bart generation
BART_GENERATION = 'bart_gen'
BART_GENERATION_CONFIG_PATH = 'config/generation/BART.yaml'

# fine-tuned llm for generation
FINETUNED_LLM_GENERATION = 'finetuned_llm'
FINETUNED_LLM_GENERATION_CONFIG_PATH = 'config/generation/FINETUNED_LLM.yaml'

# QWEN_MODEL = "Qwen/Qwen2.5-7B-Instruct"
QWEN_MODEL = "Qwen/Qwen3-8B"

BERT = 'bert'
BERT_CONFIG_PATH = 'config/models/BERT.yaml'

BART = 'bart'
BART_CONFIG_PATH = 'config/models/BART.yaml'

# rtcp model
RTCP = 'rtcp'
RTCP_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/RTCP_REC.yaml'
RTCP_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/RTCP_NEG.yaml'
RTCP_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT = 'config/models/RTCP_ES.yaml'

# unimind model
UNIMIND = 'unimind'
UNIMIND_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/UNIMIND_REC.yaml'

# color model
COLOR = 'color'
COLOR_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/COLOR_REC.yaml'

# ppdpp model
PPDPP = 'ppdpp'
PPDPP_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/PPDPP_REC.yaml'
PPDPP_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/PPDPP_NEG.yaml'
PPDPP_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT = 'config/models/PPDPP_ES.yaml'
PPDPP_CONFIG_PATH_FOR_PERSUATION = 'config/models/PPDPP_PG.yaml'

# trip model
TRIP = 'trip'
TRIP_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/TRIP_REC.yaml'
TRIP_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/TRIP_NEG.yaml'
TRIP_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT = 'config/models/TRIP_ES.yaml'
TRIP_CONFIG_PATH_FOR_PERSUATION = 'config/models/TRIP_PG.yaml'

# Bayes-Adaptive LLM
BAYES_ADAPTIVE = 'bayes_adaptive_llm'
BAYES_CONFIG_PATH_FOR_PERSUATION = 'config/models/BAYES_P4G.yaml'

# DPDP model
DPDP = 'dpdp'
DPDP_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/DPDP_REC.yaml'
DPDP_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/DPDP_NEG.yaml'
DPDP_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT = 'config/models/DPDP_ES.yaml'

PREFERENCE_PPDPP = 'preference_ppdpp'
PREFERENCE_PPDPP_CONFIG_PATH = 'config/models/PREFERENCE_PPDPP.yaml'

# MODPL old version
MODPL = 'modpl'
MODPL_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/MODPL_REC.yaml'
MODPL_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/MODPL_NEG.yaml'
MODPL_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT = 'config/models/MODPL_ES.yaml'

# contextual MODPL
CONTEXTUAL_MODPL = "ct_modpl"
CONTEXTUAL_MODPL_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/CT_MODPL_REC.yaml'
CONTEXTUAL_MODPL_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/CT_MODPL_NEG.yaml'
CONTEXTUAL_MODPL_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT = 'config/models/CT_MODPL_ES.yaml'

# Set Max PADPP
SMP_PADPP = "smp_padpp"
SMP_PADPP_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/SMP_PADPP_REC.yaml'
SMP_PADPP_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/SMP_PADPP_NEG.yaml'
SMP_PADPP_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT = 'config/models/SMP_PADPP_ES.yaml'

# Min Dist PADPP
MIN_DIST_PADPP = "min_dist_padpp"
MIN_DIST_PADPP_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/MIN_DIST_PADPP_REC.yaml'
MIN_DIST_PADPP_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/MIN_DIST_PADPP_NEG.yaml'
MIN_DIST_PADPP_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT = 'config/models/MIN_DIST_PADPP_ES.yaml'

# DDQL
DDQL = "ddql"
DDQL_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/DDQL_REC.yaml'
DDQL_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/DDQL_NEG.yaml'
DDQL_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT = 'config/models/DDQL_ES.yaml'

# Envelope
ENVELOPE = "envelope"
ENVELOPE_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/ENVELOPE_REC.yaml'
ENVELOPE_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/ENVELOPE_NEG.yaml'
ENVELOPE_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT = 'config/models/ENVELOPE_ES.yaml'

# Proactive Large Language Model
# PRO_LLM = "pro_llm"
# PRO_LLM_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/PRO_LLM_REC.yaml'
# PRO_LLM_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/PRO_LLM_NEG.yaml'
# PRO_LLM_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT = 'config/models/PRO_LLM_ES.yaml'

# Prompt Refiner
PROMPT_REFINER = "prompt_refiner"
PROMPT_REFINER_CONFIG_PATH_FOR_RECOMMENDATION = 'config/models/PROMPT_REFINER_REC.yaml'
PROMPT_REFINER_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/PROMPT_REFINER_NEG.yaml'
PROMPT_REFINER_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT = 'config/models/PROMPT_REFINER_ES.yaml'
PROMPT_REFINER_CONFIG_PATH_FOR_PERSUATION = 'config/models/PROMPT_REFINER_PG.yaml'

# proactive chain-of-thought (ProCOT)
PRO_COT = "pro_cot"
PRO_COT_CONFIG_PATH = "config/models/PRO_COT.yaml"

# standard prompting
STANDARD = "standard"
STANDARD_CONFIG_PATH = "config/models/STANDARD.yaml"

# ICL_AIF
ICL_AIF = "icl_aif"
ICL_AIF_CONFIG_PATH = "config/models/ICL_AIF.yaml"

# Proactive
PROACTIVE = "proactive"
PROACTIVE_CONFIG_PATH = "config/models/PROACTIVE.yaml"

# Ask-an-Expert
ANE = 'ane'
ANE_CONFIG_PATH = "config/models/ANE.yaml"

# GDP-Zero
GDP_ZERO = 'gdp_zero'
GDP_ZERO_CONFIG_PATH_FOR_NEGOTIATION = 'config/models/GDP_ZERO_NEG.yaml'
GDP_ZERO_CONFIG_PATH_FOR_EMOTIONAL_SUPPORT = 'config/models/GDP_ZERO_ES.yaml'

# types of evaluators
OFFLINE = 'offline'
ONLINE = 'online'

# metrics
ACCURACY = 'acc'
PRF1 = 'prf1'
BLEU_N = 'bleu_n'
ROUGE_N = 'rouge_n'
DIST_N = 'dist_n'

PLANNING = 'planning'
GENERATION = 'generation'

OFFLINE_METRICS = {
    PLANNING: [ACCURACY, PRF1],
    GENERATION: [BLEU_N, ROUGE_N, DIST_N]
}

# success rate
AVG_TURN = 'avg_turn'
SUCCESS_RATE = 'sr'
TOTAL_REWARD = 'total_reward'

# User statisfaction
USER_REWARD = 'user_reward'

# objectives for recommendation
ITEM_FREQ = 'item_freq'

# objectives for negotiation
SL_RATIO = 'sl_ratio'
FAIRNESS = 'fairness'

# objectives for emotional support
TOXICITY = 'toxicity'

# llama 3
LLAMA3 = "llama3"
LLAMA3_GENERATION_CONFIG_PATH = 'config/generation/LLAMA3.yaml'
LLAMA3_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"

#
QWEN = "qwen"
QWEN_GENERATION_CONFIG_PATH = 'config/generation/QWEN.yaml'

LLM_MODEL_MAPPING = {
    LLAMA3: LLAMA3_MODEL, 
    QWEN: QWEN_MODEL
}

EOS_TOKEN_MAPPING = {
    LLAMA3: "<|eot_id|>",
    QWEN: "<|im_end|>"
}

# prompts for llm generation
CHATGPT = 'chatgpt'
CHATGPT_GENERATION_CONFIG_PATH = 'config/generation/CHATGPT.yaml'

# vicuna generation
VICUNA = 'vicuna'
VICUNA_GENERATION_CONFIG_PATH = 'config/generation/VICUNA.yaml'

# wrapper generation
WRAPPER_GENERATION = 'wrapper_gen'
WRAPPER_GENERATION_CONFIG_PATH = 'config/generation/WRAPPER.yaml'

# the prompts for chat gpt
# including prompts for recommendation, negotiation and emotional support
CHATGPT_PROMPT_FOR_RECOMMENDATION = [
    {"role": "system",
     "content": "Now enter the role-playing mode. "
                "In the following conversation, you will play as a recommender in a recommendation game."},
    {"role": "user",
     "content": "You are the recommender who is trying to recommend an item to the user. "
                "Please reply with only one short and succinct sentence. {}."
     },
]

CHATGPT_PROMPT_FOR_NEGOTIATION = [
    {"role": "system",
     "content": "Now enter the role-playing mode. "
                "In the following conversation, you will play as a Buyer in a price bargaining game."},
    {"role": "user",
     "content": "You are the Buyer who is trying to buy the {} with the price of {}. Product description: {} \nPlease "
                "reply with only one short and succinct sentence. {}"
     }
]

CHATGPT_PROMPT_FOR_EMOTIONAL_SUPPORT = [
    {"role": "system",
     "content": "Now enter the role-playing mode. In the following conversation, you will play as a therapist in a counselling conversation with a patient."},

    {"role": "user",
     "content": "You are the therapist who is trying to help the patient reduce their emotional distress and help them understand and work through the challenges. "
                "Please reply with only one short and succinct sentence. {}."
     }
]

CHATGPT_PROMPT_FOR_PERSUATION = [
    {"role": "system",
     "content":(
            "Now enter role-playing mode. You are a Persuader in a persuasion game.\n"
            "In this conversation, you try to persuade the Persuadee to donate to the charity 'Save the Children'.\n"
            "Save the Children is headquartered in London and helps fight poverty around the world, "
            "especially in developing countries and war zones. Small donations like $1 or $2 can already "
            "make a meaningful difference for children in need.\n"
            "You must always speak as the Persuader, using natural, conversational language.\n"
     )
    },
    {"role": "user",
     "content":(
            "You are the Persuader who is trying to convince the Persuadee to donate to Save the Children.\n"
            "Your high-level strategy for this turn is:\n"
            "{goal_description}\n\n"
            "The conversation history between you (Persuader) and the Persuadee is given below this instruction.\n"
            "{dialogue_context}\n\n"
            "When you answer, follow these general rules:\n"
            "- Consider whole conversation context.\n"
            "- Apply the given strategy in a natural way within the same sentence.\n"
            "- Use varied, natural wording across turns; avoid repeating the same sentence openings or phrasing.\n"
            "- Add new, concise and specific content instead of simply repeating yourself or restating the Persuadee's words.\n"
            "Please reply with only one short and succinct sentence.\n"
        )
    }
]
# prompts for the llama 3 mode
# including the recommendation, negotiation and emotional support
LLAMA3_PROMPT_FOR_RECOMMENDATION = [
    {"role": "system",
     "content": "Now enter the role-playing mode. "
                "In the following conversation, you will play as a recommender in a recommendation game."},
    {"role": "user",
     "content": f"You are the recommender who is trying to recommend an item to the user. "
                "Your topic sets: {}."
                "Please reply with only one short and succinct sentence. {}"
     },
]

LLAMA3_PROMPT_FOR_NEGOTIATION = [
    {"role": "system",
     "content": "Now enter the role-playing mode. "
                "In the following conversation, you will play as a buyer in a price bargaining game."},
    {"role": "user",
     "content": "You are the buyer who is trying to buy the {} with the price of {}. Product description: {} \n . "
                "Please reply with only one short and succinct sentence. {}"
     }
]

LLAMA3_PROMPT_FOR_EMOTIONAL_SUPPORT = [
    {"role": "system",
     "content": "Now enter the role-playing mode. In the following conversation, you will play as a therapist in a counselling conversation with a patient."},

    {"role": "user",
     "content": "You are the therapist who is trying to help the patient reduce their emotional distress and help them understand and work through the challenges. "
                "Please reply with only one short and succinct sentence. {}"
     }
]

LLAMA3_PROMPT_FOR_PERSUATION = [
    {"role": "system",
     "content":(
            "Now enter role-playing mode. You are a Persuader in a persuasion game.\n"
            "In this conversation, you try to persuade the Persuadee to donate to the charity 'Save the Children'.\n"
            "Save the Children is headquartered in London and helps fight poverty around the world, "
            "especially in developing countries and war zones. Small donations like $1 or $2 can already "
            "make a meaningful difference for children in need.\n"
            "You must always speak as the Persuader, using natural, conversational language.\n"
     )
    },
    {"role": "user",
     "content":(
            "You are the Persuader who is trying to convince the Persuadee to donate to Save the Children.\n"
            "Your high-level strategy for this turn is:\n"
            "{goal_description}\n\n"
            "The conversation history between you (Persuader) and the Persuadee is given below this instruction.\n"
            "{dialogue_context}\n\n"
            "When you answer, follow these general rules:\n"
            "- Consider whole conversation context.\n"
            "- Apply the given strategy in a natural way within the same sentence.\n"
            "- Use varied, natural wording across turns; avoid repeating the same sentence openings or phrasing.\n"
            "- Add new, concise and specific content instead of simply repeating yourself or restating the Persuadee's words.\n"
            "Please reply with only one short and succinct sentence.\n"
        )
    }
]


# prompts for the llama 3 mode
# including the recommendation, negotiation and emotional support
QWEN_PROMPT_FOR_RECOMMENDATION = [
    {"role": "system",
     "content": "Now enter the role-playing mode. "
                "In the following conversation, you will play as a recommender in a recommendation game."},
    {"role": "user",
     "content": f"You are the recommender who is trying to recommend an item to the user. "
                "Your topic sets: {}."
                "Please reply with only one short and succinct sentence. {}"
     },
]

QWEN_PROMPT_FOR_NEGOTIATION = [
    {"role": "system",
     "content": "Now enter the role-playing mode. "
                "In the following conversation, you will play as a buyer in a price bargaining game."},
    {"role": "user",
     "content": "You are the buyer who is trying to buy the {} with the price of {}. Product description: {} \n . "
                "Please reply with only one short and succinct sentence. {}"
     }
]

QWEN_PROMPT_FOR_EMOTIONAL_SUPPORT = [
    {"role": "system",
     "content": "Now enter the role-playing mode. In the following conversation, you will play as a therapist in a counselling conversation with a patient."},

    {"role": "user",
     "content": "You are the therapist who is trying to help the patient reduce their emotional distress and help them understand and work through the challenges. "
                "Please reply with only one short and succinct sentence. {}"
     }
]


QWEN_PROMPT_FOR_PERSUATION = [
    {"role": "system",
     "content":(
            "Now enter role-playing mode. You are a Persuader in a persuasion game.\n"
            "In this conversation, you try to persuade the Persuadee to donate to the charity 'Save the Children'.\n"
            "Save the Children is headquartered in London and helps fight poverty around the world, "
            "especially in developing countries and war zones. Small donations like $1 or $2 can already "
            "make a meaningful difference for children in need.\n"
            "You must always speak as the Persuader, using natural, conversational language.\n"
     )
    },
    {"role": "user",
     "content":(
            "You are the Persuader who is trying to convince the Persuadee to donate to Save the Children.\n"
            "Your high-level strategy for this turn is:\n"
            "{goal_description}\n\n"
            "The conversation history between you (Persuader) and the Persuadee is given below this instruction.\n"
            "{dialogue_context}\n\n"
            "When you answer, follow these general rules:\n"
            "- Consider whole conversation context.\n"
            "- Apply the given strategy in a natural way within the same sentence.\n"
            "- Use varied, natural wording across turns; avoid repeating the same sentence openings or phrasing.\n"
            "- Add new, concise and specific content instead of simply repeating yourself or restating the Persuadee's words.\n"
            "Please reply with only one short and succinct sentence.\n"
        )
    }
]

# prompts for the wrapper class
WRAPPER_PROMPT_FOR_RECOMMENDATION = []
WRAPPER_PROMPT_FOR_NEGOTIATION = []
WRAPPER_PROMPT_FOR_EMOTIONAL_SUPPORT = []

# the prompts for vicuna model
# including prompts for recommendation, negotiation and emotional support
VICUNA_PROMPT_FOR_RECOMMENDATION = [
    {"role": "system",
     "content": "Now enter the role-playing mode."
                "In the following conversation, you will play as a recommender in a recommendation game."},
    {"role": "user",
     "content": "Please reply with only one short and succinct sentence. {} . Now start the game."
     }
]

VICUNA_PROMPT_FOR_NEGOTIATION = [
    {"role": "system",
     "content": "Now enter the role-playing mode. "
                "In the following conversation, you will play as a buyer in a price bargaining game."},
    {"role": "user",
     "content": "You are the buyer who is trying to buy the {} with the price of {}. Product description: {}\nPlease "
                "reply with only one short and succinct sentence. {} Now start the game."
     }
]

VICUNA_PROMPT_FOR_EMOTIONAL_SUPPORT = [
    {"role": "system",
     "content": "Now enter the role-playing mode. In the following conversation, you will play as a therapist in a counselling conversation with a patient."},
    {"role": "assistant",
     "content": "You are the therapist who is trying to help the patient reduce their emotional distress and help them understand and work through the challenges. "
                "Please reply with only one short and succinct sentence. {} .Are you ready to play the game?"},
    {"role": "assistant", "content": "Yes, I'm ready to play the game!"}

]

LLM_MODEL = "gpt-4o-mini"

# planning and generation prompts for negotiation
PLAN_NEGOTIATION_INSTRUCTION = """
    Now enter the role-playing mode. In the following conversation, you will play as a Buyer in a price bargaining game.
    Your actions and their descriptions are as follows:
    1. '<|greet|>': 'Please say hello or chat randomly.'.
    2. '<|inquire|>': 'Please ask any question about product, year, price, usage, etc.'.
    3. '<|inform|>': 'Please provide information about the product, year, usage, etc.'.
    4. '<|propose|>': 'Please initiate a price or a price range for the product.'.
    5. '<|counter|>': 'Please propose a new price or a new price range.'.
    6. '<|counter-noprice|>': 'Please propose a vague price by using comparatives with existing price.'.
    7. '<|confirm|>': 'Please ask a question about the information to be confirmed.'.
    8. '<|affirm|>': 'Please give an affirmative response to a confirm.'.
    9. '<|deny|>': 'Please give a negative response to a confirm.'.
    10. '<|agree|>': 'Please agree with the proposed price.'.
    11. '<|disagree|>': 'Please disagree with the proposed price.'.
    You are the Buyer who is trying to buy the {} with the price of {}. Product description: {}
    Based on the following conversation, please propose an appropriate action. The conversation is as follows:
"""

GEN_NEGOTIATION_INSTRUCTION = """
    Now enter the role-playing mode. In the following conversation, you will play as a Buyer in a price bargaining game.
    You are the Buyer who is trying to buy the {} with the price of {}. Product description: {}
    Based on the following conversation and your action, please generate a corresponding response. The conversation is as follows:
"""   

ACTION_INSTRUCTION = "Now please indicate the appropriate action:"
GENERATION_INSTRUCTION = "Given that your action is:' {}', your corresponding response is: "

# planning and generation prompts for recommendation                
RECOMMENDATION_INSTRUCTION = "Now enter the role-playing mode. In the following conversation, you will play as a recommender in a recommendation game."

# planning and generation prompts for emotional support
PLAN_EMOTIONAL_SUPPORT_INSTRUCTION = """
    Now enter the role-playing mode. In the following conversation, you will play as a Therapist in a counselling conversation with a Patient.
    Your actions and their descriptions are as follows:
    1.'<|Question|>': "Please ask the Patient to elaborate on the situation they just described.".
    2.'<|Self-disclosure|>': "Please provide a statement relating to the Patient about the situation they just described."/
    3.'<|Affirmation and Reassurance|>': "Please provide affirmation and reassurance to the Patient on the situation they just described.".
    4.'<|Providing Suggestions|>': "Please provide suggestion to the Patient on the situation they just described.".
    5.'<|Others|>': "Please chat with the Patient.".
    6.'<|Reflection of feelings|>': "Please acknowledge the Patient's feelings about the situation they described.".
    7.'<|Information|>': "Please provide factual information to help the Patient with their situation.".
    8.'<|Restatement or Paraphrasing|>': "Please acknowledge the Patient's feelings by paraphrasing their situation.".
    You are the therapist who is trying to help the patient reduce their emotional distress and help them understand and work through the challenges.
    Based on the following conversation, please propose an appropriate action. The conversation is as follows:
"""

GEN_EMOTIONAL_SUPPORT_INSTRUCTION = """
    Now enter the role-playing mode. In the following conversation, you will play as a Therapist in a counselling conversation with a Patient.
    You are the therapist who is trying to help the patient reduce their emotional distress and help them understand and work through the challenges. 
    Based on the following conversation and your action, please generate a corresponding response. The conversation is as follows:
"""

# a mapping from goal to textual description
# for the negotiation task
NEGOTIATION_GOAL2DESCRIPTION = {'greet': 'Please say hello or chat randomly.',
                                'inquire': 'Please ask any question about product, year, price, usage, etc.',
                                'inform': 'Please provide information about the product, year, usage, etc.',
                                'propose': 'Please initiate a price or a price range for the product.',
                                'counter': 'Please propose a new price or a new price range.',
                                
                                'counter-noprice': 'Please propose a vague price by using comparatives with existing price.',
                                
                                'confirm': 'Please ask a question about the information to be confirmed.',
                                'affirm': 'Please give an affirmative response to a confirm.',
                                
                                'deny': 'Please give a negative response to a confirm.',
                                
                                'agree': 'Please accept the lastest proposed price of the Seller.',
                                'disagree': 'Please disagree with the lastest proposed price of the Seller.',
                                
                                # 'agree': 'You must to agree with the proposed price of the seller.',
                                # 'disagree': 'You must to disagree with the proposed price of the seller.',
                                
                                # for standard prompting. There is no instruction for dialogue strategy
                                'Standard': ""}

NEGOTIATION_REWRITE_PROMPT = """
        Assume you are a bargaining analyst. Given the following conversation history, a Buyer response, and a naive action instruction, 
        please revise the action instruction to make it aligned with the conversation history and the Buyer response.
        Please do not rewrite the action instruction entirely.
        You answer should be in the following format: "Answer: X"
"""

NEGOTIATION_REWRITE_COT_PROMPT = """
        Please do not rewrite the action instruction entirely.
        Please add nessessary information to the given action instruction to make it aligned with the conversation history and the Buyer response. 
        The conversation history: {}
        The Buyer response: {}
        The action instruction: {}
        Question: What are the rewritten action instruction ? Answer:
"""

# a mapping from goal to textual description
# for the recommendation task
DURECDIAL_GOAL2DESCRIPTION = {'Ask about weather': 'Please provide information about the weather.',
                              'Play music': 'Please select an appropriate song from your given topic set and reply that song is playing.',
                              'Music recommendation': 'Please recommend the song \"{}\" to the user',
                              'Q&A': 'Please answer questions asked by the user.',
                              'Chat about stars': "Please select an appropriate movie star from your given topic set and provide information about the movie star.",
                              'Music on demand': 'Please select an appropriate song from your given topic set and reply that song is suitable for the user demand',
                              'Movie recommendation': 'Please recommend the movie \"{}\" to the user.',
                              'Say goodbye': 'Please say goodbye to the user.',
                              'Ask about date': 'Please provide information regarding date.',
                              'Ask questions': 'Please select an appropriate topic from your given topic set and ask questions regarding that topic',
                              'Greetings': 'Please say hello or chat randomly.',
                              'POI recommendation': 'Please recommend the restaurant \"{}\" to the user.',
                              'Food recommendation': 'Please recommend the food \"{}\" to the user.'
                              }

# a mapping from goal to textual description
# for the emotional support task
ES_CONV_GOAL2DESCRIPTION = {"Question": "Please ask the Patient to elaborate on the situation they just described.",
                            "Self-disclosure": "Please provide a statement relating to the Patient about the situation they just described.",
                            "Affirmation and Reassurance": "Please provide affirmation and reassurance to the Patient on the situation they just described.",
                            "Providing Suggestions": "Please provide suggestion to the Patient on the situation they just described.",
                            "Others": "Please chat with the Patient.",
                            "Reflection of feelings": "Please acknowledge the Patient's feelings about the situation they described.",
                            "Information": "Please provide factual information to help the Patient with their situation.",
                            "Restatement or Paraphrasing": "Please acknowledge the Patient's feelings by paraphrasing their situation.",

                            # for standard prompting. There is no instruction for dialogue strategy
                            "Standard": ""}

EMOTIONAL_SUPPORT_REWRITE_PROMPT = """
        Assume you are a psychological analyst:. Given the following conversation history, a Therapist response, and a naive action instruction, 
        please revise the action instruction to make it aligned with the conversation history and the Therapist response.
        Please do not rewrite the action instruction entirely.
        You answer should be in the following format: "Answer: X"
"""

EMOTIONAL_SUPPORT_REWRITE_COT_PROMPT = """
        Please do not rewrite the action instruction entirely.
        Please add nessessary information to the given action instruction to make it aligned with the conversation history and the Therapist response. 
        The conversation history: {}
        The Therapist response: {}
        The action instruction: {}
        Question: What are the rewritten action instruction ? Answer:
"""

# a mapping from goal to textual description
# for the recommendation task, inspired dataset
INSPIRED_GOAL2DESCRIPTION = {
    'opinion_inquiry': 'Please asks for user’s opinion on the \"{}\" movie-related attributes.',
    'acknowledgment': 'Please acknowledge the user preference.',
    'no_strategy': 'Please randomly respond the user.',
    'encouragement': 'Please praise of the user’ movie taste and encouragement to watch \"{}\"',
    'personal_opinion': "Please express your subjective opinion about \"{}\", including its plot, actors, or other movie attributes.",
    'rephrase_preference': 'Please rephrase the user preference for confirmation.',
    'transparency': 'Please disclose your thinking process of understanding the user’ preference.',
    'offer_help': 'Please disclose explicit intention to help the user or being transparent.',
    'personal_experience': 'Please share your personal experience related to a movie.',
    'experience_inquiry': 'Please ask for user’s experience on movie watching, such as whether the user has watched a certain movie or not.',
    'self_modeling': 'Please becomes a role model to do something first so that the user would follow.',
    'preference_confirmation': 'Please ask or rephrase the user’s preference.',
    'similarity': 'Please being like-minded toward the users about their movie preference to produce similarity among them.',
    'credibility': 'provide provide factual information about movie attributes, such as the plot, actors, or awards that \"{}\" has'
}


# a mapping from goal to textual description
# for the emotional support task
P4G_GOAL2DESCRIPTION = {"logical-appeal": "Please use of reasoning and evidence to convince the persuadee.",
                            "emotion-appeal": "Please elicit the specific emotions to influence the persuadee.",
                            
                            "credibility-appeal": """Please use credentials and cite organizational impacts to establish credibility and earn the user’s trust.""",
                            
                            "proposition-of-donation": """Please explicitly invite the persuadee to donate or take the next concrete step toward donating.""",
                            
                            "foot-in-the-door": """Please use the strategy of starting with small donation requests
                                                to facilitate compliance followed by larger requests.""",
                                                
                            "self-modeling": """Please use the self-modeling strategy where you first indicates
                                            the persuadee own intention to donate and chooses to act as a
                                            role model for the persuadee to follow.""",
                            
                            "personal-story": """Please use narrative exemplars to illustrate someone donation
                                            experiences or the beneficiaries positive outcomes, which can
                                            motivate others to follow the actions.""",
                            
                            "donation-information": """Please provide specific information about the donation task,
                                                    such as the donation procedure, donation range, etc. By providing detailed action guidance, this strategy can enhance the
                                                    persuadee’s self-efficacy and facilitates behavior compliance.""",
                            
                            "source-related-inquiry": """Please ask if the persuadee is aware of the organization (i.e.,
                                                        the source in our specific donation task).""",
                            
                            "task-related-inquiry": """Please ask about the persuadee opinion and expectation related
                                                    to the task, such as their interests in knowing more about the
                                                    organization.""",

                            "personal-related-inquiry": """Please asks about the persuadee previous personal experiences
                                                        relevant to charity donation.""",
                                                        

                            "greeting": "Please say hello or greet the user.",
                            
                            # for standard prompting. There is no instruction for dialogue strategy
                            "Standard": ""}

#prompt for collect preference pairs
PREFERENCE_PAIR_PROMPT_P4G = "You are the Persuader. " \
        "Generate the Persuader reply that advances persuasion in a way that persuades the Persuadee to donate to Save the Children.\n" \
        "Conversation so far:\n"

# preference pair extraction for negotiation
PREFERENCE_PAIR_PROMPT_NEGOTIATION = "You are the Buyer. " \
        "Generate the Buyer reply that advances negotiation in a way that helps reach an agreement with the Seller.\n" \
        "Conversation so far:\n"