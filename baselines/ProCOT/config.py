from config.config import ModelConfig
from config.constants import rec_special_tokens_dict, neg_special_tokens_dict, es_special_tokens_dict, pg_special_tokens_dict


class ProCOTConfig(ModelConfig):
    """
    ProCot is a prompt based method
    therefore it only requires a prompt for planning
    """
    prompt = ""
    max_gen_tokens = 10

    def __init__(self, params):
        """
        constructor for class ProCot config
        :param params: a dictionary that contains parameters and their values
        """
        super().__init__()
        for k, v in params.items():
            setattr(self, k, v)


class ProCOTConfigForRecommendation(ProCOTConfig):
    """
    class pro_cot config for recommendation scenario
    """
    special_tokens_dict = rec_special_tokens_dict

    # general task instruction for proactive recommendation
    prompt = """
    Assume you are a recommender who is trying to recommend a {} to an user. 
    Your goal is to proactively lead the conversation with the user towards the target: *{}*
    Given the conversation history, in order to recommend the item to the user successfully, please select the most appropriate dialogue strategy.
    You can only reply by selecting one of the following dialogue strategy: 
    {}
    You answer should be in the following format: "Answer: X"
    """
    
    # chain of thought prompting
    cot_prompt = """
    The answer should start with a concise analysis of the recommendation progress
    and the user's mental states, and then follow by “To reach this goal, the most
    appropriate strategy is: ”.
    The following is the conversation history: {}
    Question: How is the current recommendation progress and the user's mental states, and
    which one is the most appropriate dialogue strategy? Answer:
    """
    
    # chain of thought prompting
    rewrite_prompt = """
    """
    
    rewrite_prompt_cot = """
    """
    
    # action mapping
    action_mapping = {
        "Ask about weather": "Ask about weather",
        "Play music": "Play music",
        "Music recommendation": "Music recommendation",
        "Q&A": "Q&A",
        "Chat about stars": "Chat about stars",
        "Music on demand": "Music on demand",
        "Movie recommendation": "Movie recommendation",
        "Say goodbye": "Say goodbye",
        "Ask about date": "Ask about date",
        "Ask questions": "Ask questions",
        "Greetings": "Greetings",
        "POI recommendation": "POI recommendation",
        "Food recommendation": "Food recommendation"
    }
    

class ProCOTConfigForNegotiation(ProCOTConfig):
    """
    class RTCP config for the negotiation scenario
    """
    special_tokens_dict = neg_special_tokens_dict
    
    # general task instruction
    prompt = """
    Assume you are the buyer. Given the conversation history, in order to reach a
    better deal with the seller, please first analyse the current bargain progress and
    the buyer’s target price in a concise summary, then select one of the following
    dialogue strategy: "Greetings". "Ask a question". "Answer a question". "Propose the
    first price." "Propose a counter price." "Use comparatives." "Confirm information."
    "Affirm confirmation." "Deny confirmation." "Agree with the proposal." "Disagree
    with the proposal".
    You answer should be in the following format: "Answer: X"
    """
    
    # chain of thought prompting
    cot_prompt = """
    The answer should start with a concise analysis of the current bargain progress
    and the buyer’s target price, and then follow by “To reach this goal, the most
    appropriate strategy is: ”.
    The following is the conversation history: {}. 
    Question: How is the current bargain progress and the buyer’s target price, and
    which one is the most appropriate dialogue strategy? Answer:
    """
    
    # chain of thought prompting
    rewrite_prompt = """
    Assume you are the expert analyst. Given the conversation history and a naive action instruction, 
    in order to reach a better deal with the seller, please modify 
    the following action instruction appropriately.
    Do not modify the action instruction entirely.
    You answer should be in the following format: "Answer: X"
    """
   
    rewrite_prompt_cot = """
    Please revise the naive action instruction Ensure to conduct thorough research on the product's market value before negotiating. 
    Maintain emotional composure throughout the conversation to avoid influencing offers negatively Be cautious about revealing too much information or making unrealistic demands that could deter the Seller. 
    Actively listen to the Seller’s needs and motivations, and ask clarifying questions to uncover potential areas for negotiation. 
    Respond constructively to rejections by exploring alternatives rather than walking away too quickly. 
    Recognize the importance of timing, context.
    The following is the conversation history: {}
    Here is the naive action instruction: {}
    Question: What are the rewritten action instruction ? Answer:
    """
    
    # action mapping
    action_mapping = {
        "Greetings": "greet",
        "Ask a question": "inquire",
        "Answer a question": "inform",
        "Propose the first price": "propose",
        "Propose a counter price": "counter",
        "Use comparatives": "counter-noprice",
        "Confirm information": "confirm",
        "Affirm confirmation": "affirm",
        "Deny confirmation": "deny",
        "Agree with the proposal": "agree",
        "Disagree with the proposal": "disagree"
    }
    pass


class ProCOTConfigForEmotionalSupport(ProCOTConfig):
    """
    class RTCP config for the emotional support conversation
    """
    special_tokens_dict = es_special_tokens_dict
    # general task instruction
    prompt = """
    Assume you are the therapist. Given the conversation history, in order to help
    the patient reduce their emotional distress and help them understand and work
    through the challenges, please first analyse the current therapy progress and the
    patient’s emotional state in a concise summary, then select one of the following
    dialogue strategy: "Question". "Self-disclosure". "Affirmation and Reassurance".
    "Providing Suggestions". "Reflection of feelings". "Information". "Restatement or
    Paraphrasing".
    """
    
    # chain of thought prompting
    cot_prompt = """
    The answer should start with a concise analysis of the current therapy progress
    and the patient’s emotional state, and then follow by “To reach this goal, the
    most appropriate strategy is: ”.
    The following is the conversation history: {}
    Question: How is the current therapy progress and the patient’s emotional state,
    and which one is the most appropriate dialogue strategy? Answer:
    """

    # chain of thought prompting
    rewrite_prompt = """
    Assume you are the expert analyst. Given the conversation history and a naive action instruction, 
    in order to help the patient reduce their emotional distress and help them understand and work
    through the challenges, please modify  the following action instruction appropriately.
    Do not modify the action instruction entirely.
    You answer should be in the following format: "Answer: X"
    """
    
    rewrite_prompt_cot = """
    Please add necessary information to the naive action instruction. 
    Ensure that the therapist accurately recognizes and reflects the patient's feelings without mislabeling, minimizing, or overlooking key emotions. 
    Incorporate open-ended, empathetic questions that encourage the patient to explore their feelings, focusing on the patient's experience rather than shifting attention away. 
    Validate the patient's protective responses before addressing deeper vulnerabilities, especially in trauma-related contexts, and avoid premature pushes into vulnerability that may overwhelm the patient.
    The following is the conversation history: {}
    Here is the naive action instruction: {}
    Question: What are the rewritten action instruction ? Answer:
    """
    
    action_mapping = {
        "Question": "Question",
        "Self-disclosure": "Self-disclosure",
        "Affirmation and Reassurance": "Affirmation and Reassurance",
        "Providing Suggestions": "Providing Suggestions",
        "Reflection of feelings": "Reflection of feelings",
        "Information": "Information",
        "Restatement or Paraphrasing": "Restatement or Paraphrasing",
    }
    pass
    
    
class ProCOTConfigForPersuation(ProCOTConfig):
    """
    class Proactive config for the persuation conversation
    """
    special_tokens_dict = pg_special_tokens_dict
    
    # general task instruction
    prompt = """
    Assume you are the Persuader. Given the conversation history, in order to convince the persuadee to
    donate for charity, please select the most appropriate dialogue strategy.
    You can only reply by selecting one of the following dialogue strategy to reach the goal: "Greeting". "Logical
    appeal". "Emotion appeal". "Credibility appeal". "Foot in the door". "Self-modeling". "Personal story". "Donation
    information". "Source-related inquiry". "Task-related inquiry". "Personal-related inquiry".
    You answer should be in the following format: "Answer: X"
    """
    
    # chain of thought prompting
    cot_prompt = """
    The answer should start with a concise analysis of the current persuation progress
    and the persuadee’s emotional state, and then follow by “To reach this goal, the
    most appropriate strategy is: ”.
    The following is the conversation history: {}
    Question: Which one is the most appropriate dialogue strategy? Answer:
    """

    # chain of thought prompting
    rewrite_prompt = """
    Assume you are the expert analyst. Given the conversation history and a naive action instruction, 
    in order to convince the persuadee to donate for charity, please modify the following action instruction appropriately.
    Do not modify the action instruction entirely.
    You answer should be in the following format: "Answer: X"
    """
    
    rewrite_prompt_cot = """
    Please revise the naive action instruction Ensure to take time to understand the Persuadee’s interests, values, or philanthropic priorities before discussing the charity’s needs 
    Make a clear, specific request regarding the donation amount, purpose, and impact Actively listen and respond to the Persuadee’s concerns, objections, or questions without dominating the conversation 
    Use emotionally connecting language (pathos) rather than relying solely on logic, facts, or jargon Build rapport and trust before asking.
    The following is the conversation history: {}
    Here is the naive action instruction: {}
    Question: What are the rewritten action instruction ? Answer:
    """
       
    action_mapping = {
        "Logical appeal": "logical-appeal",
        "Emotion appeal": "emotion-appeal",
        "Credibility appeal": "credibility-appeal",
        "Foot in the door": "foot-in-the-door",
        "Self-modeling": "self-modeling",
        "Personal story": "personal-story",
        "Donation information": "donation-information",
        "Source-related inquiry": "source-related-inquiry",
        "Task-related inquiry": "task-related-inquiry",
        "Personal-related inquiry": "personal-related-inquiry",
        "Greeting": "greeting"
    }
    pass
