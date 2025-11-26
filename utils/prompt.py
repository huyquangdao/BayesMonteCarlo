import os
import copy

from dotenv import load_dotenv
import openai

from googleapiclient import discovery
import json

import transformers
from transformers import pipeline
import torch

from tenacity import (
    retry,
    stop_after_attempt,
    wait_random_exponential,
    retry_if_exception_type
)  # for exponential backoff

from config.constants import LLM_MODEL, LLAMA3, CHATGPT, LLAMA3_MODEL, QWEN_MODEL, QWEN

load_dotenv()

# Build a compatibility layer for openai v0.28 (ChatCompletion) and v1.x (OpenAI client)
API_KEY = os.getenv("API_KEY")
MODEL = LLM_MODEL
OpenAIClient = getattr(openai, "OpenAI", None)

if OpenAIClient:
    client = OpenAIClient(api_key=API_KEY) if API_KEY else OpenAIClient()
    _chat_completion = client.chat.completions.create
    retry_exceptions = tuple(
        exc for exc in (
            getattr(openai, "APIError", None),
            getattr(openai, "APIConnectionError", None),
            getattr(openai, "RateLimitError", None),
            getattr(openai, "ServiceUnavailableError", None),
            getattr(openai, "Timeout", None),
        ) if exc
    )
else:
    openai.api_key = API_KEY
    _chat_completion = openai.ChatCompletion.create
    error_module = getattr(openai, "error", None)
    retry_exceptions = tuple(
        getattr(error_module, name, None)
        for name in ["APIError", "APIConnectionError", "RateLimitError", "ServiceUnavailableError", "Timeout"]
        if error_module and getattr(error_module, name, None)
    )

# Fallback to a broad exception to keep tenacity happy if none of the OpenAI-specific exceptions are available.
if not retry_exceptions:
    retry_exceptions = (Exception,)


@retry(
    retry=retry_if_exception_type(retry_exceptions),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(10)
)
def chat_completion_with_backoff(**kwargs):
    return _chat_completion(**kwargs)

# API for toxicity evaluation
PERSPECTIVE_API_KEY = os.getenv('PERSPECTIVE_KEY')

sentiment_analysis = pipeline(model="cardiffnlp/twitter-roberta-base-sentiment")

def create_llm_pipeline(model_name, eos_token):
    # llama3 pipeline
    llm_pipeline = transformers.pipeline(
        "text-generation",
        model=model_name,
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    terminators = [
        llm_pipeline.tokenizer.eos_token_id,
        llm_pipeline.tokenizer.convert_tokens_to_ids(eos_token)
    ]
    return llm_pipeline, terminators

def call_llm_model(prompt, temperature=0.0, max_token=30, n_return_sequences=1, llm_pipeline = None, terminators = None):
    """
    function that calls the llama3 model
    :param prompt: the input prompt
    :param temperature: the prompting temperature
    :param max_token: max gen tokens
    :return:
    """
    if llm_pipeline is None or terminators is None:
        # Fallback: return empty text when no pipeline is available (e.g., chatgpt mode without local pipeline)
        if n_return_sequences > 1:
            return ["" for _ in range(n_return_sequences)]
        return ""
    
    response = llm_pipeline(
        prompt,
        max_new_tokens=max_token,
        eos_token_id=terminators,
        do_sample=True,
        temperature=temperature,
        top_p=0.95,
        top_k=40,
        min_p =0,
        num_return_sequences=n_return_sequences
    )
    if n_return_sequences > 1:
        return [x["generated_text"][-1]["content"] for x in response]
    else:
        return response[0]["generated_text"][-1]["content"]


def reformat_demonstration(demonstration, is_agent_start=False):
    """
    function that reformat the demonstrative conversation
    @param demonstration: the given conversation
    @param is_agent_start: True if the system starts the conversation else False
    @return: the reformated demonstrative conversation
    """
    new_demonstration = []
    role = 0
    if is_agent_start:
        role = -1
    for utt in demonstration:
        if role % 2 == 0:
            new_demonstration.append({'role': 'user', 'content': utt})
        elif role == -1 or role % 2 != 0:
            new_demonstration.append({'role': 'assistant', 'content': utt})
        role += 1
    return new_demonstration


def _get_message_content(choice):
    """Extract message content from OpenAI response choice across API versions."""
    message = getattr(choice, "message", None)
    if message is not None:
        content = getattr(message, "content", None)
        if content is None and isinstance(message, dict):
            content = message.get("content")
        if content is not None:
            return content
    if isinstance(choice, dict):
        return choice.get("message", {}).get("content")
    return None


def call_llm(prompt, n=1, temperature=0.0, max_token=10, model_type='chatgpt', **kwargs):
    """
    function that calls llm for n times using the given prompt
    :param prompt: the given input prompt
    :param n: number of times we call the llm
    :param temperature: the temperature we use to prompt the llm
    :param max_token: the maximum number of output tokens
    :param model_type: the name of the large language mdoel
    :return:
    """
    responses = []
    # call llm for n times
    for i in range(n):
        # the llm is the chatgpt model
        if model_type == CHATGPT:
            # call the llm with backoff
            response = chat_completion_with_backoff(
                model=MODEL,
                messages=prompt,
                temperature=temperature,
                max_tokens=max_token
            )
            responses.append(_get_message_content(response.choices[0]) or "")
        # the llm is the llama 3 model
        else:
            # do something here
            # no not enable thinking
            no_think = "/no_think" if model_type == QWEN else ""
            for p in prompt:
                for k in p.keys():
                    if 'system' in p[k]:
                        p['content'] = no_think + p['content']    
                        
            # print(prompt)        
            responses.append(call_llm_model(prompt, temperature, max_token, **kwargs))
            # print(responses)
            # assert 1 == 0
    responses = [response.replace("<think>", "").replace("</think>", "").strip() for response in responses]
    return responses


def get_llm_based_assessment_for_recommendation(target_topic, 
                                                simulated_conversation,
                                                demonstration=None,
                                                n=10,
                                                temperature=1.1,
                                                max_tokens=50,
                                                profile_description=None,
                                                model_type='chatgpt',
                                                **kwargs
                                                ):
    """
    function that computes an target-driven assessment given the current conversation
    :param target_topic: the target item
    :param simulated_conversation: the generated conversation
    :param demonstration: an demonstrative example
    :param n: the number of times we prompt the model
    :param temperature: the temperature used to prompt the llm
    :param max_tokens: the maximal number of tokens used to prompt the llm
    :return:
    """
    # messages = []
    # if demonstration is not None:
    #     system_instruction_1 = ''' This is an example of a {} conversation between an user (you) and the system.
    #     In this conversation, the user (you) accepted  the item : {}
    #     '''.format(demonstration['target_goal'], demonstration['target_topic'])
    #
    #     # the first instruction prompt
    #     messages = [
    #         {"role": "system", "content": system_instruction_1},
    #     ]
    #     # 1-shot demonstration
    #     for utt in reformat_demonstration(demonstration,
    #                                       is_agent_start=demonstration['goal_type_list'][0] == 'Greetings'):
    #         messages.append(utt)

    accept_string = ""
    reject_string = "reject"

    system_instruction_2 = f"""
    Based on the given conversation between an user and the recommender, please decide whether the user has accepted the item: {target_topic}.
    The conversation is:
    """
    
    system_instruction_3 = f"""Please decide whether the user has accepted the item: {target_topic}.
    Based on the give conversation, please decide whether the user is happy and willing to accept the target item: {target_topic}. 
    You can only reply with one of the following sentences: "Yes, the user has accepted the recommended item". "No, the user has not accepted the recommended item".
    """
    # the second instruction prompt
    messages = [
        {"role": "system", "content": system_instruction_2},
    ]
    # simulated conversation
    copied_conv = copy.deepcopy(simulated_conversation)
    for utt in copied_conv:
        # switch role
        if utt['role'] == 'system':
            utt['role'] = 'assistant'
        else:
            utt['role'] = 'user'
        temp = {'role': utt['role'], 'content': utt['content']}
        messages.append(temp)

    messages.append(
        {'role': 'user', 'content': system_instruction_3}
    )

    responses = []

    # prompt llm for n times
    if model_type == CHATGPT:
        for i in range(n):
            # calling the chat gpt
            response = chat_completion_with_backoff(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            responses.append(response.choices[0]['message']['content'])
    # calling the llama 3
    else:
        no_think = "/no_think" if model_type == QWEN else ""
        for p in messages:
            for k in p.keys():
                if 'system' in p[k]:
                    p['content'] = no_think + p['content']  
        
        responses.extend(call_llm_model(messages, temperature, max_tokens, n_return_sequences=n, **kwargs))

    # convert the text-based assessment to scalar based assessment
    # processing the llm's outputs
    # convert the text-based assessment to scalar based assessment
    # is_successful = 0
    # for response in responses:
    #     if response.lower() == accept_string.lower():
    #         is_successful += 1

    # print(responses)
    # assert 1 == 0
    return responses
    # return float(is_successful) / n


def get_llm_based_assessment_for_negotiation(simulated_conversation,
                                             n=10,
                                             temperature=1.1,
                                             max_tokens=20,
                                             model_type='chatgpt',
                                             **kwargs
                                             ):
    """
    function that assesses if there is a deal between the user and the system in a negotiation conversation
    :param simulated_conversation: 
    :param n:
    :param temperature: 
    :param max_tokens: 
    :return:
    """
    # the reward computation function for negotiation scenario
    # the following code is borrowed from the PPDPP official implementation
    # evaluating the progress at the last two rounds
    dial = ''
    for utt in simulated_conversation:
        if utt['role'] == 'user':
            role = 'Seller'
        else:
            role = 'Buyer'
        dial += f"{role}: {utt['content']}"
        dial += ". "

    # construct the message to prompt the llm
    # following the prompt from PPDPP
    messages = [{"role": "system",
                 "content": f"Given a conversation between a Buyer and a Seller, please decide whether the Buyer and the Seller have reached a deal."},
                {"role": "user",
                 "content": f"""You have to follow the instructions below during chat. 
                            1. Please decide whether the Buyer and the Seller have reached a deal at the end of the conversation. 
                            2. If they have reached a deal, please extract the deal price as [price]. 
                            You can only reply with one of the following sentences: "They have reached a deal at [price]". "They have not reached a deal."
                            The following is the conversation between a Buyer and a Seller: 
                            Buyer: Can we meet in the middle at 15? 
                            Seller: Deal, let's meet at 15 for this high-quality balloon.
                            Question: Have they reached a deal ? 
                            Answer: They have reached a deal at $15.
                            The following is the conversation between a Buyer and a Seller: 
                            Buyer: Can we meet in the middle at $1500? 
                            Seller: Deal, let's meet at $1500 for this high-quality balloon.
                            Question: Have they reached a deal ? 
                            Answer: They have reached a deal at $1500.
                            The following is the conversation between a Buyer and a Seller: 
                            Buyer: Can we meet in the middle at 150? 
                            Seller: Deal, let's meet at 150 for this high-quality balloon.
                            Question: Have they reached a deal ? 
                            Answer: They have reached a deal at $150.
                            The following is the conversation between a Buyer and a Seller:
                            Buyer: I'd be willing to pay $5400 for the truck.
                            Seller: I'm still a bit hesitant, but I'm willing to meet you halfway at $5600.
                            Question: Have they reached a deal? 
                            Answer: They have not reached a deal.
                            The following is the conversation: {dial}\n 
                            Question: Have they reached a deal? 
                            Answer: """
                }]

    responses = []
    # prompt llm for n times
    if model_type == CHATGPT:
        for i in range(n):
            # calling the chat gpt
            response = chat_completion_with_backoff(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            responses.append(response.choices[0]['message']['content'])
    else:
        no_think = "/no_think" if model_type == QWEN else ""
        for p in messages:
            for k in p.keys():
                if 'system' in p[k]:
                    p['content'] = no_think + p['content']    

        responses.extend(call_llm_model(messages, temperature, max_tokens, n_return_sequences=n, **kwargs))
    # print(model_type)
    # assert 1 == 0
    # responses = call_llm(messages,
    #                      n=n,
    #                      temperature=temperature,
    #                      max_token=50,
    #                      model_type=model_type,
    #                      **kwargs
    #                      )
            
    # convert the text-based assessment to scalar based assessment
    # processing the llm's outputs
    return responses


def get_llm_based_assessment_for_emotional_support(state,
                                                   simulated_conversation,
                                                   n=10,
                                                   temperature=1.1,
                                                   max_tokens=30,
                                                   model_type='chatgpt',
                                                   **kwargs
                                                   ):
    """
    function that assesses if the supporter successfully confront the seeker in a emotional support conversation
    :param simulated_conversation: the simulated conversation between the seeker and the supporter
    :param n: the number of prompting the LLMs
    :param temperature: the temperature used for prompting the LLMs
    :param max_tokens: the maximal number of tokens generated by the LLMs
    :return:
    """
    # the reward computation function for emotional support conversation
    # the following code is borrowed from the PPDPP official implementation
    dial = ''
    for utt in simulated_conversation:
        if utt['role'] == 'user':
            role = 'Patient'
        else:
            role = 'Supporter'
        dial += f"{role}: {utt['content']}"
        dial += ". "

    # construct the message to prompt the llm
    messages = [{"role": "system",
                 "content": "Given a conversation between a Therapist and a Patient, please assess whether the Patient' emotional issue has been solved after the conversation."},
                {"role": "user",
                 "content": "You can only reply with one of the following sentences: No, the Patient feels worse. No, the Patient feels the same. No, but the Patient feels better. Yes, the Patient's issue has been solved.\n\n"
                            "The following is a conversation about %s regarding %s: %s\nQuestion: Has the Patient's issue been solved? Answer: " % (
                                state['task_background']['emotion_type'], state['task_background']['problem_type'],
                                dial)}]

    responses = []
    # prompt llm for n times
    if model_type == CHATGPT:
        for i in range(n):
            # calling the chat gpt
            response = chat_completion_with_backoff(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            responses.append(response.choices[0]['message']['content'])
    # calling the llama 3
    else:
        no_think = "/no_think" if model_type == QWEN else ""
        for p in messages:
            for k in p.keys():
                if 'system' in p[k]:
                    p['content'] = no_think + p['content']    

        responses.extend(call_llm_model(messages, temperature, max_tokens, n_return_sequences=n, **kwargs))

    # convert the text-based assessment to scalar based assessment
    # processing the llm's outputs
    return responses


def get_llm_based_assessment_for_persuation(state,
                                            simulated_conversation,
                                            n=10,
                                            temperature=1.1,
                                            max_tokens=30,
                                            model_type='chatgpt',
                                            **kwargs
                                            ):
    # the reward computation function for persuation conversation
    # the following code is inspired from the PPDPP official implementation
    dial = ''
    for utt in simulated_conversation:
        if utt['role'] == 'user':
            role = 'Persuadee'
        else:
            role = 'Persuader'
        dial += f"{role}: {utt['content']}"
        dial += ". "

    # construct the message to prompt the llm
    messages = [{"role": "system",
                 "content": "Given a conversation between a Persuader and a Persuadee, please assess whether the Persuadee has agreed to donate to Save the Children charity after the conversation."},
                {"role": "user",
                 "content": f"""You can only reply with one of the following sentences: No, the Persuadee has not agreed to donate. Yes, the Persuadee has agreed to donate.\n\n
                        The following is the conversation between a Persuader and a Persuadee: 
                        Persuader: To donate to Save the Children, simply click on the "Donate Now" button on their website.
                        Persuadee: I'll go ahead and donate $25 to Save the Children.
                        Question: Has the Persuadee agreed to donate ?
                        Answer: Yes, the Persuadee has agreed to donate.
                        The following is the conversation between a Persuader and a Persuadee: 
                        Persuader: Your $25 donation will be securely processed on the Save the Children website.
                        Persuadee: I'll go ahead and donate $25 to Save the Children.
                        Question: Has the Persuadee agreed to donate ?
                        Answer: Yes, the Persuadee has agreed to donate.
                        The following is the conversation between a Persuader and a Persuadee: 
                        Persuader: To donate to Save the Children, simply visit their website at savethechildren.org.
                        Persuadee: I'll look into it and see if I can make a donation.
                        Question: Has the Persuadee agreed to donate ?
                        Answer: No, the Persuadee has not agreed to donate.
                        Following is the conversation between the Persuader and the Persuadee: {dial}. 
                        Question: Has the Persuadee agreed to donate? 
                        Answer: """
                }
    ]
    
    responses = []
    # prompt llm for n times
    if model_type == CHATGPT:
        for i in range(n):
            # calling the chat gpt
            response = chat_completion_with_backoff(
                model=MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            responses.append(response.choices[0]['message']['content'])
    # calling the llama 3
    else:
        no_think = "/no_think" if model_type == QWEN else ""
        for p in messages:
            for k in p.keys():
                if 'system' in p[k]:
                    p['content'] = no_think + p['content']    

        responses.extend(call_llm_model(messages, temperature, max_tokens, n_return_sequences=n, **kwargs))

    # convert the text-based assessment to scalar based assessment
    # processing the llm's outputs
    return responses


def get_toxicity_assessment_for_emotional_support(generated_system_utt):
    """
    method that compute the toxicity score for emotional support conversation
    :param generated_system_utt: the generated system utterance
    :return:
    """
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=PERSPECTIVE_API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    analyze_request = {
        'comment': {'text': generated_system_utt},
        'requestedAttributes': {'TOXICITY': {}}
    }

    response = client.comments().analyze(body=analyze_request).execute()
    toxicity_score = response['attributeScores']['TOXICITY']['summaryScore']['value']
    return toxicity_score


def get_user_sentiment_for_item_recommendation(generated_user_utterance):
    """
    method that compute the user sentiment for target-driven recommendation
    :param generated_user_utterance: the generated utterance of the user
    :return:
    """
    sentiment = sentiment_analysis(generated_user_utterance)
    return sentiment
