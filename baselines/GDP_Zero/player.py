import logging
import numpy as np

from typing import List, Tuple
import re

from utils.prompt import get_llm_based_assessment_for_negotiation, get_llm_based_assessment_for_recommendation, \
    get_llm_based_assessment_for_emotional_support

from config.constants import EMOTIONAL_SUPPORT, NEGOTIATION, RECOMMENDATION, PERSUATION, SL_RATIO, SUCCESS_RATE, AVG_TURN, FAIRNESS, INSPIRED, USER_REWARD, ITEM_FREQ
from utils.prompt import call_llm, get_user_sentiment_for_item_recommendation

logger = logging.getLogger(__name__)

def generate_bins(low, high, n):
    if n <= 0:
        return []
    bin_width = (high - low) / n
    bins = [(low + i * bin_width, low + (i + 1) * bin_width) for i in range(n)]
    return bins

class DialogPlanner(object):

    def __init__(self):
        pass

    def get_valid_moves(self, state):
        # 1 if the i-th dialog act is valid, 0 otherwise
        pass

    def predict(self, state) -> "Tuple[np.ndarray, float]":
        """
        function that predicts dialogue actions
        @param state: the current sate of the dialogue
        @return: list of actions corresponding to the current state
        """
        pass


class RTCPlayer(DialogPlanner):

    def __init__(self, dataset, policy_model, policy_tokenizer, goal2id, epsilon=1, max_sequence_length=512,
                 padding='max_length',
                 pad_to_multiple_of=True,
                 device=None,
                 use_demonstration=True,
                 n=5
                 ):
        """
        constructor for RTCPlayer. This player use RTCP as the prior policy model.
        @param kwargs:
        """
        super().__init__()
        self.dataset = dataset
        self.policy_model = policy_model
        self.policy_tokenizer = policy_tokenizer
        self.goal2id = goal2id
        self.dialog_acts = {v: k for k, v in self.goal2id.items()}
        self.max_sequence_length = max_sequence_length
        self.padding = padding
        self.pad_to_multiple_of = pad_to_multiple_of
        self.device = device
        self.epsilon = epsilon
        self.use_demonstration = use_demonstration
        self.n = n
        pass

    def get_valid_moves(self, state):
        """
        function that indentify valid moves.
        @param state: the curernt state of the dialogue.
        @return: a list contains identifications of valid moves
        """
        # every dialogue act in the action space is valid.
        return np.array([1 for _ in self.goal2id.keys()])

    def predict(self, state) -> "Tuple[np.ndarray, float]":
        """
        predict dialogue actions using the RTCP policy model.
        @param state: the current state of the conversation.
        @return: a tuple of actions prior probability corresponding to the current state
        """
        # predict a prior probability distribution over action space.
        prior = predict_action_rtcp_mcts(policy_model=self.policy_model,
                                         policy_tokenizer=self.policy_tokenizer, state=state, goal2id=self.goal2id,
                                         max_sequence_length=self.max_sequence_length, padding=self.padding,
                                         pad_to_multiple_of=self.pad_to_multiple_of, device=self.device)
        # compute the value of the current state with llm.
        v = self.heuristics(state)
        return prior, v

    def heuristics(self, state):
        """
        Function that return state values based on prompting and LLMs.
        @param state: the current state of the conversation
        @return: a list of estimated state values.
        """
        # using LLM-based target-driven assessment.
        # very expensive to evaluate during testing
        # rule-based state value function
        target_item = state['task_background']['target_topic']
        target_goal = state['task_background']['target_goal']
        if self.dataset == 'inspired':
            target_item = re.sub(r'\(\d+\)', '', target_item)

        check = False
        # compute the outcome of the conversation
        for utt in state['dialogue_context']:
            # check if the item appear in the user responses.
            if utt['role'] == 'assistant' and target_item.lower().strip() in utt['content'].lower().strip():
                check = True

        # if not check:
        #     # failed to recommend the target item.
        #     return -1

        # score = reward_func(conversations=state['dialogue_context'], target_topic=target_item, target_goal=target_goal)
        score = get_llm_based_assessment(
            target_topic=state['task_background']['target_topic'],
            simulated_conversation=state['dialogue_context'],
            demonstration=state['demonstration'] if self.use_demonstration else None,
            n=self.n
        )
        return score


class LLMPlayer(DialogPlanner):

    def __init__(self, game_config, action_mapping, model_config):
        """
        constructor for the LLM player class
        :param game_config: the name of the current dialogue game
        :param action_mapping: the action mapping which map goals to indices
        :param model_config: the prompt for the llm player
        """
        self.game_config = game_config
        self.game_name = self.game_config.name
        self.model_config = model_config
        self.prompt = model_config.prompt
        self.cot_prompt = model_config.cot_prompt
        self.goal2id = action_mapping
        self.model_type = self.model_config.model_type

        self.id2goal = {v: k for k, v in self.goal2id.items()}
        self.smoothing = 0.1
        self.n = 5
        pass

    def get_valid_moves(self, state):
        """
        return valid goals
        :param state: the current state
        :return:
        """
        return np.array([1 for _ in self.goal2id.keys()])

    def heuristics(self, state):
        """
        return the value of the given state
        :param state: the current state of the conversation
        :return:
        """
        simulated_conversation = state['dialogue_context']
        
        # base cases
        if len(simulated_conversation) == 2:
            return 0.0

        if self.game_name == NEGOTIATION:
            responses = get_llm_based_assessment_for_negotiation(simulated_conversation=simulated_conversation,
                                                                 n=5,
                                                                 temperature=1.1,
                                                                 max_tokens=20,
                                                                 model_type=self.model_type
                                                                 )

            deals = []
            rewards = []
            
            # compute fairness score
            # fairness score should be defined as
            for response in responses:
                # compute the neg_sr
                if 'have not' in response.lower():
                    # no deal
                    deals.append(0)
                elif 'have reached' in response.lower():
                    # there is a deal
                    deals.append(1)

                # collect the dealed price
                # and now we compute the fairness score
                prices = re.findall(r"[-+]?\d*\.?\d+", response.replace(",", ""))
                if len(prices) > 0:
                    deal_price = float(prices[0])
                    # compute the sale list ratio
                    reward = (deal_price - state['task_background']['seller_price']) / (
                            state['task_background']['buyer_price'] - state['task_background']['seller_price'])
                    rewards.append(reward)

            neg_sr = sum(deals) / len(deals)
            system_response = simulated_conversation[-2]['content']
            action = state['pre_goals'][-1]

            system_prices = re.findall(r"[-+]?\d*\.?\d+", system_response.replace(",", ""))
            # extracting the price proposed by the system
            
            if len(system_prices) > 0 and action != "inform":
                system_price = max(system_prices)
            else:
                system_price = state['task_background']['seller_price']

            system_price = float(system_price)
            # encourage the system to gain more benefit
            # this reward is to encourage the model to propose beneficial price for its self.
            
            sl_ratio = (system_price - state['task_background']['seller_price']) / (
                    state['task_background']['buyer_price'] - state['task_background']['seller_price'])

            # clipping the values
            # if the ratio is larger than 1 then it is equivalent to 1
            # otherwise it equals to 0
            if sl_ratio >= 1.0:
                sl_ratio = 1.0
                
            elif sl_ratio < -1.0:
                sl_ratio = -1.0

            # fairness
            # we compute the fairness score, which will be larger if the proposed price is close to the middle price
            # this should be conflicting to the sl_ratio price.
            mid_price = (state['task_background']['seller_price'] + state['task_background']['buyer_price']) / 2

            # if the system price is equivalent to the mid price
            # we give the system a high fairness reward
            fairness = 0.5 - abs(system_price - mid_price) / (
                    state['task_background']['seller_price'] - state['task_background']['buyer_price'])

            # clipping the values
            # if the fairness is larger than 0.5 then it is equivalent to 0.5
            # otherwise it equals to 0
            if fairness >= 0.5:
                fairness = 0.5
            elif fairness < -0.5:
                fairness = -0.5
            
            # heuristics value            
            # scalarization
            heuristic = 0
            if SL_RATIO in self.game_config.objectives:
                heuristic += self.model_config.objective_weight[0] * sl_ratio 
            # fairness
            if FAIRNESS in self.game_config.objectives:
                heuristic += self.model_config.objective_weight[1] * fairness 
            # SR
            if SUCCESS_RATE in self.game_config.objectives:
                heuristic += self.model_config.objective_weight[-1] * neg_sr
                
            print(heuristic)
            return heuristic
        
        # NOTE: the heuristics for recommendation
        elif self.game_name == RECOMMENDATION:
            
            system_response = simulated_conversation[-2]['content']
            goal = state['pre_goals'][-1]

            # the targeted item
            target_item = state['task_background']['target_topic']

            # objective reward
            # check if the target item appear in the generated system response.
            # if self.dataset_config.dataset_name == INSPIRED:
            #     target_item = re.sub(r'\(\d+\)', '', target_item)

            # construct the reward
            # the reward should contain multiple values corresponding to different objectives
            # the reward is in turn level
            # objective and subjective assessment.
            obj_reward = 0.0

            # add some negative reward if the conversation keeps going.
            avg_turn_reward = -0.1

            ## Recommendaion reward
            if 'recommendation' in goal:
                # a small reward for recommendation action
                # encouraging the model to make more recommendations
                obj_reward = 0.2
                # # compute the objective assessment
                # # check if the target item is in the generated system response
                if target_item.replace(" ", "").lower().strip() in system_response.replace(" ", "").lower().strip():
                    # give a very right reward if the target item is recommended successfully
                    obj_reward = 1.0

            # compute the objective assessment
            # check if the predicted topic is the target item
            # then we give a reward to the model (i.e the item freq)
            # if target_item.lower().strip() in topic.lower().strip():
            #     obj_reward = 1.0

            # get the user generated response
            user_response = state['dialogue_context'][-1]['content']

            # compute the user sentiment using a pretrained sentiment analysis model
            user_sentiment_results = get_user_sentiment_for_item_recommendation(user_response)[0]

            sentiment_label, sentiment_score = user_sentiment_results['label'], user_sentiment_results['score']

            # positive sentiment
            # a high reward
            user_reward = 0
            if sentiment_label == "LABEL_2":
                user_reward = sentiment_score
            # neutral
            # zero reward
            elif sentiment_label == "LABEL_1":
                user_reward = 0
            # negative sentiment
            # negative reward
            elif sentiment_label == "LABEL_0":
                user_reward = - sentiment_score

            # vector-valued reward function
            reward = []

            heuristic = 0
            # if we use subjective reward
            if USER_REWARD in self.game_config.objectives:
                heuristic += self.model_config.objective_weight[0] * user_reward
            # if we use objective reward
            if ITEM_FREQ in self.game_config.objectives:
                heuristic += self.model_config.objective_weight[1] * obj_reward
            # if we use avg turn.
            if AVG_TURN in self.game_config.objectives:
                heuristic += self.model_config.objective_weight[-1] * avg_turn_reward

            print(heuristic)
            return heuristic

        # NOTE:  the heuristics for emotional support should be modified
        elif self.game_name == EMOTIONAL_SUPPORT:
            responses = get_llm_based_assessment_for_emotional_support(state,
                                                                       simulated_conversation=simulated_conversation,
                                                                       n=5,
                                                                       temperature=1.1,
                                                                       max_tokens=20,
                                                                       model_type=self.model_type
                                                                       )

            # used to compute the es_sr
            # indicate whether the supporter solved the seeker problem.
            rewards = []
            for response in responses:
                for key in self.game_config.reward_dict:
                    if key in response.lower():
                        rewards.append(self.game_config.reward_dict[key])
                        break
            # compute the emotional support sr
            if len(rewards) == 0:
                score = 0
            else:
                score = sum(rewards) / len(rewards)
        elif self.game_name == PERSUATION:
            # Persuasion uses preference pairs; keep heuristic neutral to allow search to proceed.
            score = 0.0
        else:
            raise Exception('Something is wrong here ....')
        return score

    def _get_generated_da(self, responses, state):
        """
        method that processes the outputs from the responses
        :param responses: the responses from the llm
        :return:
        """

        if self.game_config.name == NEGOTIATION:
            seller_price = state['task_background']['seller_price']
            buyer_price = state['task_background']['buyer_price']
            
            processed_responses = []
            for action in responses:
                # extract the proposed price
                proposed_prices = re.findall(r"[-+]?\d*\.?\d+", action.replace(",", ""))
                if len(proposed_prices) > 0:
                    system_price = max(proposed_prices)
                else:
                    system_price = buyer_price
                    
                # convert the proposed price to bin number
                proposed_price = float(system_price)
                
                # computing the bin label
                # quantizing the price range into n bins
                bins = generate_bins(buyer_price, seller_price, n=self.model_config.n_topics)
                bin_label = 0
                for i, bin in enumerate(bins):
                    if proposed_price >= bin[0] and proposed_price <= bin[1]:
                        bin_label = i
                                
                if '\"' in action:
                    action = action.replace('\"', '').strip()
                if "." in action:
                    action = action.replace('.', '').strip()

                try:
                    action = self.model_config.action_mapping[action]
                except:
                    # handling exception cases
                    temp_action = "None"
                    for k, v in self.model_config.action_mapping.items():
                        if k.lower() in action.lower():
                            temp_action = v
                            
                    # Heuristic approaches
                    # the better solution is to obtain more information
                    if temp_action == "None":
                        if self.game_config.name == NEGOTIATION:
                            temp_action = "inquire"
                        elif self.game_config.name == EMOTIONAL_SUPPORT:
                            temp_action = "Question"
                    action = temp_action

                processed_responses.append((action, bin_label))
        
        elif self.game_config.name == RECOMMENDATION:
            processed_responses = []
            for res in responses:
                # extract the proposed price
                actions = re.findall(r'"([^"]*)"', res.replace(",", ""))
                if len(actions) > 0:
                    action = actions[0]
                else:
                    # heuristics
                    action = "Question & Answer"
        
                if '\"' in action:
                    action = action.replace('\"', '').strip()
                if "." in action:
                    action = action.replace('.', '').strip()
                
                action = action.strip()
                try:
                    action = self.model_config.action_mapping[action]
                except:
                    # handling exception cases
                    temp_action = "None"
                    for k, v in self.model_config.action_mapping.items():
                        if k.lower() in action.lower():
                            temp_action = v
                    # Heuristic approaches
                    # the better solution is to obtain more information
                    if temp_action == "None":
                        if self.model_config.domain in ["movie", "music"]:
                            temp_action = "Q&A"
                        elif self.model_config.domain in ["poi"]:
                            temp_action = "Ask about weather"
                    action = temp_action
                processed_responses.append(action)

        elif self.game_config.name == PERSUATION:
            processed_responses = []
            for res in responses:
                # Expect goal/dialog-act tokens like [greeting] or raw labels; extract best guess.
                actions = re.findall(r'\[([^\]]+)\]', res)
                action = actions[0] if len(actions) > 0 else res.strip()
                action = action.strip().lower()

                matched = None
                for k, v in self.model_config.action_mapping.items():
                    if k.lower() == action or k.lower() in action:
                        matched = v
                        break
                if matched is None:
                    # heuristic default: pick a neutral/first action to keep search going
                    matched = self.model_config.action_mapping.get("question", None)
                    if matched is None and len(self.model_config.action_mapping) > 0:
                        matched = list(self.model_config.action_mapping.values())[0]
                processed_responses.append(matched)
        else:
            processed_responses = []
                
        return processed_responses

    def predict(self, state) -> "Tuple[np.ndarray, float]":
        """
        the method that predicts the goal and its value
        :param state:
        :return:
        """
        # predict a prior probability distribution over action space.
        # test k times and compute prob. See num_return_sequences in the API
        # the value would be our objective function
        dialogue = ''
        for utt in state['dialogue_context']:
            dialogue += f"{utt['role']}: {utt['content']} "

        # predict dialogue strategies
        if self.game_config.name == RECOMMENDATION:
            prompt = [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": self.cot_prompt.format(self.model_config.domain2strategies[self.model_config.domain], dialogue)}
            ]
            print(prompt)
        else:
            prompt = [
                {"role": "system", "content": self.prompt},
                {"role": "user", "content": self.cot_prompt.format(dialogue)}
            ] 

        # produce the responses with the llm
        responses = call_llm(prompt=prompt,
                             n=5,
                             temperature=1.1,
                             max_token=30,
                             model_type=self.model_type
                             )
        
        sampled_das = self._get_generated_da(responses, state)

        # convert to prob distribution
        prob = np.zeros(len(self.goal2id))
        prob += self.smoothing
        for da in sampled_das:
            if da not in self.goal2id:
                continue
            prob[self.goal2id[da]] += 1
        prob /= prob.sum()
        
        # compute the value of the current state with llm.
        v = self.heuristics(state)
        return prob, v
