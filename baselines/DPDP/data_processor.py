from collections import defaultdict
import torch
from torch.nn.utils.rnn import pad_sequence
import re
import copy

from base.data_processor import DataProcessorForRecommendation, \
    DataProcessorForNegotiation, DataProcessorForEmotionalSupport
from base.torch_dataset import BaseTorchDataset
from utils.game import random_weights

from config.constants import USER_TOKEN, SYSTEM_TOKEN, GOAL_TOKEN, TOPIC_TOKEN, SEP_TOKEN, PATH_TOKEN, TARGET, \
    CONTEXT_TOKEN, IGNORE_INDEX, SELLER_TOKEN, BUYER_TOKEN, SEEKER_TOKEN, SUPPORTER_TOKEN

from utils.prompt import get_llm_based_assessment_for_negotiation, get_user_sentiment_for_item_recommendation

def generate_bins(low, high, n):
    if n <= 0:
        return []
    bin_width = (high - low) / n
    bins = [(low + i * bin_width, low + (i + 1) * bin_width) for i in range(n)]
    return bins


def generate_rewards_for_recommendation(system_response, user_response, target_item, goal):

    ## Recommendaion reward
    obj_reward = 0
    if 'recommendation' in goal:
        # a small reward for recommendation action
        # encouraging the model to make more recommendations
        obj_reward = 0.2
        # # compute the objective assessment
        # # check if the target item is in the generated system response
        if target_item.replace(" ", "").lower().strip() in system_response.replace(" ", "").lower().strip():
            # give a very right reward if the target item is recommended successfully
            obj_reward = 1.0
            o_done = 1

    # compute the objective assessment
    # check if the predicted topic is the target item
    # then we give a reward to the model (i.e the item freq)
    # if target_item.lower().strip() in topic.lower().strip():
    #     obj_reward = 1.0

    # get the user generated response
    # user_response = state['dialogue_context'][-1]['content']
    
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
    
    return obj_reward, user_reward

def generate_rewards_for_negotiation(instance, n = 10, model_type = 'llama3', max_tokens = 10):
    
    # generate the reward for offline reinforcement learning
    # the current dialogue context
    updated_dialogue_context = copy.deepcopy(instance['dialogue_context'])
    
    # the action
    # system's response and user response by the user
    action = instance['goal']
    system_response = instance['response']
    
    # offline reinforcement learning
    if 'usr_response' in instance:
        user_response = instance['usr_response']

        # print(action, system_response, user_response)

        # update the dialogue context
        updated_dialogue_context.extend(
            [{'role': 'assistant', 'content': system_response},
            {'role': 'user', 'content': user_response}
            ]
        )

        responses = get_llm_based_assessment_for_negotiation(simulated_conversation=updated_dialogue_context,
                                                            n = n,
                                                            temperature=1.1,
                                                            model_type=model_type,
                                                            max_tokens=max_tokens)
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
                reward = (deal_price - instance['task_background']['seller_price']) / (
                        instance['task_background']['buyer_price'] - instance['task_background']['seller_price'])
                rewards.append(reward)

        # deal rate
        neg_sr = sum(deals) / len(deals)

        # computing the price gain
        # extracting the price proposed by the system
        system_prices = re.findall(r"[-+]?\d*\.?\d+", system_response.replace(",", ""))

        # extracting the price proposed by the system
        if len(system_prices) > 0 and action != "inform":
            system_price = max(system_prices)
        else:
            system_price = instance['task_background']['seller_price']

        system_price = float(system_price)

        # encourage the system to gain more benefit
        # this reward is to encourage the model to propose beneficial price for its self.
        sl_ratio = (system_price - instance['task_background']['seller_price']) / (
                instance['task_background']['buyer_price'] - instance['task_background']['seller_price'])

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
        mid_price = (instance['task_background']['seller_price'] + instance['task_background']['buyer_price']) / 2

        # if the system price is equivalent to the mid price
        # we give the system a high fairness reward
        fairness = 0.5 - abs(system_price - mid_price) / (
                instance['task_background']['seller_price'] - instance['task_background']['buyer_price'])

        # clipping the values
        # if the fairness is larger than 0.5 then it is equivalent to 0.5
        # otherwise it equals to 0
        if fairness >= 0.5:
            fairness = 0.5
        elif fairness < -0.5:
            fairness = -0.5

        return sl_ratio, fairness, neg_sr
    # inference time, no user response
    else:
        raise Exception("Something is wrong here....")


class DPDPTorchDataset(BaseTorchDataset):

    def collate_fn(self, batch):
        """
        collate function that converts a batch of data features to batched tensor
        :param batch: a batch of data features
        :return:
        """
        # dialogue context
        input_features = defaultdict(list)
        labels = []
        next_input_features = defaultdict(list)

        # preference vectors
        rewards = []
        dones = []
        for instance in batch:
            input_features['input_ids'].append(instance['input_ids'])
            labels.append(instance['label'])
            rewards.append(instance['reward'])
            dones.append(instance['done'])

            # the feature of the next state
            if instance['next_input_ids'] is not None:
                next_input_features['input_ids'].append(instance['next_input_ids'])

        # padding the input features
        # for the current state
        input_features = self.tokenizer.pad(
            input_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_sequence_length
        )
        # convert features to torch tensors
        for k, v in input_features.items():
            if not isinstance(v, torch.Tensor):
                # input_features[k] = torch.as_tensor(v)
                input_features[k] = torch.as_tensor(v, device=self.device)

        # the label and the preference weights
        labels = torch.LongTensor(labels).to(self.device)
        rewards = torch.Tensor(rewards).to(self.device)
        dones = torch.Tensor(dones).to(self.device)

        if len(next_input_features) > 0:
            # padding the input features
            # for the next state
            next_input_features = self.tokenizer.pad(
                next_input_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
                max_length=self.max_sequence_length
            )
            # convert features to torch tensors
            for k, v in next_input_features.items():
                if not isinstance(v, torch.Tensor):
                    # next_input_features[k] = torch.as_tensor(v)
                    next_input_features[k] = torch.as_tensor(v, device=self.device)

            new_batch = {
                "context": input_features,
                "rewards": rewards,
                "next_state": next_input_features,
                "labels": labels,
                "done": dones
            }
        else:
            new_batch = {
                "context": input_features,
                "labels": labels,
            }
        return new_batch

    def preprocess_data(self, instances, convert_example_to_feature):
        """
        method that preprocess an data instances
        @param instances: an instance from the data
        @return: a processed instance which consists of input_ids and labels
        """
        processed_instances = []
        for instance in instances:
            if not self.is_gen:
                # data processor for policy/reward training
                state, action, reward, next_state, done = convert_example_to_feature(self.tokenizer, instance,
                                                                                        self.max_sequence_length,
                                                                                        self.goal2id,
                                                                                        self.n_objectives)
                new_instance = {
                    "input_ids": state,
                    "label": action,
                    "reward": reward,
                    "next_input_ids": next_state,
                    "done": done

                }
            else:
                input_ids, label = convert_example_to_feature(self.tokenizer, instance, self.max_sequence_length,
                                                              self.max_target_length, self.is_test)
                new_instance = {
                    "input_ids": input_ids,
                    "label": label,
                }
            processed_instances.append(new_instance)
        return processed_instances



class DPDPDataProcessorForRecommendation(DataProcessorForRecommendation):


    def construct_state(self, tokenizer, dialogue_context, target, target_goal, prev_paths, max_sequence_length=512):
        """
        method that construct the state
        :param tokenizer: the tokenizer
        :param dialogue_context: the given dialogue context
        :param prev_paths: the previous planned path
        :param max_sequence_length: the maximal number of tokens
        :return:
        """
        dialogue_str = ""
                
        for utt in dialogue_context:
            if utt['role'] == "user":
                dialogue_str += USER_TOKEN
            elif utt['role'] == 'assistant':
                dialogue_str += SYSTEM_TOKEN
            dialogue_str += utt['content']

        path_str = ""
        for goal in prev_paths:
            path_str += GOAL_TOKEN
            path_str += goal
            # path_str += TOPIC_TOKEN
            # path_str += topic
            path_str += SEP_TOKEN

        input_str = f"{PATH_TOKEN}: {path_str} {TARGET}: {target_goal} {target} {CONTEXT_TOKEN}: {dialogue_str}"
        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
        input_ids = input_ids[-(max_sequence_length - 2):]
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

        return input_ids

    def __call__(self, tokenizer, instance, max_sequence_length=512, action_to_id=None):
        """
        feature function for the BART policy model
        :param tokenizer: the huggingface tokenizer for the Bert model
        :param instance: an particular instance
        :param max_sequence_length: the maximum sequence length
        :param action_to_id: action mapping
        :return: a tokenized input sequence and its corresponding label
        """

        # current dialogue context
        # pretraining step
        if 'usr_response' in instance:
            
            # need to process the data
            if "processed_data" not in instance:
                dialogue_context = instance['dialogue_context']
                
                # previously planned goals
                prev_paths = instance['pre_goals']
                prev_topics = instance['pre_topics']
                
                # target topic and goal
                target = instance['task_background']['target_topic']
                target_goal = instance['task_background']['target_goal']
                            
                # construct the state
                state = self.construct_state(tokenizer, dialogue_context, target, target_goal, prev_paths, max_sequence_length)
                
                # construct the next state
                next_dialogue_context = copy.deepcopy(dialogue_context)
                next_dialogue_context.extend(
                    [
                        {'role': 'assistant', 'content': instance['response']},
                        {'role': 'user', 'content': instance['usr_response']}
                    ]
                )

                # update the next previous planned path
                next_prev_paths = copy.deepcopy(prev_paths)
                next_prev_paths.append(
                    instance['goal']
                )

                # next state
                next_state = self.construct_state(tokenizer, next_dialogue_context, target, target_goal, next_prev_paths, max_sequence_length)
                
                # done
                done = done = instance['done']
                
                # action mapping
                goals_to_ids, _ = action_to_id
                # we only predict the goal
                
                # action
                action = goals_to_ids[instance['goal']]
                
                # target item
                target_item = instance['task_background']['target_topic']
                            
                # rewards for recommendation
                # user sentiment and item frequency
                rewards = generate_rewards_for_recommendation(instance['response'], instance['usr_response'], target_item, instance['goal'])
            # the data has been processed already
            else:
                state, action, rewards, next_state, done = instance['processed_data']
    
        # rl finetuning step
        else:
            dialogue_context = instance['dialogue_context']
            prev_paths = instance['pre_goals']
            
            target = instance['task_background']['target_topic']
            target_goal = instance['task_background']['target_goal']
            
            state = self.construct_state(tokenizer, dialogue_context, target, target_goal, prev_paths, max_sequence_length)
            
            action = 0
            rewards = 0, 0, 0
            next_state = self.construct_state(tokenizer, dialogue_context, target, target_goal, prev_paths, max_sequence_length)
            done = 0

        return state, action, rewards, next_state, done


class DPDPDataProcessorForNegotiation(DataProcessorForNegotiation):


    def construct_state(self, tokenizer, dialogue_context, prev_paths, max_sequence_length=512):
        """
        method that construct the state
        :param tokenizer: the tokenizer
        :param dialogue_context: the given dialogue context
        :param prev_paths: the previous planned path
        :param max_sequence_length: the maximal number of tokens
        :return:
        """
        # processing the dialogue context
        dialogue_str = ""
        for utt in dialogue_context:
            if utt['role'] == "user":
                dialogue_str += SELLER_TOKEN
            elif utt['role'] == 'assistant':
                dialogue_str += BUYER_TOKEN
            dialogue_str += utt['content']

        # processing the previous planned path
        path_str = ""
        for goal in prev_paths:
            if isinstance(goal, tuple):
                goal = goal[0]
            path_str += GOAL_TOKEN
            path_str += goal
            path_str += SEP_TOKEN

        # convert features to token ids
        # the current state s
        input_str = f"{PATH_TOKEN}: {path_str} {CONTEXT_TOKEN}: {dialogue_str}"
        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
        input_ids = input_ids[-(max_sequence_length - 2):]
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

        return input_ids


    def __call__(self, tokenizer, instance, max_sequence_length=512, action_to_id=None, n_bins = 5):
        """
        feature function for the PPDPP model for the negotiation scenario
        :param tokenizer: a huggingface tokenizer
        :param instance: a input instance
        :param max_sequence_length: max sequence length
        :param action_to_id:
        :return:
        """
        # training time
        if 'usr_response' in instance:
            
            # processing the data
            if "processed_data" not in instance: 
                dialogue_context = instance['dialogue_context']
                prev_paths = instance['pre_goals']
                state = self.construct_state(tokenizer, dialogue_context, prev_paths, max_sequence_length)

                # the ground truth label
                # construct the action a
                buyer_price = instance['task_background']['buyer_price']
                seller_price = instance['task_background']['seller_price']

                # construct the action
                # extract the proposed price
                res = instance['response']
                prices = re.findall(r"[-+]?\d*\.?\d+", res.replace(",", ""))
                prices = [x for x in prices if float(x) >= buyer_price and float(x) <= seller_price]
                if len(prices) > 0:
                    proposed_price = max(prices)
                else:
                    proposed_price = buyer_price

                proposed_price = float(proposed_price)
                # computing the bin label
                # quantizing the price range into n bins
                bins = generate_bins(buyer_price, seller_price, n=n_bins)
                bin_label = 0
                for i, bin in enumerate(bins):
                    if proposed_price >= bin[0] and proposed_price <= bin[1]:
                        bin_label = i
                
                # construct the action
                action = action_to_id[(instance['goal'], bin_label)]

                # construct the rewards
                rewards = generate_rewards_for_negotiation(instance)
                
                # computing the next state s
                # update the next dialogue context
                next_dialogue_context = copy.deepcopy(dialogue_context)
                next_dialogue_context.extend(
                    [
                        {'role': 'assistant', 'content': instance['response']},
                        {'role': 'user', 'content': instance['usr_response']}
                    ]
                )

                # update the next previous planned path
                next_prev_paths = copy.deepcopy(prev_paths)
                next_prev_paths.append(
                    instance['goal']
                )

                next_state = self.construct_state(tokenizer, next_dialogue_context, next_prev_paths, max_sequence_length)
                
                done = instance['done']
            else:
                # compute the feature vector for the next state
                # calling this method recursively to process the features for the next dialogue state.
                # return the input_ids, label and next_ids
                state, action, rewards, next_state, done = instance['processed_data']
                
            return state, action, rewards, next_state, done
        # inference time
        else:
            
            dialogue_context = instance['dialogue_context']
            prev_paths = instance['pre_goals']
            state = self.construct_state(tokenizer, dialogue_context, prev_paths, max_sequence_length)
            
            action = 0
            rewards = 0, 0, 0
            next_state = self.construct_state(tokenizer, dialogue_context, prev_paths, max_sequence_length)
            done = 0
            
            return state, action, rewards, next_state, done


class DPDPDataProcessorForEmotionalSupport(DataProcessorForEmotionalSupport):

    def __call__(self, tokenizer, instance, max_sequence_length=512, action_to_id=None):
        """
        feature function for the PPDPP model for the negotiation scenario
        :param tokenizer: a huggingface tokenizer
        :param instance: a input instance
        :param max_sequence_length: max sequence length
        :param action_to_id:
        :return:
        """
        # the features that we consider.
        # dialogue context, previous goals, topics and the target item
        dialogue_context = instance['dialogue_context']
        prev_paths = instance['pre_goals']

        # processing the dialogue context
        dialogue_str = ""
        for utt in dialogue_context:
            if utt['role'] == "user":
                dialogue_str += SEEKER_TOKEN
            elif utt['role'] == 'assistant':
                dialogue_str += SUPPORTER_TOKEN
            dialogue_str += utt['content']

        # processing the previous planned path
        path_str = ""
        for goal in prev_paths:
            path_str += GOAL_TOKEN
            path_str += goal
            path_str += SEP_TOKEN

        # convert features to token ids
        input_str = f"{PATH_TOKEN}: {path_str} {CONTEXT_TOKEN}: {dialogue_str}"
        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
        input_ids = input_ids[-(max_sequence_length - 2):]
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

        # the ground truth label
        label = action_to_id[instance['goal']]

        # compute the feature vector for the next state
        # calling this method recursively to process the features for the next dialogue state.
        # return the input_ids, label and next_ids
        return input_ids, label


class DPDPTorchDatasetForRecommendation(DPDPTorchDataset):
    """
    PPDPP Torch dataset class for recommendation
    Just use it for coding convenience
    """
    def preprocess_data(self, instances, convert_example_to_feature):
        """
        method that preprocess an data instances
        @param instances: an instance from the data
        @return: a processed instance which consists of input_ids and labels
        """
        processed_instances = []
        for instance in instances:
            # data processor for policy/reward training
            state, action, reward, next_state, done = convert_example_to_feature(tokenizer = self.tokenizer, 
                                                                                instance = instance,
                                                                                max_sequence_length = self.max_sequence_length,
                                                                                action_to_id = self.goal2id,
                                                                                )
            new_instance = {
                "input_ids": state,
                "label": action,
                "reward": reward,
                "next_input_ids": next_state,
                "done": done

            }
            processed_instances.append(new_instance)
            
        return processed_instances


class DPDPTorchDatasetForNegotiation(DPDPTorchDataset):
    """
    PPDPP torch dataset for negotiation
    just use it for coding convenience
    """
    def preprocess_data(self, instances, convert_example_to_feature):
        """
        method that preprocess an data instances
        @param instances: an instance from the data
        @return: a processed instance which consists of input_ids and labels
        """
        processed_instances = []
        for instance in instances:
            # data processor for policy/reward training
            state, action, reward, next_state, done = convert_example_to_feature(tokenizer = self.tokenizer, 
                                                                                 instance = instance,
                                                                                 max_sequence_length = self.max_sequence_length,
                                                                                 action_to_id = self.goal2id,
                                                                                 )
            new_instance = {
                "input_ids": state,
                "label": action,
                "reward": reward,
                "next_input_ids": next_state,
                "done": done

            }
            processed_instances.append(new_instance)
        return processed_instances


class DPDPTorchDatasetForEmotionalSupport(DPDPTorchDataset):
    """
    PPDPP torch dataset for negotiation
    just use it for coding convenience
    """
    pass
