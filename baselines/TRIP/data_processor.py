from collections import defaultdict
import torch
from torch.nn.utils.rnn import pad_sequence
import re

from base.data_processor import DataProcessorForRecommendation, DataProcessorForPreferenceEstimation, \
    DataProcessorForNegotiation, DataProcessorForEmotionalSupport, DataProcessorForPersuation

from base.torch_dataset import BaseTorchDataset
from utils.game import random_weights
from config.constants import USER_TOKEN, SYSTEM_TOKEN, GOAL_TOKEN, TOPIC_TOKEN, SEP_TOKEN, PATH_TOKEN, TARGET, \
    CONTEXT_TOKEN, IGNORE_INDEX, SELLER_TOKEN, BUYER_TOKEN, SEEKER_TOKEN, SUPPORTER_TOKEN, PERSUADER_TOKEN, PERSUADEE_TOKEN


def generate_bins(low, high, n):
    if n <= 0:
        return []
    bin_width = (high - low) / n
    bins = [(low + i * bin_width, low + (i + 1) * bin_width) for i in range(n)]
    return bins


class TRIPDataProcessorForRecommendation(DataProcessorForRecommendation):

    def __call__(self, tokenizer, instance, max_sequence_length=512, action_to_id=None, **kwargs):
        """
        feature function for the BART policy model
        :param tokenizer: the huggingface tokenizer for the Bert model
        :param instance: an particular instance
        :param max_sequence_length: the maximum sequence length
        :param action_to_id: action mapping
        :return: a tokenized input sequence and its corresponding label
        """
        dialogue_context = instance['dialogue_context']
        prev_paths = instance['pre_goals']
        prev_topics = instance['pre_topics']
        
        target = instance['task_background']['target_topic']
        target_goal = instance['task_background']['target_goal']
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
        
        # convert features to token ids
        # if the mental states are available
        if "mental_states" not in instance:
            input_str = f"{PATH_TOKEN}: {path_str} {TARGET}: {target_goal} {target} {CONTEXT_TOKEN}: {dialogue_str}"
        else:
            mental_states = instance["mental_states"]
            input_str = f"{mental_states} {PATH_TOKEN}: {path_str} {TARGET}: {target_goal} {target} {CONTEXT_TOKEN}: {dialogue_str}"
        
        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
        input_ids = input_ids[-(max_sequence_length - 2):]
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

        goals_to_ids, _ = action_to_id
        
        # we only predict the goal
        label = goals_to_ids[instance['goal']]
        return input_ids, label


class TRIPDataProcessorForNegotiation(DataProcessorForNegotiation):

    def __call__(self, tokenizer, instance, max_sequence_length=512, action_to_id=None, n_bins = 5, is_so_game = True):
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
        # if the mental states are available
        if "mental_states" not in instance:
            input_str = f"{PATH_TOKEN}: {path_str} {CONTEXT_TOKEN}: {dialogue_str}"
        else:
            mental_states = instance["mental_states"]
            input_str = f"{PATH_TOKEN}: {path_str} {CONTEXT_TOKEN}: {dialogue_str} {mental_states}" 

        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
        input_ids = input_ids[-(max_sequence_length - 2):]
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

        # the ground truth label
        buyer_price = instance['task_background']['buyer_price']
        seller_price = instance['task_background']['seller_price']

        # multi objective game
        if not is_so_game:
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

            label = action_to_id[(instance['goal'], bin_label)]
        # single objective game
        else:
            label = action_to_id[instance['goal']]

        # compute the feature vector for the next state
        # calling this method recursively to process the features for the next dialogue state.
        # return the input_ids, label and next_ids
        return input_ids, label


class TRIPDataProcessorForEmotionalSupport(DataProcessorForEmotionalSupport):

    def __call__(self, tokenizer, instance, max_sequence_length=512, action_to_id=None, **kwargs):
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



class TRIPDataProcessorForPersuation(DataProcessorForPersuation):

    def __call__(self, tokenizer, instance, max_sequence_length=512, action_to_id=None, **kwargs):
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
                dialogue_str += PERSUADEE_TOKEN
            elif utt['role'] == 'assistant':
                dialogue_str += PERSUADER_TOKEN
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


class TRIPDataProcessorForPreferenceEstimation(DataProcessorForPreferenceEstimation):

    def __call__(self, tokenizer, instance, max_sequence_length, is_test=False):
        """
        feature function for PPDPP preference estimation
        :param tokenizer: the huggingface tokenizer
        :param instance: a string which is the user profile description
        :param max_sequence_length: the maximal number of tokens in the input sequence.
        :return: a preprocessed input
        """
        # each instance contains a input string which is the user profile description
        # and its corresponding reward vector
        if not is_test:
            input_str, reward_vector = instance
        else:
            # inference time
            # reward vector is a placeholder used for coding convenience
            input_str = instance
            reward_vector = [0]
        # tokenizing and converting the input sequence into a sequence of token ids
        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
        input_ids = input_ids[-(max_sequence_length - 2):]
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]
        return input_ids, reward_vector


class TRIPTorchDatasetForRecommendation(BaseTorchDataset):
    """
    PPDPP Torch dataset class for recommendation
    Just use it for coding convenience
    """
    pass


class TRIPTorchDatasetForPreferenceEstimation(BaseTorchDataset):
    """
    PPDPP torch dataset for preference estimation
    just use for coding convenience
    """
    pass


class TRIPTorchDatasetForNegotiation(BaseTorchDataset):
    """
    PPDPP torch dataset for negotiation
    just use it for coding convenience
    """
    pass


class TRIPTorchDatasetForEmotionalSupport(BaseTorchDataset):
    """
    PPDPP torch dataset for emotional support conversation
    just use it for coding convenience
    """
    pass


class TRIPTorchDatasetForPersuation(BaseTorchDataset):
    """
    PPDPP torch dataset for persuation conversation
    just use it for coding convenience
    """
    pass
