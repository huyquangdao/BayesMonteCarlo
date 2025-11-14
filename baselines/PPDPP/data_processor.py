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


class PPDPPDataProcessorForRecommendation(DataProcessorForRecommendation):

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

        input_str = f"{PATH_TOKEN}: {path_str} {TARGET}: {target_goal} {target} {CONTEXT_TOKEN}: {dialogue_str}"
        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
        input_ids = input_ids[-(max_sequence_length - 2):]
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

        goals_to_ids, _ = action_to_id

        # we only predict the goal
        label = goals_to_ids[instance['goal']]
        return input_ids, label


class PPDPPDataProcessorForNegotiation(DataProcessorForNegotiation):

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
        input_str = f"{PATH_TOKEN}: {path_str} {CONTEXT_TOKEN}: {dialogue_str}"
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

        return input_ids, label


class PPDPPDataProcessorForEmotionalSupport(DataProcessorForEmotionalSupport):

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


class PPDPPDataProcessorForPersuation(DataProcessorForPersuation):

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


class PreferencePPDPPDataProcessorForRecommendation(PPDPPDataProcessorForRecommendation):

    def __call__(self, tokenizer, instance, max_sequence_length=512, action_to_id=None, n_objectives=3):
        """
        feature function for the preference PPDPP model
        :param tokenizer: a huggingface tokenizer
        :param instance: a input instance
        :param max_sequence_length: max sequence length
        :param action_to_id:
        :param n_objectives:
        :return:
        """
        # compute the feature vector for the current state
        input_ids, label = super().__call__(tokenizer, instance, max_sequence_length, action_to_id)
        if 'next_state' in instance:
            next_ids, _ = super().__call__(tokenizer, instance['next_state'], max_sequence_length, action_to_id)
        else:
            next_ids = None
        # using a predefine weight vector
        if 'w' in instance:
            w = instance['w']
        # sampled the weight here
        else:
            w = random_weights(dim=n_objectives)
        return input_ids, label, w, next_ids


class PPDPPDataProcessorForPreferenceEstimation(DataProcessorForPreferenceEstimation):

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


class PPDPPTorchDatasetForRecommendation(BaseTorchDataset):
    """
    PPDPP Torch dataset class for recommendation
    Just use it for coding convenience
    """
    pass


class PPDPPTorchDatasetForPreferenceEstimation(BaseTorchDataset):
    """
    PPDPP torch dataset for preference estimation
    just use for coding convenience
    """
    pass


class PPDPPTorchDatasetForNegotiation(BaseTorchDataset):
    """
    PPDPP torch dataset for negotiation
    just use it for coding convenience
    """
    pass


class PPDPPTorchDatasetForEmotionalSupport(BaseTorchDataset):
    """
    PPDPP torch dataset for emotional support
    just use it for coding convenience
    """
    pass


class PPDPPTorchDatasetForPersuation(BaseTorchDataset):
    """
    PPDPP torch dataset for persuation conversation
    just use it for coding convenience
    """
    pass

class PreferencePPDPPTorchDatasetForRecommendation(BaseTorchDataset):

    def __init__(self, n_objectives, tokenizer, instances, goal2id=None, max_sequence_length=512, padding='max_length',
                 pad_to_multiple_of=True, device=None, convert_example_to_feature=None, max_target_length=50,
                 is_test=False, is_gen=False, is_preference=False):
        """
        constructor for class preference PPDPP model
        :param n_objectives: number of objectives
        :param tokenizer: a huggingface tokenizer
        :param instances: a list of instances
        :param goal2id: a dictionary map goals to ids
        :param max_sequence_length: maximal number of tokens in the input sequence
        :param padding: padding type
        :param pad_to_multiple_of:
        :param device:
        :param convert_example_to_feature:
        :param max_target_length:
        :param is_test:
        :param is_gen:
        :param is_preference:
        """
        self.n_objectives = n_objectives
        super().__init__(tokenizer, instances, goal2id, max_sequence_length, padding, pad_to_multiple_of, device,
                         convert_example_to_feature, max_target_length, is_test, is_gen)

    def collate_fn(self, batch):
        """
        collate function that converts a batch of data features to batched tensor
        :param batch: a batch of data features
        :return:
        """
        input_features = defaultdict(list)
        labels = []
        weights = []
        next_input_features = defaultdict(list)
        for instance in batch:
            input_features['input_ids'].append(instance['input_ids'])
            labels.append(instance['label'])
            weights.append(instance['w'])

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

        # labels for response generation task
        if self.is_gen:
            labels = pad_sequence(
                [torch.tensor(label, dtype=torch.long) for label in labels],
                batch_first=True, padding_value=IGNORE_INDEX)

            # labels = labels.to(self.device)
        # labels for goal prediction task
        else:
            # labels for policy training
            # labels = torch.LongTensor(labels)
            # # the preference vector
            # weights = torch.Tensor(weights)
            # # labels for policy training
            labels = torch.LongTensor(labels).to(self.device)
            # the preference vector
            weights = torch.Tensor(weights).to(self.device)

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
                "next_state": next_input_features,
                "labels": labels,
                "w": weights
            }
        else:
            new_batch = {
                "context": input_features,
                "labels": labels,
                "w": weights
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
                input_ids, label, w, next_ids = convert_example_to_feature(self.tokenizer, instance,
                                                                           self.max_sequence_length,
                                                                           self.goal2id, self.n_objectives)

                if next_ids is not None:
                    new_instance = {
                        "input_ids": input_ids,
                        "label": label,
                        "w": w,
                        "next_input_ids": next_ids

                    }
                else:
                    new_instance = {
                        "input_ids": input_ids,
                        "label": label,
                        "w": w,
                        "next_input_ids": next_ids
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
