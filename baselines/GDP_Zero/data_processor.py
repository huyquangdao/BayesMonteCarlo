from collections import defaultdict
import torch

from base.data_processor import DataProcessorForRecommendation, DataProcessorForNegotiation, \
    DataProcessorForEmotionalSupport
from base.torch_dataset import BaseTorchDataset
from config.constants import USER_TOKEN, SYSTEM_TOKEN, GOAL_TOKEN, TOPIC_TOKEN, SEP_TOKEN, PATH_TOKEN, TARGET, \
    CONTEXT_TOKEN, IGNORE_INDEX, SEEKER_TOKEN, SUPPORTER_TOKEN, BUYER_TOKEN, SELLER_TOKEN


class GDPZeroDataProcessorForRecommendation(DataProcessorForRecommendation):

    def __call__(self, tokenizer, instance, max_sequence_length=512, action_to_id=None):
        """
        feature function for the BART policy model
        :param tokenizer: the huggingface tokenizer for the Bert model
        :param instance: an particular instance
        :param max_sequence_length: the maximum sequence length
        :param action_to_id: action mapping
        :return: a tokenized input sequence and its corresponding label
        """
        dialogue_context = instance['dialogue_context']
        prev_goals = instance['pre_goals']
        prev_topics = instance['pre_topics']
        target_item = instance['task_background']['target_topic']

        # construct the context string
        input_str = ""
        # loop over the dialogue context
        idx = 0
        for _, utt in enumerate(dialogue_context):
            if utt['role'] == "user":
                input_str += USER_TOKEN + " "
            elif utt['role'] == 'assistant':
                input_str += GOAL_TOKEN + " "
                input_str += prev_goals[idx] + " "
                input_str += SYSTEM_TOKEN + " "
                idx += 1
            input_str += utt['content']
        # the context string for RTCP
        input_str = f"{input_str} {TARGET} {target_item} {GOAL_TOKEN}"
        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
        input_ids = input_ids[-(max_sequence_length - 2):]
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

        # construct the previous planned path
        path_string = ""
        for goal, topic in list(zip(prev_goals, prev_topics)):
            path_string += F"{GOAL_TOKEN} {goal} {TOPIC_TOKEN} {topic} {SEP_TOKEN} "

        path_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(path_string))
        path_ids = path_ids[-(max_sequence_length - 2):]
        path_ids = [tokenizer.cls_token_id] + path_ids + [tokenizer.sep_token_id]

        # make sure we have the goal and topic mapping
        assert len(action_to_id) == 2
        goal2id, topic2id = action_to_id

        # if not inference time.
        if goal2id is not None and topic2id is not None:
            label_goal, label_topic = goal2id[instance['goal']], topic2id[instance['topic']]
        else:
            label_goal, label_topic = None, None

        return input_ids, path_ids, label_goal, label_topic


class GDPZeroDataProcessorForNegotiation(DataProcessorForNegotiation):

    def __call__(self, tokenizer, instance, max_sequence_length=512, action_to_id=None):
        """
        feature function for the RTCP policy model in the negotiation scenario
        :param tokenizer: the huggingface tokenizer for the Bert model
        :param instance: an particular instance
        :param max_sequence_length: the maximum sequence length
        :param action_to_id: action mapping
        :return: a tokenized input sequence and its corresponding label
        """
        dialogue_context = instance['dialogue_context']
        prev_goals = instance['pre_goals']

        # construct the context string
        input_str = ""
        # loop over the dialogue context
        idx = 0
        for _, utt in enumerate(dialogue_context):
            if utt['role'] == "user":
                input_str += SELLER_TOKEN + " "
            elif utt['role'] == 'assistant':
                input_str += GOAL_TOKEN + " "
                input_str += prev_goals[idx] + " "
                input_str += BUYER_TOKEN + " "
                idx += 1
            input_str += utt['content']
        # the context string for RTCP
        input_str = f"{input_str} {TARGET} {GOAL_TOKEN}"
        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
        input_ids = input_ids[-(max_sequence_length - 2):]
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

        # construct the previous planned path
        path_string = ""
        for goal in prev_goals:
            path_string += F"{GOAL_TOKEN} {goal} {SEP_TOKEN} "

        path_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(path_string))
        path_ids = path_ids[-(max_sequence_length - 2):]
        path_ids = [tokenizer.cls_token_id] + path_ids + [tokenizer.sep_token_id]

        # make sure we have the goal and topic mapping
        goal2id = action_to_id

        # if not inference time.
        if goal2id is not None:
            label_goal = goal2id[instance['goal']]
        else:
            label_goal = None

        return input_ids, path_ids, label_goal


class GDPZeroDataProcessorForEmotionalSupport(DataProcessorForEmotionalSupport):

    def __call__(self, tokenizer, instance, max_sequence_length=512, action_to_id=None):
        """
        feature function for the RTCP policy model in the emotional support conversation
        :param tokenizer: the huggingface tokenizer for the Bert model
        :param instance: an particular instance
        :param max_sequence_length: the maximum sequence length
        :param action_to_id: action mapping
        :return: a tokenized input sequence and its corresponding label
        """
        dialogue_context = instance['dialogue_context']
        prev_goals = instance['pre_goals']

        # construct the context string
        input_str = ""
        # loop over the dialogue context
        idx = 0
        for _, utt in enumerate(dialogue_context):
            if utt['role'] == "user":
                input_str += SEEKER_TOKEN + " "
            elif utt['role'] == 'assistant':
                input_str += GOAL_TOKEN + " "
                input_str += prev_goals[idx] + " "
                input_str += SUPPORTER_TOKEN + " "
                idx += 1
            input_str += utt['content']
        # the context string for RTCP
        input_str = f"{input_str} {TARGET} {GOAL_TOKEN}"
        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
        input_ids = input_ids[-(max_sequence_length - 2):]
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

        # construct the previous planned path
        path_string = ""
        for goal in prev_goals:
            path_string += F"{GOAL_TOKEN} {goal} {SEP_TOKEN} "

        path_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(path_string))
        path_ids = path_ids[-(max_sequence_length - 2):]
        path_ids = [tokenizer.cls_token_id] + path_ids + [tokenizer.sep_token_id]

        # make sure we have the goal and topic mapping
        goal2id = action_to_id

        # if not inference time.
        if goal2id is not None:
            label_goal = goal2id[instance['goal']]
        else:
            label_goal = None

        return input_ids, path_ids, label_goal


class GDPZeroTorchDatasetForRecommendation(BaseTorchDataset):
    """
    class RTCP torch dataset for the recommendation scenario
    """

    def preprocess_data(self, instances, convert_example_to_feature):
        """
        method that preprocess an data instances
        @param instances: an instance from the data
        @return: a processed instance which consists of input_ids and labels
        """
        processed_instances = []
        for instance in instances:
            # data processor for policy training
            context_ids, path_ids, label_goal, label_topic = convert_example_to_feature(self.tokenizer, instance,
                                                                                        self.max_sequence_length,
                                                                                        self.goal2id)

            new_instance = {
                "context_ids": context_ids,
                "path_ids": path_ids,
                "label_goal": label_goal,
                "label_topic": label_topic

            }
            processed_instances.append(new_instance)
        return processed_instances

    def collate_fn(self, batch):
        """
        collate function for RTCP model in the recommendation scenario
        :param batch:
        :return:
        """
        context_input_features = defaultdict(list)
        path_input_features = defaultdict(list)
        labels_goal = []
        labels_topic = []
        for instance in batch:
            context_input_features['input_ids'].append(instance['context_ids'])
            path_input_features['input_ids'].append(instance['path_ids'])
            labels_goal.append(instance['label_goal'])
            labels_topic.append(instance['label_topic'])

        # padding the context features
        context_input_features = self.tokenizer.pad(
            context_input_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_sequence_length
        )
        # convert features to torch tensors
        for k, v in context_input_features.items():
            if not isinstance(v, torch.Tensor):
                context_input_features[k] = torch.as_tensor(v, device=self.device)

        # padding the path features
        path_input_features = self.tokenizer.pad(
            path_input_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_sequence_length
        )
        # convert features to torch tensors
        for k, v in path_input_features.items():
            if not isinstance(v, torch.Tensor):
                path_input_features[k] = torch.as_tensor(v, device=self.device)

        labels_goal = torch.LongTensor(labels_goal).to(self.device)
        labels_topic = torch.LongTensor(labels_topic).to(self.device)

        new_batch = {
            "context": context_input_features,
            "path": path_input_features,
            "labels_goal": labels_goal,
            "labels_topic": labels_topic
        }

        return new_batch


class GDPZeroTorchDatasetForNegotiation(BaseTorchDataset):
    """
    class RTCP torch dataset for the negotiation scenario
    """

    def preprocess_data(self, instances, convert_example_to_feature):
        """
        method that preprocess an data instances
        @param instances: an instance from the data
        @return: a processed instance which consists of input_ids and labels
        """
        processed_instances = []
        for instance in instances:
            # data processor for policy training
            context_ids, path_ids, label_goal = convert_example_to_feature(self.tokenizer, instance,
                                                                           self.max_sequence_length,
                                                                           self.goal2id)

            new_instance = {
                "context_ids": context_ids,
                "path_ids": path_ids,
                "label_goal": label_goal,
            }
            processed_instances.append(new_instance)
        return processed_instances

    def collate_fn(self, batch):
        """
        collate function for RTCP model in the recommendation scenario
        :param batch:
        :return:
        """
        context_input_features = defaultdict(list)
        path_input_features = defaultdict(list)
        labels_goal = []
        for instance in batch:
            context_input_features['input_ids'].append(instance['context_ids'])
            path_input_features['input_ids'].append(instance['path_ids'])
            labels_goal.append(instance['label_goal'])

        # padding the context features
        context_input_features = self.tokenizer.pad(
            context_input_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_sequence_length
        )
        # convert features to torch tensors
        for k, v in context_input_features.items():
            if not isinstance(v, torch.Tensor):
                context_input_features[k] = torch.as_tensor(v, device=self.device)

        # padding the path features
        path_input_features = self.tokenizer.pad(
            path_input_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_sequence_length
        )
        # convert features to torch tensors
        for k, v in path_input_features.items():
            if not isinstance(v, torch.Tensor):
                path_input_features[k] = torch.as_tensor(v, device=self.device)

        labels_goal = torch.LongTensor(labels_goal).to(self.device)

        new_batch = {
            "context": context_input_features,
            "path": path_input_features,
            "labels_goal": labels_goal,
        }

        return new_batch


class GDPZeroTorchDatasetForEmotionalSupport(BaseTorchDataset):
    """
    class RTCP torch dataset for the negotiation scenario
    """

    def preprocess_data(self, instances, convert_example_to_feature):
        """
        method that preprocess an data instances
        @param instances: an instance from the data
        @return: a processed instance which consists of input_ids and labels
        """
        processed_instances = []
        for instance in instances:
            # data processor for policy training
            context_ids, path_ids, label_goal = convert_example_to_feature(self.tokenizer, instance,
                                                                           self.max_sequence_length,
                                                                           self.goal2id)

            new_instance = {
                "context_ids": context_ids,
                "path_ids": path_ids,
                "label_goal": label_goal,
            }
            processed_instances.append(new_instance)
        return processed_instances

    def collate_fn(self, batch):
        """
        collate function for RTCP model in the recommendation scenario
        :param batch:
        :return:
        """
        context_input_features = defaultdict(list)
        path_input_features = defaultdict(list)
        labels_goal = []
        for instance in batch:
            context_input_features['input_ids'].append(instance['context_ids'])
            path_input_features['input_ids'].append(instance['path_ids'])
            labels_goal.append(instance['label_goal'])

        # padding the context features
        context_input_features = self.tokenizer.pad(
            context_input_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_sequence_length
        )
        # convert features to torch tensors
        for k, v in context_input_features.items():
            if not isinstance(v, torch.Tensor):
                context_input_features[k] = torch.as_tensor(v, device=self.device)

        # padding the path features
        path_input_features = self.tokenizer.pad(
            path_input_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_sequence_length
        )
        # convert features to torch tensors
        for k, v in path_input_features.items():
            if not isinstance(v, torch.Tensor):
                path_input_features[k] = torch.as_tensor(v, device=self.device)

        labels_goal = torch.LongTensor(labels_goal).to(self.device)

        new_batch = {
            "context": context_input_features,
            "path": path_input_features,
            "labels_goal": labels_goal,
        }

        return new_batch
