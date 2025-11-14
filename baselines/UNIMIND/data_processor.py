from collections import defaultdict

import torch
from torch.nn.utils.rnn import pad_sequence

from base.data_processor import DataProcessorForRecommendation

from base.torch_dataset import BaseTorchDataset
from config.constants import USER_TOKEN, SYSTEM_TOKEN, GOAL_TOKEN, TOPIC_TOKEN, SEP_TOKEN, TARGET, IGNORE_INDEX


def convert_example_to_feature_for_unimind_goal_prediction(tokenizer, instance, max_sequence_length=512,
                                                           max_target_length=50, is_test=False):
    """
    function that creates the input example for unimind goal prediction task
    @param tokenizer: hugginface tokenizer
    @param instance: the input instance
    @param max_sequence_length: maximum number of tokens in the input sequence
    @param max_target_length: maximum number of tokens in the label
    @return: input sequence for the unimind's goal prediction task.
    """
    dialogue_context = instance['dialogue_context']
    prev_goals = instance['pre_goals']
    target_item = instance['task_background']['target_topic']

    # Example of the input of unimind goal prediction
    # “[user] Who is the star of the movie < stolen life >? [goal] QA [system] It is Xun Zhou.[user]
    # She is my goddess.[goal] Chit - chat about Star[system] You have a good taste.She is the most
    # popular actress in the Golden Eagle Award.[user] I like her very much.[goal]” dialogue contexts
    input_str = ""
    for utt, goal in list(zip(dialogue_context, prev_goals)):
        if utt['role'] == "user":
            input_str += USER_TOKEN + " "
        elif utt['role'] == 'assistant':
            input_str += GOAL_TOKEN + " "
            input_str += goal + " "
            input_str += SYSTEM_TOKEN + " "
        input_str += utt['content']

    input_str = f"{input_str} {TARGET} {target_item} {GOAL_TOKEN}"
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
    input_ids = input_ids[-(max_sequence_length - 2):]
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

    # if not inference time.
    label = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{GOAL_TOKEN}: " + instance['goal']))
    label = label[:max_target_length]
    label = label + [tokenizer.eos_token_id]

    return input_ids, label


def convert_example_to_feature_for_unimind_topic_prediction(tokenizer, instance, max_sequence_length=512,
                                                            max_target_length=50, is_test=False):
    """
    function that creates the input sequence for unimind topic prediction task
    @param tokenizer:  hugginface tokenizer
    @param instance: the input instance
    @param max_sequence_length: maximum number of tokens in the input sequence
    @param max_target_length: maximum number of tokens in the target sequence
    @param is_test: if is in testing time
    @return: the input sequence of the unimind topic prediction task
    """
    dialogue_context = instance['dialogue_context']
    prev_topics = instance['pre_topics']
    input_str = ""
    target_item = instance['task_background']['target_topic']
    for utt, topic in list(zip(dialogue_context, prev_topics)):
        if utt['role'] == "user":
            input_str += USER_TOKEN + " "
        elif utt['role'] == 'assistant':
            input_str += TOPIC_TOKEN + " "
            input_str += topic + " "
            input_str += SYSTEM_TOKEN + " "
        input_str += utt['content']

    # ground truth goal
    if not is_test:
        goal = instance['goal']
    else:
        goal = instance['pred_goal']

    input_str = f"{input_str} {GOAL_TOKEN} {goal} {TARGET} {target_item} {TOPIC_TOKEN}"
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
    input_ids = input_ids[-(max_sequence_length - 2):]
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

    # if not inference time.
    label = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{TOPIC_TOKEN}: " + instance['topic']))
    label = label[:max_target_length]
    label = label + [tokenizer.eos_token_id]

    return input_ids, label


def convert_example_to_feature_for_unimind_response_generation(tokenizer, instance, max_sequence_length=512,
                                                               max_target_length=50, is_test=False):
    """
    function that creates the input sequence for unimind response generation task
    @param tokenizer:  hugginface tokenizer
    @param instance: the input instance
    @param max_sequence_length: maximum number of tokens in the input sequence
    @param max_target_length: maximum number of tokens in the target sequence
    @param is_test: if is in testing time
    @return: the input sequence of the unimind topic prediction task
    """
    dialogue_context = instance['dialogue_context']
    input_str = ""
    for utt in dialogue_context:
        if utt['role'] == "user":
            input_str += USER_TOKEN + " "
        elif utt['role'] == 'assistant':
            input_str += SYSTEM_TOKEN + " "
        input_str += utt['content']

    # ground truth goal and topic
    if not is_test:
        goal = instance['goal']
        topic = instance['topic']
    else:
        goal = instance['pred_goal']
        topic = instance['pred_topic']

    input_str = f"{input_str} {GOAL_TOKEN} {goal} {TOPIC_TOKEN} {topic} {SYSTEM_TOKEN}"
    input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
    input_ids = input_ids[-(max_sequence_length - 2):]
    input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

    # if not inference time.
    label = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(f"{SYSTEM_TOKEN}: " + instance['response']))
    label = label[:max_target_length]
    label = label + [tokenizer.eos_token_id]

    return input_ids, label


class UNIMINDDataProcessorForRecommendation(DataProcessorForRecommendation):

    def __call__(self, tokenizer, instance, max_sequence_length=512, max_target_length=50, task=None, is_test=False):
        """
        feature function for the UNIMIND policy
        :param tokenizer: the huggingface tokenizer for the Bert model
        :param instance: an particular instance
        :param max_sequence_length: the maximum sequence length
        :param max_target_length: the maximal number of generated tokens
        :param task: the current task of interest,
        :return: a tokenized input sequence and its corresponding label
        """
        # the dictionary that contains different functions
        func_dictionary = {
            'goal': convert_example_to_feature_for_unimind_goal_prediction,
            'topic': convert_example_to_feature_for_unimind_topic_prediction,
            'response': convert_example_to_feature_for_unimind_response_generation
        }
        # constructing the inputs and labels
        inputs, labels = func_dictionary[task](tokenizer=tokenizer,
                                               instance=instance,
                                               max_sequence_length=max_sequence_length,
                                               max_target_length=max_target_length,
                                               is_test=is_test
                                               )
        return inputs, labels


class UnimindTorchDataset(BaseTorchDataset):

    def __init__(self, tokenizer, instances, tasks=None, goal2id=None, max_sequence_length=512, padding='max_length',
                 pad_to_multiple_of=True, device=None, convert_example_to_feature=None, max_target_length=50,
                 is_test=False, is_gen=False, n_objectives=3, is_preference=False):
        """
        constructor for the BaseTorchDataset Class
        @param tokenizer: an huggingface tokenizer
        @param instances: a list of instances
        @param goal2id: a dictionary which maps goal to index.
        @param max_sequence_length: the maximum length of the input sequence.
        @param padding: type of padding
        @param pad_to_multiple_of: pad to multiple instances
        @param device: device to allocate the data, eg: cpu or gpu
        @param convert_example_to_feature: a function that convert raw instances to
        corresponding inputs and labels for the model.
        @param max_target_length the maximum number of the target sequence (response generation only)
        @param is_test True if inference step False if training step
        @param is_gen True if response generation else False
        """
        super(BaseTorchDataset, self).__init__()
        self.max_sequence_length = max_sequence_length
        self.tokenizer = tokenizer
        self.goal2id = goal2id
        self.pad_to_multiple_of = pad_to_multiple_of
        self.padding = padding
        self.device = device
        self.max_target_length = max_target_length
        self.is_test = is_test
        self.is_gen = is_gen
        self.is_preference = is_preference
        self.n_objectives = n_objectives

        # tasks
        self.tasks = tasks
        self.instances = self.preprocess_data(instances, convert_example_to_feature)

    def collate_fn(self, batch):

        input_features = defaultdict(list)
        labels = []
        for instance in batch:
            input_features['input_ids'].append(instance['input_ids'])
            labels.append(instance['label'])

        # padding the input features
        input_features = self.tokenizer.pad(
            input_features, padding=self.padding, pad_to_multiple_of=self.pad_to_multiple_of,
            max_length=self.max_sequence_length
        )
        # convert features to torch tensors
        for k, v in input_features.items():
            if not isinstance(v, torch.Tensor):
                input_features[k] = torch.as_tensor(v, device=self.device)

        # labels for response generation task
        labels = pad_sequence(
            [torch.tensor(label, dtype=torch.long) for label in labels],
            batch_first=True, padding_value=IGNORE_INDEX)
        labels = labels.to(self.device)

        new_batch = {
            "context": input_features,
            "labels": labels
        }
        return new_batch

    def preprocess_data(self, instances, convert_example_to_feature):
        """
        method that preprocess the input instances for unimind model
        @param instances: a set of input instances.
        @param convert_example_to_feature: a dictionary which contains key and values where values are functions
        @return: a set of processed input instances.
        """
        # if we have multiple feature functions
        processed_instances = []
        # loop overall instances.
        for instance in instances:
            # loop over task in tasks
            for task in self.tasks:
                # constructing the feature for the current task
                input_ids, label = convert_example_to_feature(tokenizer=self.tokenizer,
                                                              instance=instance,
                                                              max_sequence_length=self.max_sequence_length,
                                                              max_target_length=self.max_target_length,
                                                              is_test=self.is_test,
                                                              task=task)

                new_instance = {
                    "input_ids": input_ids,
                    "label": label
                }
                processed_instances.append(new_instance)

        return processed_instances
