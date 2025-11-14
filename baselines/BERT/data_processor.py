from base.data_processor import DataProcessorForRecommendation
from base.torch_dataset import BaseTorchDataset
from config.constants import USER_TOKEN, SYSTEM_TOKEN, GOAL_TOKEN, TOPIC_TOKEN, SEP_TOKEN, PATH_TOKEN, TARGET, \
    CONTEXT_TOKEN


class BERTDataProcessorForRecommendation(DataProcessorForRecommendation):

    def __call__(self, tokenizer, instance, max_sequence_length=512, action_to_id=None):
        """
        feature function for the BERT policy model
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
        dialogue_str = ""
        for utt in dialogue_context:
            if utt['role'] == "user":
                dialogue_str += USER_TOKEN
            elif utt['role'] == 'assistant':
                dialogue_str += SYSTEM_TOKEN
            dialogue_str += utt['content']

        path_str = ""
        for goal, topic in list(zip(prev_paths, prev_topics)):
            path_str += GOAL_TOKEN
            path_str += goal
            path_str += TOPIC_TOKEN
            path_str += topic
            path_str += SEP_TOKEN

        input_str = f"{PATH_TOKEN}: {path_str} {TARGET}: {target} {CONTEXT_TOKEN}: {dialogue_str}"
        input_ids = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(input_str))
        input_ids = input_ids[-(max_sequence_length - 2):]
        input_ids = [tokenizer.cls_token_id] + input_ids + [tokenizer.sep_token_id]

        label = action_to_id[(instance['goal'], instance['topic'])]
        return input_ids, label


class BERTTorchDatasetForRecommendation(BaseTorchDataset):
    """
    BERT Torch dataset class for recommendation
    Just use it for coding
    """
    pass
