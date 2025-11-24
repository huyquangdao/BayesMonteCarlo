"""
Skeleton data processors and torch datasets for the Bayes-Adaptive LLM pipeline.
Structured after `baselines/TRIP/data_processor.py` so we can later plug in the
task-specific logic without changing the interface.
"""

from __future__ import annotations

from typing import Any, Dict, Sequence, Tuple

from base.data_processor import (
    DataProcessorForRecommendation,
    DataProcessorForNegotiation,
    DataProcessorForEmotionalSupport,
    DataProcessorForPersuation,
    DataProcessorForPreferenceEstimation,
)
from base.torch_dataset import BaseTorchDataset
from config.constants import PERSUADEE_TOKEN, PERSUADER_TOKEN, GOAL_TOKEN, SEP_TOKEN, PATH_TOKEN, CONTEXT_TOKEN




class BayesDataProcessorForRecommendation(DataProcessorForRecommendation):
    """
    Tokenization and feature conversion for recommendation tasks.
    """

    def __call__(self, tokenizer, instance, max_sequence_length=512, action_to_id=None, **kwargs):
        raise NotImplementedError("Recommendation data processor is not implemented.")


class BayesDataProcessorForNegotiation(DataProcessorForNegotiation):
    """
    Tokenization and feature conversion for negotiation tasks.
    """

    def __call__(self,
                 tokenizer,
                 instance,
                 max_sequence_length=512,
                 action_to_id=None,
                 n_bins: int = 5,
                 is_so_game: bool = True):
        raise NotImplementedError("Negotiation data processor is not implemented.")


class BayesDataProcessorForEmotionalSupport(DataProcessorForEmotionalSupport):
    """
    Tokenization and feature conversion for emotional-support tasks.
    """

    def __call__(self, tokenizer, instance, max_sequence_length=512, action_to_id=None, **kwargs):
        raise NotImplementedError("Emotional-support data processor is not implemented.")


class BayesDataProcessorForPersuation(DataProcessorForPersuation):
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


class BayesDataProcessorForPreferenceEstimation(DataProcessorForPreferenceEstimation):
    """
    Tokenization and feature conversion for preference estimation tasks.
    """

    def __call__(self, tokenizer, instance, max_sequence_length, is_test: bool = False):
        raise NotImplementedError("Preference estimation data processor is not implemented.")


class BayesTorchDatasetForRecommendation(BaseTorchDataset):
    """Torch dataset shell for recommendation."""

    pass


class BayesTorchDatasetForPreferenceEstimation(BaseTorchDataset):
    """Torch dataset shell for preference estimation."""

    pass


class BayesTorchDatasetForNegotiation(BaseTorchDataset):
    """Torch dataset shell for negotiation."""

    pass


class BayesTorchDatasetForEmotionalSupport(BaseTorchDataset):
    """Torch dataset shell for emotional-support."""

    pass


class BayesTorchDatasetForPersuation(BaseTorchDataset):
    """Torch dataset shell for persuasion."""

    pass
