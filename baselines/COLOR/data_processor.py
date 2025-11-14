from collections import defaultdict
import torch

from base.data_processor import DataProcessorForRecommendation
from base.torch_dataset import BaseTorchDataset

from baselines.COLOR.utils import convert_example_to_feature_for_color_planning, \
    convert_example_to_feature_for_color_bridge_mapping


def max_seq_length(list_l):
    """
    function that returns max length of a given list of elements
    :param list_l: a list of elements
    :return:
    """
    return max(len(l) for l in list_l)


def pad_sequence(list_l, max_len, padding_value=0):
    """
    function that pads a list of elements
    :param list_l: a list of elements
    :param max_len: the max length input
    :param padding_value: padding value
    :return:
    """
    assert len(list_l) <= max_len
    padding_l = [padding_value] * (max_len - len(list_l))
    padded_list = list_l + padding_l
    return padded_list


def list_to_tensor(list_l, padding_idx=0, device=None):
    """
    method that convert a list to a tensor
    :param list_l: a given list of elements
    :param padding_idx: padding value
    :param device: the device
    :return:
    """
    max_len = max_seq_length(list_l)
    padded_lists = []
    for list_seq in list_l:
        padded_lists.append(pad_sequence(list_seq, max_len, padding_value=padding_idx))
    input_tensor = torch.tensor(padded_lists, dtype=torch.long)
    input_tensor = input_tensor.to(device).contiguous()
    return input_tensor


def varlist_to_tensor(list_vl, padding_idx=0, device=None):
    """
    function that convert a list to a tensor
    :param list_vl:
    :param padding_idx:
    :param device:
    :return:
    """
    lens = []
    for list_l in list_vl:
        lens.append(max_seq_length(list_l))
    max_len = max(lens)

    padded_lists = []
    for list_seqs in list_vl:
        v_list = []
        for list_l in list_seqs:
            v_list.append(pad_sequence(list_l, max_len, padding_value=padding_idx))
        padded_lists.append(v_list)
    input_tensor = torch.tensor(padded_lists, dtype=torch.long)
    input_tensor = input_tensor.to(device).contiguous()
    return input_tensor


def get_attention_mask(data_tensor: torch.tensor, padding_idx=0, device=None):
    """
    function that returns the attention mask for the give tensor
    :param data_tensor: a given data tensor
    :param padding_idx: padding index
    :param device: the device
    :return:
    """
    attention_mask = data_tensor.masked_fill(data_tensor == padding_idx, 0)
    attention_mask = attention_mask.masked_fill(attention_mask != padding_idx, 1)
    attention_mask = attention_mask.to(device).contiguous()
    return attention_mask


def planner_list_to_tensor(list_l, special_padding_value=None, padding_idx=None, device=None):
    """
    method that convert a list to a tensor for the color planner
    :param list_l: a list of elements
    :param special_padding_value: a special value for padding
    :param padding_idx: the padding index
    :param device: the device
    :return:
    """
    max_len = max_seq_length(list_l)
    padded_lists = []
    for list_seq in list_l:
        if special_padding_value is None:
            padded_lists.append(pad_sequence(list_seq, max_len, padding_value=padding_idx))
        else:
            padded_lists.append(pad_sequence(list_seq, max_len, padding_value=special_padding_value))
    input_tensor = torch.tensor(padded_lists, dtype=torch.long)
    input_tensor = input_tensor.to(device).contiguous()
    return input_tensor


class COLORDataProcessorForRecommendation(DataProcessorForRecommendation):

    def __call__(self, tokenizer, instance, max_sequence_length=512, is_test=False, stage='planner'):
        """
        feature function for the BART policy model
        :param tokenizer: the huggingface tokenizer for the Bert model
        :param instance: an particular instance
        :param max_sequence_length: the maximum sequence length
        :param is_test: action mapping
        :return: a tokenized input sequence and its corresponding label
        """
        # feature function for the brownian bridge training
        if stage == 'bridge':
            return convert_example_to_feature_for_color_bridge_mapping(tokenizer, instance, max_sequence_length)
        # feature functon for the color planner training
        elif stage == 'planner':
            return convert_example_to_feature_for_color_planning(tokenizer, instance, max_sequence_length)


class COLORBridgeTorchDataset(BaseTorchDataset):

    def __init__(self, stage, tokenizer, instances, goal2id=None, max_sequence_length=512,
                 padding='max_length', pad_to_multiple_of=True, device=None, convert_example_to_feature=None,
                 max_target_length=50, n_objectives=3,
                 is_test=False, is_gen=False):
        """
        constructor for the COLOR torch dataset
        :param tokenizer: the tokenizer
        :param instances: a list of instances
        :param goal2id: a dictionary that maps goals to indexes
        :param max_sequence_length: the maximal number of input tokens
        :param padding: padding type
        :param pad_to_multiple_of: True if pad to multiple instances
        :param device: the device
        :param convert_example_to_feature: feature function
        :param max_target_length: the maximal number of target sentences.
        :param n_objectives: the number of objectives
        :param is_test: True if it is test
        :param is_gen: True if if is inference time.
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

        self.stage = stage
        self.n_objectives = n_objectives
        self.stage = stage

        self.instances = self.preprocess_data(instances, convert_example_to_feature)

    def preprocess_data(self, instances, convert_example_to_feature):
        """
        method that preprocess the input data for the TCP model.
        @param instances: a list of input instances.
        @param convert_example_to_feature: a function that processes the given input instance.
        @return:
        """
        processed_instances = []
        for instance in instances:
            # data processing for training the policy model
            features = convert_example_to_feature(self.tokenizer, instance,
                                                  self.max_sequence_length,
                                                  self.is_test,
                                                  self.stage
                                                  )
            processed_instances.extend(features)
        return processed_instances

    def collate_fn(self, mini_batch):
        """Custom collate function for dealing with batches of input data.
        Arguments:
            mini_batch: A list of input features.
        Return:
            dict: (dict) A dict of tensors.
        """
        batch_user_utterance_input = []
        batch_delta_follow_input = []
        batch_interim_subgoal_input = []
        batch_start_subgoal_input = []
        batch_target_subgoal_input = []
        batch_transition_input = []
        batch_interim_t, batch_target_T = [], []

        for sample in mini_batch:
            batch_user_utterance_input.append(sample.user_utt_ids)
            batch_delta_follow_input.append(sample.follow_ids)
            batch_transition_input.append(sample.transition_ids)
            batch_interim_subgoal_input.append(sample.interim_ids)
            batch_start_subgoal_input.append(sample.start_ids)
            batch_target_subgoal_input.append(sample.target_ids)
            batch_interim_t.append(sample.interim_t)
            batch_target_T.append(sample.target_T)

        # inputs
        user_utt_ids = list_to_tensor(batch_user_utterance_input, device=self.device)
        user_utt_mask = get_attention_mask(user_utt_ids, device=self.device)

        delta_follow_ids = list_to_tensor(batch_delta_follow_input, device=self.device)
        delta_follow_mask = get_attention_mask(delta_follow_ids, device=self.device)

        transition_ids = list_to_tensor(batch_transition_input, device=self.device)
        transition_mask = get_attention_mask(transition_ids, device=self.device)

        interim_subgoal_ids = list_to_tensor(batch_interim_subgoal_input, device=self.device)
        interim_subgoal_mask = get_attention_mask(interim_subgoal_ids, device=self.device)

        start_subgoal_ids = list_to_tensor(batch_start_subgoal_input, device=self.device)
        start_subgoal_mask = get_attention_mask(start_subgoal_ids, device=self.device)

        target_subgoal_ids = list_to_tensor(batch_target_subgoal_input, device=self.device)
        target_subgoal_mask = get_attention_mask(target_subgoal_ids, device=self.device)

        interim_t = torch.tensor(batch_interim_t, dtype=torch.long).to(self.device).contiguous()
        target_T = torch.tensor(batch_target_T, dtype=torch.long).to(self.device).contiguous()

        collated_batch = {
            "user_utterance": [user_utt_ids, user_utt_mask],
            "delta_follow": [delta_follow_ids, delta_follow_mask],
            "transition": [transition_ids, transition_mask],
            "interim_subgoal": [interim_subgoal_ids, interim_subgoal_mask],
            "start_subgoal": [start_subgoal_ids, start_subgoal_mask],
            "target_subgoal": [target_subgoal_ids, target_subgoal_mask],
            "interim_t": interim_t,
            "target_T": target_T,
        }

        return collated_batch

    @staticmethod
    def static_collate_fn(mini_batch, device=None):
        pass


class COLORPlanningTorchDataset(BaseTorchDataset):

    def __init__(self, stage, model, latent_dim, tokenizer, instances, goal2id=None, max_sequence_length=512,
                 padding='max_length', pad_to_multiple_of=True, device=None, convert_example_to_feature=None,
                 max_target_length=50, n_objectives=3,
                 is_test=False, is_gen=False):
        """
        constructor for the COLOR torch dataset
        :param model: the color model
        :param latent_dim: the dimension of the latent embedding
        :param tokenizer: the tokenizer
        :param instances: a list of instances
        :param goal2id: a dictionary that maps goals to indexes
        :param max_sequence_length: the maximal number of input tokens
        :param padding: padding type
        :param pad_to_multiple_of: True if pad to multiple instances
        :param device: the device
        :param convert_example_to_feature: feature function
        :param max_target_length: the maximal number of target sentences.
        :param n_objectives: the number of objectives
        :param is_test: True if it is test
        :param is_gen: True if if is inference time.
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

        self.stage = stage
        self.n_objectives = n_objectives
        self.latent_dim = latent_dim
        self.model = model
        self.latent_dim = latent_dim

        self.instances = self.preprocess_data(instances, convert_example_to_feature)

    def preprocess_data(self, instances, convert_example_to_feature):
        """
        method that preprocess the input data for the TCP model.
        @param instances: a list of input instances.
        @param convert_example_to_feature: a function that processes the given input instance.
        @return:
        """
        processed_instances = []
        m = -1
        for instance in instances:
            # data processing for training the policy model
            feature = convert_example_to_feature(self.tokenizer, instance,
                                                 self.max_sequence_length,
                                                 self.is_test
                                                 )
            m = max(feature.transition_number, m)
            processed_instances.append(feature)
        print(m)
        return processed_instances

    def collate_fn(self, mini_batch):
        """
        Collate function for the planner class
        """
        batch_input = []
        batch_decoder_input_all = []
        batch_transition_input = []
        batch_start_subgoal_input = []
        batch_target_subgoal_input = []
        batch_user_utterance_input = []
        batch_delta_follow_input = []
        batch_transition_number = []

        for sample in mini_batch:
            batch_input.append(sample.input_ids)
            batch_decoder_input_all.append(sample.decoder_input_all_ids)
            batch_transition_input.append(sample.transition_ids)
            batch_start_subgoal_input.append(sample.start_ids)
            batch_target_subgoal_input.append(sample.target_ids)
            batch_user_utterance_input.append(sample.user_utt_ids)
            batch_delta_follow_input.append(sample.follow_ids)
            batch_transition_number.append(sample.transition_number)

        input_ids = list_to_tensor(batch_input, device=self.device)
        input_mask = get_attention_mask(input_ids, device=self.device)
        decoder_input_all_ids = list_to_tensor(batch_decoder_input_all, device=self.device)
        decoder_input_all_mask = get_attention_mask(decoder_input_all_ids, device=self.device)

        start_subgoal_ids = list_to_tensor(batch_start_subgoal_input, device=self.device)
        start_subgoal_mask = get_attention_mask(start_subgoal_ids, device=self.device)

        target_subgoal_ids = list_to_tensor(batch_target_subgoal_input, device=self.device)
        target_subgoal_mask = get_attention_mask(target_subgoal_ids, device=self.device)

        user_utt_ids = list_to_tensor(batch_user_utterance_input, device=self.device)
        user_utt_mask = get_attention_mask(user_utt_ids, device=self.device)

        delta_follow_ids = list_to_tensor(batch_delta_follow_input, device=self.device)
        delta_follow_mask = get_attention_mask(delta_follow_ids, device=self.device)

        transition_number = torch.tensor(batch_transition_number, dtype=torch.long).to(self.device).contiguous()

        batch_tc_mask = []
        for bsz, sample in enumerate(mini_batch):
            tc_mask_temp = (len(sample.decoder_input_ids_list) - 1) * [1]
            batch_tc_mask.append(tc_mask_temp)
        tc_mask = planner_list_to_tensor(batch_tc_mask, special_padding_value=0, device=self.device)
        gold_temp = torch.full((tc_mask.shape[0], tc_mask.shape[1], self.latent_dim), 0, dtype=torch.float).to(
            self.device)
        for bsz, sample in enumerate(mini_batch):
            if len(sample.decoder_input_ids_list) > 1:
                for idx, dec_ids in enumerate(sample.decoder_input_ids_list):
                    if idx == len(sample.decoder_input_ids_list) - 1:
                        continue
                    temp_ids = list_to_tensor([dec_ids], device=self.device)
                    temp_mask = get_attention_mask(temp_ids, device=self.device)
                    gold_temp[bsz, idx, :] = self.model.plm.get_time_control_embed(temp_ids, temp_mask)

        simulate_temp = torch.full((tc_mask.shape[0], tc_mask.shape[1], self.latent_dim), 0, dtype=torch.float).to(
            self.device)
        for bsz, sample in enumerate(mini_batch):
            if len(sample.decoder_input_ids_list) > 1:
                start_latent = self.model.plm.get_time_control_embed(start_subgoal_ids[bsz:bsz + 1, :],
                                                                     start_subgoal_mask[bsz:bsz + 1, :])
                target_latent = self.model.plm.get_time_control_embed(target_subgoal_ids[bsz:bsz + 1, :],
                                                                      target_subgoal_mask[bsz:bsz + 1, :])
                Z_u = self.model.plm.get_user_utt_representation(user_utt_ids[bsz:bsz + 1, :],
                                                                 user_utt_mask[bsz:bsz + 1, :])
                delta_u = self.model.plm.get_delta_u_representation(delta_follow_ids[bsz:bsz + 1, :],
                                                                    delta_follow_mask[bsz:bsz + 1, :])

                # simulate Brownian bridge trjectories
                simulate_bridge_points = self.model.plm.simulate_brownian_bridge(B_0=start_latent, B_T=target_latent,
                                                                                 T=len(sample.decoder_input_ids_list),
                                                                                 Z_u=Z_u, delta_u=delta_u)

                assert len(simulate_bridge_points) == len(sample.decoder_input_ids_list)
                for idx, embed in enumerate(simulate_bridge_points[1:]):
                    simulate_temp[bsz, idx, :] = embed

        if not self.is_test:
            collated_batch = {
                "input": [input_ids, input_mask],
                "decoder_input_all": [decoder_input_all_ids[:, :-1].contiguous(),
                                      decoder_input_all_mask[:, :-1].contiguous()],
                "label": [decoder_input_all_ids[:, 1:].contiguous(), decoder_input_all_mask[:, 1:].contiguous()],
                "transition_number": transition_number,
                "gold_bridge_embed": [gold_temp.contiguous(), tc_mask],
                "simulate_bridge_embed": [simulate_temp.contiguous(), tc_mask],
            }
        else:
            gold_bridge_list = []
            for bsz, sample in enumerate(mini_batch):
                tc_list = []
                if len(sample.decoder_input_ids_list) > 1:
                    for idx, dec_ids in enumerate(sample.decoder_input_ids_list):
                        if idx == len(sample.decoder_input_ids_list) - 1:
                            continue
                        temp_ids = list_to_tensor([dec_ids], device=self.device)
                        temp_mask = get_attention_mask(temp_ids, device=self.device)
                        rep = self.model.plm.get_time_control_embed(temp_ids, temp_mask)
                        tc_list.append(rep)
                gold_bridge_list.append(tc_list)
            collated_batch = {
                "input": [input_ids, input_mask],
                "decoder_input_all": [decoder_input_all_ids[:, :-1].contiguous(),
                                      decoder_input_all_mask[:, :-1].contiguous()],
                "transition_number": transition_number,
                "gold_bridge_embed": [gold_temp.contiguous(), tc_mask],
                "simulate_bridge_embed": [simulate_temp.contiguous(), tc_mask],
                "user_utterance": [user_utt_ids, user_utt_mask],
                "delta_follow": [delta_follow_ids, delta_follow_mask],
                "start_subgoal": [start_subgoal_ids, start_subgoal_mask],
                "target_subgoal": [target_subgoal_ids, target_subgoal_mask],
                "gold_bridge_list": gold_bridge_list,
            }

        return collated_batch

    @staticmethod
    def static_collate_fn(mini_batch, model, device=None, latent_dim=16, is_test=True):
        batch_input = []
        batch_decoder_input_all = []
        batch_transition_input = []
        batch_start_subgoal_input = []
        batch_target_subgoal_input = []
        batch_user_utterance_input = []
        batch_delta_follow_input = []
        batch_transition_number = []

        for sample in mini_batch:
            batch_input.append(sample.input_ids)
            batch_decoder_input_all.append(sample.decoder_input_all_ids)
            batch_transition_input.append(sample.transition_ids)
            batch_start_subgoal_input.append(sample.start_ids)
            batch_target_subgoal_input.append(sample.target_ids)
            batch_user_utterance_input.append(sample.user_utt_ids)
            batch_delta_follow_input.append(sample.follow_ids)
            batch_transition_number.append(sample.transition_number)

        input_ids = list_to_tensor(batch_input, device=device)
        input_mask = get_attention_mask(input_ids, device=device)
        decoder_input_all_ids = list_to_tensor(batch_decoder_input_all, device=device)
        decoder_input_all_mask = get_attention_mask(decoder_input_all_ids, device=device)

        start_subgoal_ids = list_to_tensor(batch_start_subgoal_input, device=device)
        start_subgoal_mask = get_attention_mask(start_subgoal_ids, device=device)

        target_subgoal_ids = list_to_tensor(batch_target_subgoal_input, device=device)
        target_subgoal_mask = get_attention_mask(target_subgoal_ids, device=device)

        user_utt_ids = list_to_tensor(batch_user_utterance_input, device=device)
        user_utt_mask = get_attention_mask(user_utt_ids, device=device)

        delta_follow_ids = list_to_tensor(batch_delta_follow_input, device=device)
        delta_follow_mask = get_attention_mask(delta_follow_ids, device=device)

        transition_number = torch.tensor(batch_transition_number, dtype=torch.long).to(device).contiguous()

        batch_tc_mask = []
        for bsz, sample in enumerate(mini_batch):
            tc_mask_temp = (len(sample.decoder_input_ids_list) - 1) * [1]
            batch_tc_mask.append(tc_mask_temp)
        tc_mask = planner_list_to_tensor(batch_tc_mask, special_padding_value=0, device=device)
        gold_temp = torch.full((tc_mask.shape[0], tc_mask.shape[1], latent_dim), 0, dtype=torch.float).to(
            device)
        for bsz, sample in enumerate(mini_batch):
            if len(sample.decoder_input_ids_list) > 1:
                for idx, dec_ids in enumerate(sample.decoder_input_ids_list):
                    if idx == len(sample.decoder_input_ids_list) - 1:
                        continue
                    temp_ids = list_to_tensor([dec_ids], device=device)
                    temp_mask = get_attention_mask(temp_ids, device=device)
                    gold_temp[bsz, idx, :] = model.get_time_control_embed(temp_ids, temp_mask)

        simulate_temp = torch.full((tc_mask.shape[0], tc_mask.shape[1], latent_dim), 0, dtype=torch.float).to(
            device)
        for bsz, sample in enumerate(mini_batch):
            if len(sample.decoder_input_ids_list) > 1:
                start_latent = model.get_time_control_embed(start_subgoal_ids[bsz:bsz + 1, :],
                                                            start_subgoal_mask[bsz:bsz + 1, :])
                target_latent = model.get_time_control_embed(target_subgoal_ids[bsz:bsz + 1, :],
                                                             target_subgoal_mask[bsz:bsz + 1, :])
                Z_u = model.get_user_utt_representation(user_utt_ids[bsz:bsz + 1, :],
                                                        user_utt_mask[bsz:bsz + 1, :])
                delta_u = model.get_delta_u_representation(delta_follow_ids[bsz:bsz + 1, :],
                                                           delta_follow_mask[bsz:bsz + 1, :])

                # simulate Brownian bridge trjectories
                simulate_bridge_points = model.simulate_brownian_bridge(B_0=start_latent, B_T=target_latent,
                                                                        T=len(sample.decoder_input_ids_list),
                                                                        Z_u=Z_u, delta_u=delta_u)

                assert len(simulate_bridge_points) == len(sample.decoder_input_ids_list)
                for idx, embed in enumerate(simulate_bridge_points[1:]):
                    simulate_temp[bsz, idx, :] = embed

        if not is_test:
            collated_batch = {
                "input": [input_ids, input_mask],
                "decoder_input_all": [decoder_input_all_ids[:, :-1].contiguous(),
                                      decoder_input_all_mask[:, :-1].contiguous()],
                "label": [decoder_input_all_ids[:, 1:].contiguous(), decoder_input_all_mask[:, 1:].contiguous()],
                "transition_number": transition_number,
                "gold_bridge_embed": [gold_temp.contiguous(), tc_mask],
                "simulate_bridge_embed": [simulate_temp.contiguous(), tc_mask],
            }
        else:
            gold_bridge_list = []
            for bsz, sample in enumerate(mini_batch):
                tc_list = []
                if len(sample.decoder_input_ids_list) > 1:
                    for idx, dec_ids in enumerate(sample.decoder_input_ids_list):
                        if idx == len(sample.decoder_input_ids_list) - 1:
                            continue
                        temp_ids = list_to_tensor([dec_ids], device=device)
                        temp_mask = get_attention_mask(temp_ids, device=device)
                        rep = model.get_time_control_embed(temp_ids, temp_mask)
                        tc_list.append(rep)
                gold_bridge_list.append(tc_list)
            collated_batch = {
                "input": [input_ids, input_mask],
                "decoder_input_all": [decoder_input_all_ids[:, :-1].contiguous(),
                                      decoder_input_all_mask[:, :-1].contiguous()],
                "transition_number": transition_number,
                "gold_bridge_embed": [gold_temp.contiguous(), tc_mask],
                "simulate_bridge_embed": [simulate_temp.contiguous(), tc_mask],
                "user_utterance": [user_utt_ids, user_utt_mask],
                "delta_follow": [delta_follow_ids, delta_follow_mask],
                "start_subgoal": [start_subgoal_ids, start_subgoal_mask],
                "target_subgoal": [target_subgoal_ids, target_subgoal_mask],
                "gold_bridge_list": gold_bridge_list,
            }

        return collated_batch
