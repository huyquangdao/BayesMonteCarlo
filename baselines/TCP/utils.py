import os
import json
from transformers import BertTokenizer

import dataclasses
from dataclasses import dataclass
from typing import List

from dataset.datasets import TCPTorchDataset

PAD = "[PAD]"
UNK = "[UNK]"
CLS = "[CLS]"
SEP = "[SEP]"

ACT = "[A]"  # denote an action
TPC = "[T]"  # denote a topic
BOP = "[BOP]"  # begin of knowledge hop
EOP = "[EOP]"  # end of knowledge hop
BOS = "[BOS]"  # begin of sequence
EOS = "[EOS]"  # end of sequence

SPECIAL_TOKENS_MAP = {"additional_special_tokens": [ACT, TPC, BOP, EOP, BOS, EOS]}


@dataclass(frozen=True)
class InputFeature:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    user_profile_ids: List[int]
    user_profile_segs: List[int]
    user_profile_poss: List[int]
    knowledge_ids: List[int]
    knowledge_segs: List[int]
    knowledge_poss: List[int]
    knowledge_hops: List[int]
    conversation_ids: List[int]
    conversation_segs: List[int]
    conversation_poss: List[int]

    target_ids: List[int]
    input_ids: List[int]
    gold_ids: List[int]

    def to_json_string(self):
        """Serialize this instance to a JSON string."""
        return json.dumps(dataclasses.asdict(self)) + "\n"


def get_reverse_path(goal_path: list, topic_path: list):
    """function borroweed from the official TCP implementation"""
    """parse plan path by simple concat"""
    path_str = ""
    for a, t in zip(goal_path, topic_path):
        if not t in path_str:
            path_str += "%s%s%s%s" % (ACT, a, TPC, t)
        elif not a in path_str:
            path_str += "%s%s%s%s" % (ACT, a, TPC, t)
    return path_str


def parse_conversation(tokenizer, history, max_sequence_length=512, target_item = None):
    """function borroweed from the official TCP implementation"""
    """parse plan path by simple concat"""
    tokens, segs = [], []
    if target_item is not None:
        target_tokens = tokenizer.tokenize(target_item)
        tokens = tokens + target_tokens
        segs = segs + [0] * len(target_tokens)
    for utt in history:
        h = utt['content']
        cur_turn_type = 0 if utt['role'] == 'user' else 1
        h = tokenizer.tokenize(h)
        tokens = tokens + h + [SEP]
        segs = segs + len(h) * [cur_turn_type] + [cur_turn_type]
    ids = tokenizer.convert_tokens_to_ids(tokens)
    if len(ids) > max_sequence_length - 2:
        ids = ids[2 - max_sequence_length:]
        segs = segs[2 - max_sequence_length:]
        ids = tokenizer.convert_tokens_to_ids([CLS]) + ids + tokenizer.convert_tokens_to_ids([SEP])
        segs = [1] + segs + [1]
    else:
        ids = tokenizer.convert_tokens_to_ids([CLS]) + ids
        segs = [1] + segs
    poss = list(range(len(ids)))
    assert len(ids) == len(poss) == len(segs)
    return ids, segs, poss


def parse_plan(tokenizer, plan_path, target, is_test=False):
    """function borroweed from the official TCP implementation"""
    """parse plan path by simple concat"""
    target_str = ACT + target[0] + TPC + target[1]
    target_tokens = tokenizer.tokenize(target_str)
    if is_test:
        input_tokens = [BOS, ACT] + target_tokens
        gold_tokens = [BOS, ACT] + target_tokens
    else:
        plan_tokens = tokenizer.tokenize(plan_path)
        input_tokens = [BOS] + target_tokens + plan_tokens
        gold_tokens =  target_tokens + plan_tokens + [EOS]

    # assert len(plan_path) > 0

    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)
    gold_ids = tokenizer.convert_tokens_to_ids(gold_tokens)
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)
    assert len(input_ids) == len(gold_ids)
    return target_ids, input_ids, gold_ids


def parse_raw_knowledge(tokenizer, knowledge, max_sequence_length=512):
    """function borroweed from the official TCP implementation"""
    """parse knowledge by simple concat"""
    tokens, segs = [], []
    for kg in knowledge:
        kg_str = "".join(kg)
        kg_tok = tokenizer.tokenize(kg_str)
        tokens = tokens + kg_tok + [SEP]
        segs = segs + len(kg_tok) * [0] + [1]

    ids = tokenizer.convert_tokens_to_ids(tokens)
    if len(ids) > max_sequence_length - 1:
        ids = ids[:max_sequence_length - 1]
        segs = segs[:max_sequence_length - 1]
    hops = [0] * len(ids)
    poss = list(range(len(ids)))
    assert len(ids) == len(poss) == len(segs) == len(hops)
    return ids, segs, poss, hops


def convert_example_to_feature_for_tcp_goal_topic_prediction(tokenizer, instance,
                                                             max_sequence_length=512,
                                                             is_test=False):
    # do not consider user profile since it is not available for INSPIRED dataset
    # p_ids, p_segs, p_poss = self._parse_profile(row["user_profile"])
    # profile is empty
    # p_ids, p_segs, p_poss = [], [], []
    # if len(instance['knowledge']) == 0:
    #     instance['knowledge'] = [instance['task_background']['target_topic'], "", ""]

    instance['knowledge'] = [instance['task_background']['target_topic'], "", ""]
    k_ids, k_segs, k_poss, k_hops = parse_raw_knowledge(tokenizer, instance["knowledge"], max_sequence_length)
    # k_ids, k_segs, k_poss, k_hops = [], [], [], []
    h_ids, h_segs, h_poss = parse_conversation(tokenizer, instance["dialogue_context"], max_sequence_length, instance['task_background']['target_topic'])
    # since user profile is not available
    p_ids, p_segs, p_poss = h_ids, h_segs, h_poss
    target_goal = instance['task_background']['target_goal']
    target_topic = instance['task_background']['target_topic']
    target_ids, input_ids, gold_ids = parse_plan(tokenizer, get_reverse_path(instance["reversed_goals"],
                                                                             instance['reversed_topics']),
                                                 [target_goal, target_topic],
                                                 is_test)
    inputs = {
        "user_profile_ids": p_ids,
        "user_profile_segs": p_segs,
        "user_profile_poss": p_poss,
        "knowledge_ids": k_ids,
        "knowledge_segs": k_segs,
        "knowledge_poss": k_poss,
        "knowledge_hops": k_hops,
        "conversation_ids": h_ids,
        "conversation_segs": h_segs,
        "conversation_poss": h_poss,
        "target_ids": target_ids,
        "input_ids": input_ids,
        "gold_ids": gold_ids
    }
    feature = InputFeature(**inputs)
    return feature


def predict_action_tcp(args, policy_model, policy_tokenizer, state, max_sequence_length=512, device=None, is_test=True):
    feature = convert_example_to_feature_for_tcp_goal_topic_prediction(tokenizer=policy_tokenizer,
                                                                       instance=state,
                                                                       max_sequence_length=max_sequence_length,
                                                                       is_test=is_test
                                                                       )

    collated_batch = TCPTorchDataset.static_collate_fn([feature], device=device)
    gen_seqs = policy_model.generate(
        args,
        collated_batch
    )
    sentences = combine_tokens(gen_seqs, policy_tokenizer)
    goal, topic = get_eval_output(sentences[0])
    return goal, topic


def get_tokenizer(config_dir):
    tokenizer = BertTokenizer.from_pretrained(config_dir)
    num_added_tokens = tokenizer.add_special_tokens(SPECIAL_TOKENS_MAP)
    special_token_id_dict = {
        "pad_token_id": tokenizer.pad_token_id,
        "bos_token_id": tokenizer.convert_tokens_to_ids(BOS),
        "eos_token_id": tokenizer.convert_tokens_to_ids(EOS),
    }
    return tokenizer, num_added_tokens, special_token_id_dict


def combine_tokens(output, tokenizer, vocab_list=None):
    return_sentence = []
    for batch in range(output.size(0)):
        out_tokens = tokenizer.decode(output[batch, :])
        # print(out_tokens)
        out_tokens = out_tokens.replace(BOS, "")
        out_tokens = out_tokens.replace(EOS, "")
        out_tokens = out_tokens.replace(PAD, "")
        return_sentence.append(out_tokens)
    return return_sentence


def get_eval_output(path_str):
    # parse backward path
    # i.e., [A]...[T]...[A]act[T]tpc
    try:
        eval_span = path_str.split(ACT)[-1].strip()
    except IndexError:
        eval_span = None

    if eval_span is None:
        action, topic = UNK, UNK
    else:
        try:
            action = eval_span.split(TPC)[0].strip()
        except IndexError:
            action = UNK
        try:
            topic = eval_span.split(TPC)[-1].strip()
        except IndexError:
            topic = UNK
    return (action, topic)
