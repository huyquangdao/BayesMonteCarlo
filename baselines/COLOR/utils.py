# -*- coding: utf-8 -*-
from transformers import BertTokenizer, BartTokenizer, GPT2Tokenizer
from baselines.COLOR.dataset_base import BrownianBridgeInput, PlannerInput

from config.constants import USER_TOKEN, SYSTEM_TOKEN, GOAL_TOKEN, TOPIC_TOKEN, SEP_TOKEN, TARGET, \
    IGNORE_INDEX

PAD = "[PAD]"  # consistent with Bert tokenizer
UNK = "[UNK]"  # consistent with Bert tokenizer
SEP = SEP_TOKEN

ACT = GOAL_TOKEN
TPC = TOPIC_TOKEN
BOS = "[BOS]"  # begin of sequence
EOS = "[EOS]"  # end of sequence

IGNORE_INDEX = IGNORE_INDEX


def get_tokenizer(config_dir, name="bert"):
    if name == "gpt2":
        tokenizer = GPT2Tokenizer.from_pretrained(config_dir)
        special_token_map = {"additional_special_tokens": [ACT, TPC, SEP, PAD]}
        num_added_tokens = tokenizer.add_special_tokens(special_token_map)
        special_token_id_dict = {
            "pad_token_id": tokenizer.convert_tokens_to_ids(PAD),
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
        }
    elif name == "bart":
        tokenizer = BartTokenizer.from_pretrained(config_dir)
        special_token_map = {"additional_special_tokens": [ACT, TPC]}
        num_added_tokens = tokenizer.add_special_tokens(special_token_map)
        special_token_id_dict = {
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.bos_token_id,
            "eos_token_id": tokenizer.eos_token_id,
            "sep_token_id": tokenizer.sep_token_id,
        }
    else:
        tokenizer = BertTokenizer.from_pretrained(config_dir)
        special_token_map = {"additional_special_tokens": [ACT, TPC, BOS, EOS]}
        num_added_tokens = tokenizer.add_special_tokens(special_token_map)
        special_token_id_dict = {
            "pad_token_id": tokenizer.pad_token_id,
            "bos_token_id": tokenizer.convert_tokens_to_ids(BOS),
            "eos_token_id": tokenizer.convert_tokens_to_ids(EOS),
        }
    return tokenizer, num_added_tokens, special_token_id_dict


def convert_ids_to_tokens(output, tokenizer, lang="en"):
    sentences = []
    for idx in range(output.size(0)):
        decode_tokens = tokenizer.decode(output[idx, :]).split()
        return_tokens = []
        for token in decode_tokens:
            if token == BOS:
                continue
            elif token == EOS or token == PAD:
                break
            elif token.startswith(EOS):
                break
            elif token.endswith(EOS):
                return_tokens.append(token.replace(EOS, ""))
                break
            elif token.endswith("<|endoftext|>"):
                return_tokens.append(token.replace("<|endoftext|>", ""))
                break
            elif token.upper() == "NULL":
                return_tokens.append("NULL")
            else:
                return_tokens.append(token)
        if lang == "zh":
            return_str = "".join(return_tokens)
        else:
            return_str = " ".join(return_tokens)
        sentences.append(return_str)
    return sentences


def get_eval_output(path_str: str, topic_only=False):
    if topic_only:
        try:
            if path_str.startswith(TPC):
                topic = path_str.split(TPC)[1].strip()
            else:
                topic = path_str.split(TPC)[0].strip()
        except IndexError:
            topic = UNK
        return topic
    else:
        # parse dioalog path
        # i.e., [A]act[T]tpc[A]...[T]...
        # print(path_str)
        # assert 1==0
        try:
            action = path_str.split(TPC)[0].split(ACT)[-1].strip()
        except IndexError:
            action = UNK
        try:
            if path_str.startswith(ACT):
                topic = path_str.split(ACT)[1].split(TPC)[-1].strip()
            else:
                topic = path_str.split(ACT)[0].split(TPC)[-1].strip()
        except IndexError:
            topic = UNK
        return (action, topic)


def combine_tokens(output, tokenizer):
    return_sentence = []
    for batch in range(output.size(0)):
        out_sentence = tokenizer.decode(output[batch, :]).replace(tokenizer.bos_token, "").replace(tokenizer.eos_token,
                                                                                                   "").strip()
        # out_sentence = out_sentence.split()
        # out_sentence = out_sentence[5:]
        # out_sentence = " ".join(out_sentence)
        # print(out_sentence)
        return_sentence.append(out_sentence)
    return return_sentence


def _parse_plan_path(tokenizer, sample):
    # dialog plan path
    plan_path_ids_list = []
    for idx in range(len(sample["goal_path"])):
        act_toks = tokenizer.tokenize(sample["goal_path"][idx])
        tpc_toks = tokenizer.tokenize(sample["topic_path"][idx])
        at_ids = tokenizer.convert_tokens_to_ids([ACT] + act_toks + [TPC] + tpc_toks)
        plan_path_ids_list.append(at_ids)
    return plan_path_ids_list


def _parse_input_context_planning(tokenizer, sample: dict, max_seq_len=512, turn_type_size=16):
    # some fractions are not available for this setting e.g knowledge, user profile
    sample['knowledge'] = [sample['task_background']['target_topic'], "", ""]
    # last user utterance
    if len(sample["dialogue_context"]) > 0:
        idx = len(sample['dialogue_context']) - 1
        while idx > 0 and sample["dialogue_context"][idx]['role'] != 'user':
            idx -= 1
        user_utt = sample["dialogue_context"][idx]
        assert user_utt['role'] == 'user'
        user_utt_tokens = [tokenizer.bos_token] + tokenizer.tokenize(user_utt['content']) + [
            tokenizer.eos_token]
    else:
        user_utt_tokens = [tokenizer.bos_token] + [tokenizer.eos_token]
    if len(user_utt_tokens) > max_seq_len:
        user_utt_tokens = user_utt_tokens[:max_seq_len - 1] + [tokenizer.eos_token]
    user_utt_ids = tokenizer.convert_tokens_to_ids(user_utt_tokens)

    # delta: follow discrimination
    # profile_tokens = [tokenizer.bos_token]
    # for k, v in sample["user_profile"].items():
    #     k_toks = tokenizer.tokenize(k)
    #     v_toks = tokenizer.tokenize(v)
    #     profile_tokens = profile_tokens + k_toks + v_toks
    # profile_tokens = profile_tokens + [tokenizer.sep_token]
    #### user profile is not available during interactive evaluation
    profile_tokens = []
    follow_tokens = profile_tokens + user_utt_tokens[1:]
    if len(follow_tokens) > max_seq_len:
        follow_tokens = follow_tokens[:max_seq_len - 1] + [tokenizer.eos_token]
    follow_ids = tokenizer.convert_tokens_to_ids(follow_tokens)

    # S0: domain_knowledge
    kg_tokens = [tokenizer.bos_token]
    # for kg in sample["knowledge"]:
    kg_tok = tokenizer.tokenize(" ".join(sample['knowledge']))
    kg_tokens = kg_tokens + kg_tok + [tokenizer.sep_token]

    # Dialogue history
    conv_tokens = []
    history = sample["dialogue_context"]
    if len(history) > turn_type_size:
        history = history[-turn_type_size:]
    for utt in history:
        h = utt['content']
        h_toks = tokenizer.tokenize(h)
        conv_tokens = conv_tokens + h_toks
    conv_tokens = conv_tokens + [tokenizer.eos_token]
    start_tokens = kg_tokens + conv_tokens
    if len(start_tokens) > max_seq_len:
        start_tokens = start_tokens[:max_seq_len - 1] + [tokenizer.eos_token]
    start_ids = tokenizer.convert_tokens_to_ids(start_tokens)

    # transition: user profile + S0 + ST
    transition_tokens = profile_tokens + start_tokens[1:-1] + [tokenizer.sep_token]

    act_toks = tokenizer.tokenize(sample['task_background']['target_goal'])
    tpc_toks = tokenizer.tokenize(sample['task_background']['target_topic'])

    target_tokens = [ACT] + act_toks + [TPC] + tpc_toks
    transition_tokens = transition_tokens + target_tokens + [tokenizer.eos_token]
    if len(transition_tokens) > max_seq_len:
        transition_tokens = transition_tokens[:max_seq_len - 1] + [tokenizer.eos_token]
    transition_ids = tokenizer.convert_tokens_to_ids(transition_tokens)

    # ST: target
    target_tokens = [tokenizer.bos_token] + target_tokens + [tokenizer.eos_token]
    if len(target_tokens) > max_seq_len:
        target_tokens = target_tokens[:max_seq_len - 1] + [tokenizer.eos_token]
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)

    # input
    input_tokens = [tokenizer.bos_token]
    input_tokens = input_tokens + conv_tokens[:-1] + [tokenizer.sep_token] + kg_tokens[1:] + profile_tokens[
                                                                                             1:-1] + [
                       tokenizer.eos_token]
    if len(input_tokens) > max_seq_len:
        input_tokens = input_tokens[:max_seq_len - 1] + [tokenizer.eos_token]
    input_ids = tokenizer.convert_tokens_to_ids(input_tokens)

    # decoder input
    decoder_input_ids = []
    decoder_input_lengths = []
    decoder_input_all = [tokenizer.bos_token]
    is_end_flag = False

    if len(sample["goal_path"]) == 1:
        is_end_flag = True
        for idx in range(len(sample["goal_path"])):
            act_toks = tokenizer.tokenize(sample["goal_path"][idx])
            tpc_toks = tokenizer.tokenize(sample["topic_path"][idx])
            at_ids = tokenizer.convert_tokens_to_ids([ACT] + act_toks + [TPC] + tpc_toks)
            decoder_input_ids.append(at_ids)
            decoder_input_lengths.append(len(at_ids))
            decoder_input_all = decoder_input_all + [ACT] + act_toks + [TPC] + tpc_toks
    elif len(sample["goal_path"]) > 1:
        for idx in range(len(sample["goal_path"]) - 1):
            act_toks = tokenizer.tokenize(sample["goal_path"][idx])
            tpc_toks = tokenizer.tokenize(sample["topic_path"][idx])
            at_ids = tokenizer.convert_tokens_to_ids([ACT] + act_toks + [TPC] + tpc_toks)
            decoder_input_ids.append(at_ids)
            decoder_input_lengths.append(len(at_ids))
            decoder_input_all = decoder_input_all + [ACT] + act_toks + [TPC] + tpc_toks
    else:
        raise Exception("action path is empty")
    decoder_input_all = decoder_input_all + [tokenizer.eos_token]
    decoder_input_all_ids = tokenizer.convert_tokens_to_ids(decoder_input_all)
    assert len(decoder_input_all_ids) == (sum(decoder_input_lengths) + 2)

    if is_end_flag:
        transition_number = 0
    else:
        transition_number = len(decoder_input_ids)

    inputs = {
        "input_ids": input_ids,
        "decoder_input_ids_list": decoder_input_ids,
        "decoder_input_all_ids": decoder_input_all_ids,
        "transition_ids": transition_ids,
        "start_ids": start_ids,
        "target_ids": target_ids,
        "user_utt_ids": user_utt_ids,
        "follow_ids": follow_ids,
        "transition_number": transition_number,
    }
    return inputs


def _parse_input_context(tokenizer, sample, max_seq_len=512, turn_type_size=16):
    # some fractions are not available for this setting e.g knowledge, user profile
    sample['knowledge'] = [sample['task_background']['target_topic'], "", ""]
    # last user utterance
    if len(sample["dialogue_context"]) > 0:
        idx = len(sample['dialogue_context']) - 1
        while idx > 0 and sample["dialogue_context"][idx]['role'] != 'user':
            idx -= 1
        user_utt = sample["dialogue_context"][idx]
        assert user_utt['role'] == 'user'
        user_utt_tokens = [tokenizer.bos_token] + tokenizer.tokenize(user_utt['content']) + [
            tokenizer.eos_token]
    else:
        user_utt_tokens = [tokenizer.bos_token] + [tokenizer.eos_token]
    if len(user_utt_tokens) > max_seq_len:
        user_utt_tokens = user_utt_tokens[:max_seq_len - 1] + [tokenizer.eos_token]
    user_utt_ids = tokenizer.convert_tokens_to_ids(user_utt_tokens)

    # delta: follow discrimination
    # profile_tokens = [tokenizer.bos_token]
    # for k, v in sample["user_profile"].items():
    #     k_toks = tokenizer.tokenize(k)
    #     v_toks = tokenizer.tokenize(v)
    #     profile_tokens = profile_tokens + k_toks + v_toks
    # profile_tokens = profile_tokens + [tokenizer.sep_token]
    #### user profile is not available during interactive evaluation
    profile_tokens = []
    follow_tokens = profile_tokens + user_utt_tokens[1:]
    if len(follow_tokens) > max_seq_len:
        follow_tokens = follow_tokens[:max_seq_len - 1] + [tokenizer.eos_token]
    follow_ids = tokenizer.convert_tokens_to_ids(follow_tokens)

    # S0: domain_knowledge
    kg_tokens = [tokenizer.bos_token]
    # for kg in sample["knowledge"]:
    kg_tok = tokenizer.tokenize(" ".join(sample['knowledge']))
    kg_tokens = kg_tokens + kg_tok + [tokenizer.sep_token]

    # Dialogue history
    conv_tokens = []
    history = sample["dialogue_context"]
    if len(history) > turn_type_size:
        history = history[-turn_type_size:]
    for utt in history:
        h = utt['content']
        h_toks = tokenizer.tokenize(h)
        conv_tokens = conv_tokens + h_toks
    conv_tokens = conv_tokens + [tokenizer.eos_token]
    start_tokens = kg_tokens + conv_tokens
    if len(start_tokens) > max_seq_len:
        start_tokens = start_tokens[:max_seq_len - 1] + [tokenizer.eos_token]
    start_ids = tokenizer.convert_tokens_to_ids(start_tokens)

    # transition: user profile + S0 + ST
    transition_tokens = profile_tokens + start_tokens[1:-1] + [tokenizer.sep_token]

    act_toks = tokenizer.tokenize(sample['task_background']['target_goal'])
    tpc_toks = tokenizer.tokenize(sample['task_background']['target_topic'])

    target_tokens = [ACT] + act_toks + [TPC] + tpc_toks
    transition_tokens = transition_tokens + target_tokens + [tokenizer.eos_token]
    if len(transition_tokens) > max_seq_len:
        transition_tokens = transition_tokens[:max_seq_len - 1] + [tokenizer.eos_token]
    transition_ids = tokenizer.convert_tokens_to_ids(transition_tokens)

    # ST: target
    target_tokens = [tokenizer.bos_token] + target_tokens + [tokenizer.eos_token]
    if len(target_tokens) > max_seq_len:
        target_tokens = target_tokens[:max_seq_len - 1] + [tokenizer.eos_token]
    target_ids = tokenizer.convert_tokens_to_ids(target_tokens)

    return (user_utt_ids, follow_ids, start_ids, transition_ids, target_ids)


def convert_example_to_feature_for_color_bridge_mapping(tokenizer,
                                                        instance,
                                                        max_sequence_length=512,
                                                        is_test=False):
    # do not consider user profile since it is not available for INSPIRED dataset
    user_utt_ids, follow_ids, start_ids, transition_ids, target_ids = _parse_input_context(tokenizer,
                                                                                           instance,
                                                                                           max_seq_len=max_sequence_length)
    plan_path_ids_list = _parse_plan_path(tokenizer, instance)
    transition_length = len(plan_path_ids_list) - 1
    all_features = []
    if transition_length <= 1:
        return all_features
    for idx in range(transition_length - 1):
        interim_ids = plan_path_ids_list[idx]
        inputs = {
            "user_utt_ids": user_utt_ids,
            "follow_ids": follow_ids,
            "transition_ids": transition_ids,
            "interim_ids": interim_ids,
            "start_ids": start_ids,
            "target_ids": target_ids,
            "interim_t": idx + 1,
            "target_T": transition_length,
        }
        feature = BrownianBridgeInput(**inputs)
        all_features.append(feature)
    return all_features


def convert_example_to_feature_for_color_planning(tokenizer, instance,
                                                  max_sequence_length=512,
                                                  is_test=False):
    inputs = _parse_input_context_planning(tokenizer, instance, max_seq_len=max_sequence_length)
    feature = PlannerInput(**inputs)
    return feature
