"""
Utility helpers shared across the Bayes-Adaptive LLM pipeline.
"""

import os
from typing import Dict, List, Optional
import numpy as np
from loguru import logger
import torch




def stringify_dialogue_context(dialogue_context: List[Dict[str, str]]) -> str:
    """
    Convert a dialogue context (list of role/content dicts) to a plain text prompt.
    """
    formatted = []
    for utt in dialogue_context:
        role = utt.get("role", "assistant")
        speaker = "Persuader" if role == "assistant" else "Persuadee"
        formatted.append(f"{speaker}: {utt.get('content', '').strip()}")
    return "\n".join(formatted)


def sanitize_persona_description(description: str) -> str:
    """
    Strip leading introductions like 'Meet Alex,' to reduce personalization.
    Returns a concise persona hint for prompting.
    """
    if not description:
        return ""
    desc = description.strip()
    if desc.lower().startswith("meet "):
        parts = desc.split(".", 1)
        if len(parts) == 2:
            desc = parts[1].strip()
        else:
            desc = desc.replace("Meet ", "", 1).strip()
    return desc


def _pair_single_action(state_rep: str, target_idx: int, dialog_acts, realizations_vs):
    """
    Try to form a pair within a single action using its realizations.
    """
    dialog_acts_list = list(dialog_acts)
    if 0 <= target_idx < len(dialog_acts_list):
        label = dialog_acts_list[target_idx]
    else:
        label = str(target_idx)

    prefetch_key = f"{state_rep}__{label}"
    realization_dict = realizations_vs.get(prefetch_key)
    if not realization_dict or len(realization_dict) < 2:
        return None

    sorted_pairs = sorted(realization_dict.items(), key=lambda kv: kv[1])
    worst_pair = sorted_pairs[0]
    best_pair = sorted_pairs[-1]
    if best_pair[0] == worst_pair[0]:
        return None
    return target_idx, best_pair, worst_pair


def _pair_top_actions(probabilities, state_rep: str, dialog_acts, valid_moves, realizations_vs):
    """
    Fallback: gather realizations across actions, prioritizing top-2 by probability.
    """
    dialog_acts_list = list(dialog_acts)
    valid_moves_list = [int(action_idx) for action_idx in valid_moves]
    prob_pairs = [(idx, float(probabilities[idx])) for idx in valid_moves_list]
    top_actions = [p[0] for p in sorted(prob_pairs, key=lambda x: x[1], reverse=True)[:2]]

    all_entries = []
    for action_idx in valid_moves_list:
        if 0 <= action_idx < len(dialog_acts_list):
            lbl = dialog_acts_list[action_idx]
        else:
            lbl = str(action_idx)
        key = f"{state_rep}__{lbl}"
        entries = realizations_vs.get(key, {})
        for utt, v in entries.items():
            all_entries.append((action_idx, utt, v))

    if top_actions:
        filtered = [(a, u, v) for (a, u, v) in all_entries if a in top_actions]
        if len(filtered) >= 2:
            all_entries = filtered

    if len(all_entries) < 2:
        return None

    all_entries_sorted = sorted(all_entries, key=lambda tup: tup[2])
    worst_entry = all_entries_sorted[0]
    best_entry = all_entries_sorted[-1]
    if best_entry[1] == worst_entry[1] and len(all_entries_sorted) > 2:
        for cand in all_entries_sorted[1:]:
            if cand[1] != best_entry[1]:
                worst_entry = cand
                break
    if best_entry[1] == worst_entry[1]:
        return None
    best_idx, best_utt, best_v = best_entry
    _, worst_utt, worst_v = worst_entry
    return best_idx, (best_utt, best_v), (worst_utt, worst_v)


def get_preference_pair(
    probabilities,
    state_rep: str,
    dialog_acts,
    valid_moves,
    realizations_vs,
):
    """
    Select the best/worst realization for the most likely action from an OpenLoopMCTS search.
    Returns (action_idx, best_pair, worst_pair) where each pair is (utterance, value).
    """
    if not realizations_vs:
        return None

    probabilities = probabilities
    if probabilities is None or len(probabilities) == 0:
        return None

    valid_moves_list = [int(action_idx) for action_idx in valid_moves]
    if not valid_moves_list:
        return None

    best_prob = -float("inf")
    target_idx = None
    for action_idx in valid_moves_list:
        prob_val = float(probabilities[action_idx])
        if prob_val > best_prob:
            best_prob = prob_val
            target_idx = action_idx

    if target_idx is None:
        return None

    # First try within the most likely action.
    single_action_pair = _pair_single_action(state_rep, target_idx, dialog_acts, realizations_vs)
    if single_action_pair:
        logger.info("Selected single-action preference pair for action {}", target_idx)
        return single_action_pair
    logger.info("No single-action pair found for action {}, trying cross-action.", target_idx)
    # Fallback: cross-action using top-2 actions.
    return _pair_top_actions(probabilities, state_rep, dialog_acts, valid_moves, realizations_vs)


def coerce_to_float(value, default):
    """
    Convert common numeric representations (int/float/strings) to float, else return default.
    """
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        s = value.strip()
        import re
        if re.fullmatch(r"[+-]?\\d*\\.?\\d+(e[+-]?\\d+)?", s, re.IGNORECASE):
            return float(s)
    return default

import torch.nn as nn

def load_model(trainer, load_file_path: str, device: Optional[torch.device] = None):
    if device is None:
        device = getattr(
            trainer,
            "device",
            torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        )

    if not os.path.isfile(load_file_path):
        raise FileNotFoundError(f"Checkpoint not found: {load_file_path}")

    logger.info("Loading checkpoint from {} to {}", load_file_path, device)

    # KHÔNG map thẳng lên GPU để tránh OOM khi load
    obj = torch.load(load_file_path, map_location="cpu")

    if isinstance(obj, dict):
        state_dict = obj
        logger.info("Checkpoint type: state_dict (dict)")
    elif isinstance(obj, nn.Module):
        logger.info(
            "Checkpoint type: full nn.Module ({}), extracting state_dict",
            type(obj),
        )
        state_dict = obj.state_dict()
    else:
        raise TypeError(
            f"Unexpected checkpoint type: {type(obj)}. "
            "Expected dict (state_dict) or nn.Module."
        )

    model = getattr(trainer, "model", None)
    if model is None:
        raise RuntimeError(
            "trainer.model is None in load_model. "
            "Ensure model architecture is built before calling load_model."
        )

    if hasattr(model, "module"):
        model_to_load = model.module
    else:
        model_to_load = model

    missing, unexpected = model_to_load.load_state_dict(state_dict, strict=False)
    if missing:
        logger.warning("Missing keys when loading: {}", missing)
    if unexpected:
        logger.warning("Unexpected keys when loading: {}", unexpected)

    model.to(device)
    trainer.model = model
    return trainer.model
