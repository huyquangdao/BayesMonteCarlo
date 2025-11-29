"""
Utility helpers shared across the Bayes-Adaptive LLM pipeline.
"""

from typing import Dict, List
import numpy as np



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

    dialog_acts_list = list(dialog_acts)
    if 0 <= target_idx < len(dialog_acts_list):
        label = dialog_acts_list[target_idx]
    else:
        label = str(target_idx)

    prefetch_key = f"{state_rep}__{label}"
    realization_dict = realizations_vs.get(prefetch_key)
    if not realization_dict or len(realization_dict) < 2:
        return None

    best_pair = max(realization_dict.items(), key=lambda kv: kv[1])
    worst_pair = min(realization_dict.items(), key=lambda kv: kv[1])

    return target_idx, best_pair, worst_pair


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
