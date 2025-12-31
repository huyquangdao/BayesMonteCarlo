"""
Utility helpers shared across the Bayes-Adaptive LLM pipeline.
"""

import os
from typing import Dict, List, Optional, Tuple
import numpy as np
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn as nn
from peft import PeftModel



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


def _pair_top_actions(probabilities, state_rep: str, dialog_acts, valid_moves, realizations_vs, preferred_action: Optional[int] = None):
    """
    Fallback: gather realizations across actions, prioritizing top-2 by probability
    (and always including preferred_action if provided).
    """
    dialog_acts_list = list(dialog_acts)
    valid_moves_list = [int(action_idx) for action_idx in valid_moves]
    prob_pairs = [(idx, float(probabilities[idx])) for idx in valid_moves_list]
    top_actions = [p[0] for p in sorted(prob_pairs, key=lambda x: x[1], reverse=True)[:2]]
    if preferred_action is not None:
        try:
            pref_idx = int(preferred_action)
            top_actions = list(dict.fromkeys([pref_idx] + top_actions))
        except (TypeError, ValueError):
            pass

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
    selected_action: Optional[int] = None,
):
    """
    Select the best/worst realization for a chosen action (if provided) or the most likely action.
    Returns (action_idx, best_pair, worst_pair) where each pair is (utterance, value).
    """
    if not realizations_vs:
        return None

    if probabilities is None or len(probabilities) == 0:
        return None

    valid_moves_list = [int(action_idx) for action_idx in valid_moves]
    if not valid_moves_list:
        return None

    target_idx: Optional[int] = None
    if selected_action is not None:
        try:
            candidate_idx = int(selected_action)
        except (TypeError, ValueError):
            candidate_idx = None
        if candidate_idx is not None and 0 <= candidate_idx < len(probabilities):
            if candidate_idx in valid_moves_list:
                target_idx = candidate_idx
            else:
                logger.debug("Selected action {} not in valid moves; falling back to probabilities.", selected_action)
        else:
            logger.debug("Selected action {} is out of bounds; falling back to probabilities.", selected_action)

    if target_idx is None:
        best_prob = -float("inf")
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
    return _pair_top_actions(probabilities, state_rep, dialog_acts, valid_moves_list, realizations_vs, preferred_action=target_idx)

def load_persona_infer_model(model_config, device, log=logger) -> Tuple[Optional[torch.nn.Module], Optional[AutoTokenizer]]:
    """
    Load persona inference model/tokenizer from a saved SFT checkpoint.
    Returns (model, tokenizer) or (None, None) on failure.
    """
    model_dir = os.path.join(getattr(model_config, "saved_dir", ""), getattr(model_config, "persona_sft_model_folder"))
    print(f"[LOAD] >>> Requested to load infer persona model from: {model_dir}")
    if not model_dir or not os.path.exists(model_dir):
        log.warning("Persona SFT model dir not found ({}); using main model for persona inference.", model_dir)
        return None, None

    meta_path = os.path.join(model_dir, "meta.pt")
    meta = torch.load(meta_path, map_location="cpu") if os.path.exists(meta_path) else {}
    base_model_name = meta.get("base_model_name") or getattr(model_config, "plm", None) or model_dir
    saved_format = meta.get("saved_format")
    dtype = torch.bfloat16 if getattr(model_config, "bf16", True) else None

    try:
        tokenizer = AutoTokenizer.from_pretrained(base_model_name, cache_dir=getattr(model_config, "cached_dir", None))
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
    except Exception as exc:  # pragma: no cover
        log.warning("Failed to load persona tokenizer {}: {}", base_model_name, exc)
        return None, None

    try:
        if saved_format == "lora_adapter":
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                cache_dir=getattr(model_config, "cached_dir", None),
                torch_dtype=dtype,
            )
            adapter_dir = os.path.join(model_dir, "lora_adapter")
            model = PeftModel.from_pretrained(base_model, adapter_dir)
        elif saved_format == "full_state_dict":
            model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                cache_dir=getattr(model_config, "cached_dir", None),
                torch_dtype=dtype,
            )
            state_dict = torch.load(os.path.join(model_dir, "model.pth"), map_location="cpu")
            model.load_state_dict(state_dict)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_dir,
                cache_dir=getattr(model_config, "cached_dir", None),
                torch_dtype=dtype,
            )

        model.to(device)
        log.info("Loaded persona inference model from {}", model_dir)
        return model, tokenizer
    except Exception as exc:  # pragma: no cover
        log.warning("Failed to load persona inference model from {}: {}", model_dir, exc)
        return None, None
    
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

def has_meta_checkpoint(save_dir: str) -> bool:
    meta_path = os.path.join(save_dir, "meta.pt")
    return os.path.exists(meta_path)

def _load_plm_from_meta(base_model, save_dir: str, saved_format: str):
    print(f"[META] _load_plm_from_meta called with saved_format={saved_format}")
    print(f"[META] base_model type before load: {type(base_model)}")

    if saved_format == "lora_adapter":
        adapter_dir = os.path.join(save_dir, "lora_adapter")
        print(f"[META] Trying to load LoRA adapter from: {adapter_dir}")

        if not os.path.isdir(adapter_dir):
            raise FileNotFoundError(f"LoRA adapter dir not found: {adapter_dir}")
        
        plm = PeftModel.from_pretrained(
            base_model,
            adapter_dir,
            device_map={"": "cpu"},
        )
        print(f"[META] LoRA adapter loaded. New plm type: {type(plm)}")
        return plm

    if saved_format == "full_state_dict":
        ckpt_path = os.path.join(save_dir, "model.pth")
        print(f"[META] Trying to load full_state_dict from: {ckpt_path}")

        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        state_dict = torch.load(ckpt_path, map_location="cpu")
        print(f"[META] Loaded state_dict with {len(state_dict)} keys")
        base_model.load_state_dict(state_dict, strict=False)
        print(f"[META] State dict loaded into base_model, type now: {type(base_model)}")
        return base_model

    raise ValueError(f"Unknown saved_format in meta.pt: {saved_format}")


def load_legacy_checkpoint(self, save_dir: str, is_rl: bool) -> None:
    ckpt_name = "rl_model.pth" if is_rl else "model.pth"
    ckpt_path = os.path.join(save_dir, ckpt_name)

    print(f"[LEGACY] Trying to load legacy checkpoint from: {ckpt_path}")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No pretrained model found at {ckpt_path}")

    self.trainer.model = self.model

    device = getattr(self, "device", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[LEGACY] Using device: {device}")

    self.model = load_model(self.trainer, ckpt_path, device=device)

    print(f"[LEGACY] After load, self.model type: {type(self.model)}")
    plm_obj = getattr(self.model, "plm", self.model)
    print(f"[LEGACY] After load, self.model.plm type: {type(plm_obj)}")
    print(f"[LEGACY] plm device: {getattr(plm_obj, 'device', 'unknown')}")


def load_from_meta_checkpoint(self, save_dir: str) -> None:
    meta_path = os.path.join(save_dir, "meta.pt")
    print(f"[META] Loading meta from: {meta_path}")
    meta = torch.load(meta_path, map_location="cpu")

    print(f"[META] meta content: {meta}")
    saved_format = meta.get("saved_format", "full_state_dict")
    print(f"[META] saved_format = {saved_format}")

    base_model = getattr(self.model, "plm", None)
    print(f"[META] base_model type before _load_plm_from_meta: {type(base_model)}")

    plm = _load_plm_from_meta(base_model, save_dir, saved_format)

    print(f"[META] plm type returned from _load_plm_from_meta: {type(plm)}")

    if hasattr(self.model, "plm"):
        print("[META] Assigning loaded plm to self.model.plm")
        self.model.plm = plm
    else:
        print("[META] self.model has no 'plm', replacing self.model by plm backbone")
        self.model = plm

    self.trainer.model = self.model
    print(f"[META] After meta load, self.trainer.model type: {type(self.trainer.model)}")
    plm_obj = getattr(self.trainer.model, "plm", self.trainer.model)
    print(f"[META] After meta load, plm device: {getattr(plm_obj, 'device', 'unknown')}")


def save_finetuned_model(self, save_dir=None):
    save_dir = save_dir or self.model_config.saved_dir
    os.makedirs(save_dir, exist_ok=True)

    model = getattr(self.model, "plm", self.model)

    meta = {
        "base_model_name": getattr(self.model_config, "plm", None),
        "use_lora": isinstance(model, PeftModel),
    }

    if isinstance(model, PeftModel):
        # chỉ lưu LoRA adapter
        adapter_dir = os.path.join(save_dir, "lora_adapter")
        model.save_pretrained(adapter_dir)
        meta["saved_format"] = "lora_adapter"
        torch.save(meta, os.path.join(save_dir, "meta.pt"))
        print(f"[SAVE] LoRA adapter saved to {adapter_dir}")
    else:
        # fallback: full state_dict
        ckpt_path = os.path.join(save_dir, "model.pth")
        torch.save(model.state_dict(), ckpt_path)
        meta["saved_format"] = "full_state_dict"
        torch.save(meta, os.path.join(save_dir, "meta.pt"))
        print(f"[SAVE] Full model state_dict saved to {ckpt_path}")
