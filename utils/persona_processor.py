import json
from pathlib import Path
from typing import Optional, Tuple

from tqdm import tqdm

from config.constants import BIG5_PERSONALITY, DECISION_MAKING_STYLE, INFER_PERSONA_PROMPT, PREFERENCE_PAIR_PROMPT_P4G, BIG5_PERSONALITY_DES
from utils.prompt import call_llm


def _infer_persona_from_description(
    description: str,
    llm_pipeline=None,
    terminators=None,
    model_type: str = "chatgpt",
) -> Tuple[Optional[str], Optional[str]]:
    """
    Infer Big5 personality and decision-making style from a free-form persona description.
    Returns (personality, decision_making) where personality is one of BIG5_PERSONALITY or None.
    """
    if not description:
        return None, None

    prompt = [
        {
            "role": "system",
            "content": (
                "You are an annotator. Read the persona description and return a JSON object with keys "
                '"personality" (one of {openness, conscientiousness, extraversion, agreeableness, neuroticism}, lowercase) '
                'and "decision_making" (one of {directive, analytical, conceptual, behavioral}, lowercase). '
                "Return JSON only."
            ),
        },
        {
            "role": "user",
            "content": f"Persona description:\n{description}\nJSON:",
        },
    ]

    response = ""
    try:
        responses = call_llm(
            prompt,
            n=1,
            temperature=0.2,
            max_token=128,
            model_type=model_type,
            llm_pipeline=llm_pipeline,
            terminators=terminators,
        )
        response = responses[0] if responses else ""
    except Exception as e:
        print(f"[persona_processor] LLM call failed: {e}")
        return None, None

    # Parse structured JSON first; fall back to heuristic extraction
    personality, decision_making = _parse_persona_response(response)
    return personality, decision_making


def _parse_persona_response(response: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse LLM response into (personality, decision_making) with fallbacks."""
    if not response:
        return None, None

    # Try JSON first
    try:
        parsed = json.loads(response)
    except Exception:
        parsed = None

    personality = None
    decision_making = None

    if isinstance(parsed, dict):
        personality = (parsed.get("personality") or "").strip().lower()
        decision_making = parsed.get("decision_making") or parsed.get("decision-making")
        if decision_making is not None:
            decision_making = str(decision_making).strip()

    # Heuristic fallback: search for big5 keywords in raw text
    if personality not in BIG5_PERSONALITY:
        lower = response.lower()
        for trait in BIG5_PERSONALITY:
            if trait in lower:
                personality = trait
                break
        else:
            personality = None

    if decision_making is None:
        lower = response.lower()
        for style in DECISION_MAKING_STYLE:
            if style in lower:
                decision_making = style
                break
        if decision_making is None:
            decision_making = response.strip()

    return personality, decision_making


def _extract_history(record: dict) -> str:
    """Extract dialogue history string from a preference-pair record."""
    prompt = record.get("prompt") or ""
    marker = "Conversation so far:"
    if marker in prompt:
        return prompt.split(marker, 1)[1].strip()

    parts = []
    if record.get("system_utterance"):
        parts.append(f"System: {record['system_utterance']}")
    if record.get("user_utterance"):
        parts.append(f"User: {record['user_utterance']}")
    return "\n".join(parts).strip()


def process_persona_file(
    input_path: str,
    output_path: Optional[str] = None,
    llm_pipeline=None,
    terminators=None,
    model_type: str = "chatgpt",
) -> str:
    """
    Read a JSONL file of preference pairs and write a simplified JSONL with
    dialog_index, turn, action, hist_dialog, personality, decision_making.
    """
    in_path = Path(input_path)
    if output_path is None:
        output_path = in_path.with_suffix(".persona.jsonl")
    out_path = Path(output_path)

    # count lines for a bounded progress bar
    with in_path.open("r", encoding="utf-8") as fin:
        total_lines = sum(1 for line in fin if line.strip())

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(tqdm(fin, total=total_lines, desc="Preprocessing personas")):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                history = _extract_history(record)
                persona_hint_raw = record.get("persona_hint") or {}
                description = persona_hint_raw.get("description", "")
                personality, decision_making = _infer_persona_from_description(
                    description,
                    llm_pipeline=llm_pipeline,
                    terminators=terminators,
                    model_type=model_type,
                )

                # Preserve original preference pair structure and enrich with persona fields
                persona_label = personality or persona_hint_raw.get("personality")
                persona_description = BIG5_PERSONALITY_DES.get(personality) if personality else ""

                prompt_body = history
                prompt = PREFERENCE_PAIR_PROMPT_P4G.format(persona_label or "unknown", persona_description or "").rstrip()
                if prompt_body:
                    prompt += "\n" + prompt_body

                base_record = {
                    "prompt": prompt,
                    "chosen": record.get("chosen"),
                    "rejected": record.get("rejected"),
                    "dialog_index": record.get("dialog_index"),
                    "turn": record.get("turn"),
                    "action": record.get("action"),
                    "system_utterance": record.get("system_utterance"),
                    "user_utterance": record.get("user_utterance"),
                }

                out_record = {
                    **base_record,
                    "hist_dialog": history,
                }

                # preserve/enrich persona hint
                persona_hint = record.get("persona_hint") or {}
                if description:
                    persona_hint["description"] = description
                if personality is not None:
                    persona_hint["personality"] = personality
                if decision_making is not None:
                    persona_hint["decision_making"] = decision_making
                out_record["persona_hint"] = persona_hint
                fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[persona_processor] Skipping line {idx} due to error: {e}")

    return str(out_path)


def build_personality_sft_data(
    input_path: str,
    output_path: Optional[str] = None,
    prompt_template: str = INFER_PERSONA_PROMPT,
    target_key: str = "personality",
    use_prompt_template: bool = True,
    include_text: bool = True,
) -> str:
    """
    Convert a persona jsonl file (e.g., the output of process_persona_file) into
    SFT-ready chat records. Supports switching between prompt-based construction
    and using pre-existing messages to avoid duplication.

    Each output row contains:
      - messages: chat-style list ready for tokenizer.apply_chat_template
      - text (optional): plain text concatenation for training without chat templates
    """
    in_path = Path(input_path)
    if output_path is None:
        output_path = in_path.with_suffix(".sft.jsonl")
    out_path = Path(output_path)

    # Pre-compute a lightweight system prompt so we do not repeat the template in the user turn.
    sys_prompt = prompt_template.split("{dialogue_history}", 1)[0].strip() or "You are a personality inference classifier."

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for idx, line in enumerate(fin):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except Exception:
                continue

            history = (record.get("hist_dialog") or record.get("dialogue") or record.get("context") or "").strip()
            label = (record.get(target_key) or "").strip()
            if not history or not label:
                continue

            # If we already have messages and the caller disables prompt templating, reuse them.
            messages = record.get("messages") if not use_prompt_template else None
            if not messages:
                messages = [
                    {"role": "system", "content": sys_prompt},
                    {"role": "user", "content": history},
                    {"role": "assistant", "content": label},
                ]
            else:
                # ensure we end with the target label
                if not (messages and messages[-1].get("role") == "assistant"):
                    messages = list(messages) + [{"role": "assistant", "content": label}]

            out_record = {"messages": messages}

            if include_text:
                text = "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in messages)
                out_record["text"] = text

            fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")

    return str(out_path)
