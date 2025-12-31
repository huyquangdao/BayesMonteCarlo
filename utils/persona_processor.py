import json
from pathlib import Path
from typing import Optional, Tuple

from tqdm import tqdm

from config.constants import BIG5_PERSONALITY, DECISION_MAKING_STYLE
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
                description = (record.get("persona_hint") or {}).get("description", "")
                personality, decision_making = _infer_persona_from_description(
                    description,
                    llm_pipeline=llm_pipeline,
                    terminators=terminators,
                    model_type=model_type,
                )

                out_record = {
                    "dialog_index": record.get("dialog_index"),
                    "turn": record.get("turn"),
                    "action": record.get("action"),
                    "hist_dialog": history,
                    "personality": personality,
                    "decision_making": decision_making,
                }
                fout.write(json.dumps(out_record, ensure_ascii=False) + "\n")
            except Exception as e:
                print(f"[persona_processor] Skipping line {idx} due to error: {e}")

    return str(out_path)
