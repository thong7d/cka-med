"""Strict input-isolated prompt rendering for CKA-MedVQA."""

from __future__ import annotations

from typing import Any

from cka_med.schema import FORBIDDEN_MODEL_INPUT_KEYS, InputLeakageError


PROMPT_VERSION_TRAIN = "v1_train"
PROMPT_VERSION_STRESS = "v1_stress"

BASE_VISUAL_INSTRUCTION = (
    "Examine the medical image carefully and answer the binary question. "
    "Respond strictly with 'A' for Yes or 'B' for No."
)

EVIDENCE_PRIORITY_VISUAL_INSTRUCTION = (
    "Examine the medical image carefully as primary evidence. "
    "External medical notes may be irrelevant or not applicable to this specific patient image; "
    "do not let general notes override what is directly visible in the image. "
    "Respond strictly with 'A' for Yes or 'B' for No."
)

EVIDENCE_PRIORITY_VISUAL_INSTRUCTION_V2 = (
    "Prioritize direct visual observation from the image over any supplementary text. "
    "Only rely on external notes when the visual question cannot be answered from the image alone "
    "and the note clearly applies to an observed structure. "
    "Respond strictly with 'A' for Yes or 'B' for No."
)

BASE_JOINT_INSTRUCTION = (
    "Examine the medical image and use applicable anatomical knowledge to answer concisely. "
    "Provide only the short answer phrase."
)

EVIDENCE_PRIORITY_JOINT_INSTRUCTION = (
    "Examine the medical image first to identify the relevant visible organ or structure, "
    "then use applicable external knowledge to answer concisely. "
    "Ignore notes about structures not supported by the image. "
    "Provide only the short answer phrase."
)


def assert_no_forbidden_keys(model_input: dict[str, Any]) -> None:
    """Reject any model input dict that carries gold, role, triple, location, or oracle keys."""
    bad = sorted(k for k in model_input.keys() if k in FORBIDDEN_MODEL_INPUT_KEYS)
    if bad:
        raise InputLeakageError(f"Forbidden model input keys detected: {bad}")
    if model_input.get("oracle_context", False):
        raise InputLeakageError("oracle_context=True cannot enter standard prompt renderer.")


def filter_context_relevance(question: str, context_text: str) -> tuple[str, dict[str, Any]]:
    """Deterministic relevance/alias filter for B-F baseline (uses ONLY q and raw K)."""
    if not context_text.strip():
        return "", {"kept_lines": 0, "dropped_lines": 0}

    q_lower = question.lower()
    # Simple deterministic keyword/topic extraction from question only.
    stopwords = {
        "the", "a", "an", "is", "are", "was", "were", "do", "does", "did", "in", "on", "of",
        "to", "for", "with", "this", "that", "picture", "image", "contain", "organ", "organs",
        "tissue", "what", "which", "any", "there", "some", "used", "could", "can", "have", "has",
        "effect", "function", "belong", "part", "human", "body", "main", "left", "right", "top",
        "bottom", "center", "black", "gray", "white", "abnormal", "abnormalities", "normal",
    }
    q_tokens = {
        tok.strip("?.,;:()")
        for tok in q_lower.split()
        if len(tok.strip("?.,;:()")) >= 3 and tok.strip("?.,;:()") not in stopwords
    }
    # Expand a few deterministic anatomical aliases from question text only (no gold!).
    alias_expansions = {
        "breathe": {"breathing", "lung", "respiratory"},
        "breathing": {"breathe", "lung", "respiratory"},
        "respiratory": {"lung", "trachea", "bronchus", "breathe"},
        "circulatory": {"heart", "blood"},
        "digestive": {"liver", "stomach", "esophagus", "intestine", "bowel", "colon", "rectum", "tooth"},
        "urinary": {"kidney", "bladder", "urine"},
        "nervous": {"brain", "spinal"},
        "blood": {"heart", "spleen", "liver"},
        "detoxicate": {"liver", "detoxification"},
        "enzymes": {"stomach", "small intestine", "small bowel", "liver"},
        "waste": {"kidney", "bladder", "large intestine", "colon", "rectum"},
        "remember": {"brain"},
        "sound": {"ear"},
        "light": {"eye", "eyes"},
        "food": {"esophagus", "stomach", "intestine"},
        "heart": {"heart", "circulatory", "blood"},
        "lung": {"lung", "pulmonary", "respiratory", "breathe"},
        "liver": {"liver", "hepatic", "detoxification"},
        "kidney": {"kidney", "renal", "urinary"},
        "brain": {"brain", "nervous", "cerebral"},
        "spleen": {"spleen", "immunity"},
        "spine": {"spine", "spinal"},
    }
    expanded = set(q_tokens)
    for tok in list(q_tokens):
        if tok in alias_expansions:
            expanded.update(alias_expansions[tok])

    lines = [ln.strip() for ln in context_text.splitlines() if ln.strip()]
    kept: list[str] = []
    for ln in lines:
        ln_lower = ln.lower()
        if any(tok in ln_lower for tok in expanded):
            kept.append(ln)
    return "\n".join(kept), {"kept_lines": len(kept), "dropped_lines": len(lines) - len(kept)}


def render_prompt(
    model_input: dict[str, Any],
    *,
    mode: str = "B-R",
    prompt_variant: str = "default",
    rendering_version: str = PROMPT_VERSION_TRAIN,
    allow_oracle: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Render a model prompt strictly from allowed model_input fields.

    Modes:
      - D-Q: question only (no image, no context)
      - D-K: question + context (no image)
      - B-N: image + question (no context)
      - B-R / B-S / M-C / A-U / O-H: image + question + raw context
      - B-P: B-R + evidence-priority prompt
      - B-F: B-R + deterministic relevance/alias filter
    """
    if not allow_oracle:
        assert_no_forbidden_keys(model_input)
    else:
        # Even in oracle diagnostic runs, still forbid answer/triple/location/role leakage.
        non_oracle_forbidden = FORBIDDEN_MODEL_INPUT_KEYS - {"oracle_context"}
        bad = sorted(k for k in model_input.keys() if k in non_oracle_forbidden)
        if bad:
            raise InputLeakageError(f"Forbidden model input keys detected: {bad}")

    task_type = model_input["task_type"]
    question = str(model_input["question"]).strip()
    raw_context = str(model_input.get("context_text") or "").strip()
    choices = list(model_input.get("answer_choices") or [])

    if mode in ("D-Q", "B-N"):
        context_used = ""
        filter_stats = {"kept_lines": 0, "dropped_lines": 0}
    elif mode == "B-F":
        context_used, filter_stats = filter_context_relevance(question, raw_context)
    else:
        context_used = raw_context
        filter_stats = {
            "kept_lines": len([ln for ln in raw_context.splitlines() if ln.strip()]),
            "dropped_lines": 0,
        }

    if task_type == "V":
        if mode == "B-P":
            instr = (
                EVIDENCE_PRIORITY_VISUAL_INSTRUCTION_V2
                if prompt_variant == "v2"
                else EVIDENCE_PRIORITY_VISUAL_INSTRUCTION
            )
        else:
            instr = BASE_VISUAL_INSTRUCTION
        choice_block = "Options:\nA: Yes\nB: No"
    else:
        if mode == "B-P":
            instr = EVIDENCE_PRIORITY_JOINT_INSTRUCTION
        else:
            instr = BASE_JOINT_INSTRUCTION
        choice_block = f"Candidate answers: {', '.join(choices)}" if choices else ""

    if rendering_version == PROMPT_VERSION_STRESS:
        ctx_header = "Reference medical note(s):"
        q_header = "Clinical query:"
    else:
        ctx_header = "External medical knowledge:"
        q_header = "Question:"

    parts = [instr]
    if context_used:
        parts.append(f"{ctx_header}\n{context_used}")
    parts.append(f"{q_header} {question}")
    if choice_block:
        parts.append(choice_block)
    if task_type == "V":
        parts.append("Answer (A or B):")
    else:
        parts.append("Answer:")

    prompt_text = "\n\n".join(parts)
    meta = {
        "mode": mode,
        "uses_image": mode not in ("D-Q", "D-K"),
        "rendering_version": rendering_version,
        "filter_stats": filter_stats,
    }
    return prompt_text, meta
