"""Answer normalization and scoring helpers."""

from __future__ import annotations

import re
from typing import Iterable


BINARY_CHOICE_TO_ANSWER = {"a": "yes", "b": "no"}
ANSWER_TO_BINARY_CHOICE = {"yes": "A", "no": "B"}

ABSTAIN_MARKERS = (
    "abstain",
    "cannot determine",
    "insufficient evidence",
    "not enough information",
    "unable to determine",
    "i cannot answer",
)

# Canonicalize common multi-word/spacing/casing variants in SLAKE anatomy/function answers.
DEFAULT_ANSWER_ALIASES: dict[str, str] = {
    "yes": "yes",
    "no": "no",
    "y": "yes",
    "n": "no",
    "lungs": "lung",
    "eyes": "eye",
    "kidneys": "kidney",
    "spinal cord": "spinal cord",
    "small bowel": "small bowel",
    "small intestine": "small bowel",
    "colon, small bowel": "colon, small bowel",
    "rectum, small bowel": "rectum, small bowel",
    "promote blood flow": "promote blood flow",
    "breathing": "breathe",
    "breathe": "breathe",
    "respiration": "breathe",
}


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    s = str(text).strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.strip(" \t\r\n.,;:!?\"'")
    return s


def normalize_answer(
    raw_text: str | None,
    *,
    task_type: str,
    aliases: dict[str, str] | None = None,
    allowed_vocab: Iterable[str] | None = None,
) -> tuple[str | None, bool, bool]:
    """Return (normalized_answer, abstained, valid_output)."""
    cleaned = normalize_text(raw_text)
    if not cleaned:
        return None, False, False

    for marker in ABSTAIN_MARKERS:
        if cleaned == marker or cleaned.startswith(marker):
            return None, True, True

    alias_map = dict(DEFAULT_ANSWER_ALIASES)
    if aliases:
        for k, v in aliases.items():
            alias_map[normalize_text(k)] = normalize_text(v)

    if task_type == "V":
        # Support direct choice tokens A/B or yes/no prefix.
        first_tok = cleaned.split()[0].strip("().:,")
        if first_tok in BINARY_CHOICE_TO_ANSWER:
            return BINARY_CHOICE_TO_ANSWER[first_tok], False, True
        if cleaned in ("yes", "no"):
            return cleaned, False, True
        if first_tok in ("yes", "no"):
            return first_tok, False, True
        return None, False, False

    # Task J or C
    mapped = alias_map.get(cleaned, cleaned)
    if allowed_vocab is not None:
        vocab_norm = {alias_map.get(normalize_text(v), normalize_text(v)) for v in allowed_vocab}
        if mapped not in vocab_norm:
            # Also try exact token/phrase match against choice letter if present (e.g., "A: Lung")
            m = re.match(r"^[a-z]\s*[:.)-]\s*(.+)$", cleaned)
            if m:
                inner = alias_map.get(normalize_text(m.group(1)), normalize_text(m.group(1)))
                if inner in vocab_norm:
                    return inner, False, True
            return mapped, False, False
    return mapped, False, True
