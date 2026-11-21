"""Eligibility filtering and V / J / C classification for SLAKE English QA."""

from __future__ import annotations

import re
from typing import Any

from cka_med.inference.scoring import normalize_text


# Organ keywords that must NOT appear verbatim in a Joint (J) question,
# otherwise the question self-reveals the target organ without needing visual inference.
ORGAN_LEAK_TERMS = (
    "lung",
    "lungs",
    "heart",
    "liver",
    "kidney",
    "kidneys",
    "brain",
    "spleen",
    "stomach",
    "esophagus",
    "colon",
    "rectum",
    "small bowel",
    "small intestine",
    "large intestine",
    "bladder",
    "spinal cord",
    "spine",
    "trachea",
    "larynx",
    "pharynx",
    "tooth",
    "teeth",
    "tongue",
    "thyroid",
    "pancreas",
    "gallbladder",
    "eye",
    "eyes",
    "ear",
    "ears",
)

CLINICAL_KVQA_RELATIONS = frozenset({"treat", "prevent", "cause", "symptom", "damage"})
ALLOWED_JOINT_RELATIONS = frozenset(
    {"belong to", "part of", "function", "effect", "be used for", "could", "can", "role"}
)

PRIMARY_VISUAL_CONTENT_TYPES = frozenset({"Abnormality", "Organ", "Position"})
CONTROL_VISUAL_CONTENT_TYPES = frozenset({"Modality", "Plane", "Color", "Size", "Quantity", "Shape"})


def is_cxr_candidate(row: dict[str, Any]) -> bool:
    loc = str(row.get("location") or "")
    return str(row.get("modality") or "") == "X-Ray" and ("Chest" in loc or loc == "Lung")


def _question_leaks_organ(question: str) -> str | None:
    q_low = question.lower()
    for term in ORGAN_LEAK_TERMS:
        if re.search(rf"\b{re.escape(term)}\b", q_low):
            return term
    return None


def classify_joint_family(row: dict[str, Any]) -> str | None:
    triple = row.get("triple")
    if not isinstance(triple, list) or len(triple) < 2:
        return None
    rel = normalize_text(str(triple[1]))
    if rel in ("belong to", "part of"):
        return "organ_system"
    if rel in ("function", "effect", "be used for", "could", "can", "role"):
        return "organ_function"
    return None


def classify_slake_row(
    row: dict[str, Any],
    *,
    train_j_vocab: set[str] | None = None,
) -> tuple[str, str, str | None]:
    """Classify an English SLAKE QA row into (task_type, rationale, exclusion_reason).

    Returns:
      task_type: 'V', 'J', or 'C'
      rationale: human-readable classification reason
      exclusion_reason: None if eligible for V/J core, else explicit exclusion string
    """
    if str(row.get("q_lang")) != "en":
        return "C", "non_english", "non_english"

    base_type = str(row.get("base_type") or "")
    answer_type = str(row.get("answer_type") or "")
    content_type = str(row.get("content_type") or "")
    question = str(row.get("question") or "").strip()
    norm_ans = normalize_text(str(row.get("answer") or ""))

    if not question or not norm_ans:
        return "C", "empty_qa", "empty_question_or_answer"

    if base_type == "vqa":
        if answer_type != "CLOSED":
            return "C", f"vqa_open_{content_type}", "vqa_open_ended_excluded_from_binary_core"
        if norm_ans not in ("yes", "no"):
            return "C", f"vqa_closed_non_binary_{content_type}", "vqa_closed_non_yesno"
        if content_type in PRIMARY_VISUAL_CONTENT_TYPES:
            return "V", f"visual_closed_yesno_{content_type.lower()}", None
        if content_type in CONTROL_VISUAL_CONTENT_TYPES:
            return "C", f"visual_control_{content_type.lower()}", f"visual_control_excluded_from_primary_V:{content_type}"
        return "C", f"visual_other_{content_type}", f"unrecognized_content_type:{content_type}"

    if base_type == "kvqa":
        triple = row.get("triple")
        if not isinstance(triple, list) or len(triple) < 2:
            return "C", "kvqa_missing_triple", "kvqa_missing_triple"
        rel = normalize_text(str(triple[1]))
        if rel in CLINICAL_KVQA_RELATIONS:
            return "C", f"clinical_kvqa_{rel}", f"clinical_kvqa_without_specialist_review:{rel}"
        if rel not in ALLOWED_JOINT_RELATIONS:
            return "C", f"kvqa_other_rel_{rel}", f"unsupported_kvqa_relation:{rel}"

        leaked_organ = _question_leaks_organ(question)
        if leaked_organ is not None:
            return "C", f"kvqa_organ_leaked_{leaked_organ}", f"organ_leaked_in_question:{leaked_organ}"

        # Exclude ambiguous/uninformative "None" or overly generic answers
        if norm_ans in ("none", "a little", "much", "no damage", "little"):
            return "C", "kvqa_generic_or_none_answer", f"generic_or_none_answer:{norm_ans}"

        if train_j_vocab is not None and norm_ans not in train_j_vocab:
            return "C", "kvqa_outside_train_vocab", f"answer_outside_train_vocab:{norm_ans}"

        family = classify_joint_family(row)
        return "J", f"joint_kvqa_{family}_{answer_type.lower()}", None

    return "C", f"unknown_base_type_{base_type}", f"unknown_base_type:{base_type}"
