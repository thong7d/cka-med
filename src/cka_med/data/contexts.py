"""Frozen KG subset curation and controlled context pack generation."""

from __future__ import annotations

import csv
import io
from pathlib import Path
import re
from typing import Any
import zipfile

from cka_med.inference.prompts import PROMPT_VERSION_STRESS, PROMPT_VERSION_TRAIN


CONTEXT_GENERATOR_VERSION = "cka_ctx_v1"

# Curated subset of KG entries (<= 80 entries) extracted strictly from SLAKE KG.zip
# with explicit provenance row IDs. Used to construct controlled multi-candidate packs.
CURATED_ORGAN_REL_TARGETS = {
    "Lung": "Respiratory System",
    "Trachea": "Respiratory System",
    "Bronchial Tube": "Respiratory System",
    "Larynx": "Respiratory System",
    "Diaphragm": "Respiratory System",
    "Heart": "Circulatory System",
    "Artery": "Circulatory System",
    "Vein": "Circulatory System",
    "Liver": "Digestive System",
    "Stomach": "Digestive System",
    "Esophagus": "Digestive System",
    "Small Intestine": "Digestive System",
    "Colon": "Digestive System",
    "Rectum": "Digestive System",
    "Tooth": "Digestive System",
    "Kidney": "Urinary System",
    "Bladder": "Urinary System",
    "Brain": "Nervous System",
    "Spinal Cord": "Nervous System",
    "Spleen": "Lymphatic System",
    "Thymus": "Lymphatic System",
    "Spine": "Motor System",
    "Eye": "Sense Organ",
    "Ear": "Sense Organ",
}

CURATED_FUNCTION_ORGANS = {
    "Lung",
    "Heart",
    "Liver",
    "Kidney",
    "Brain",
    "Small Intestine",
    "Colon",
    "Rectum",
    "Bladder",
    "Esophagus",
    "Larynx",
    "Trachea",
    "Stomach",
    "Spleen",
    "Tooth",
    "Eye",
    "Ear",
    "Spine",
}

IRRELEVANT_ORGANS = ("Ear", "Cochlea", "Tympanic Membrane", "Ossicle", "Tooth", "Salivary Gland")


def load_curated_kg_subset(kg_zip_path: Path | str) -> dict[str, Any]:
    """Load and freeze a curated provenance-backed KG subset (30-80 entries) from SLAKE KG.zip."""
    p = Path(kg_zip_path)
    if not p.exists():
        raise FileNotFoundError(f"KG.zip not found: {p}")

    organ_rel_entries: list[dict[str, Any]] = []
    organ_func_entries: list[dict[str, Any]] = []
    disease_entries: list[dict[str, Any]] = []

    with zipfile.ZipFile(p, "r") as zf:
        rel_rows = list(csv.DictReader(io.StringIO(zf.read("KG/en_organ_rel.csv").decode("utf-8-sig")), delimiter="#"))
        for idx, r in enumerate(rel_rows, start=1):
            org = r["organ"].strip()
            val = r["value"].strip()
            if CURATED_ORGAN_REL_TARGETS.get(org) == val:
                organ_rel_entries.append(
                    {
                        "entry_id": f"en_organ_rel#{idx}",
                        "source_file": "KG/en_organ_rel.csv",
                        "source_row": idx,
                        "subject": org,
                        "attribute": r["attribute"].strip(),
                        "value": val,
                        "verification": "source_asserted",
                    }
                )

        org_rows = list(csv.DictReader(io.StringIO(zf.read("KG/en_organ.csv").decode("utf-8-sig")), delimiter="#"))
        seen_org_attr: set[tuple[str, str]] = set()
        for idx, r in enumerate(org_rows, start=1):
            org = r["organ"].strip()
            attr = r["attribute"].strip()
            val = r["value"].strip()
            if org in CURATED_FUNCTION_ORGANS and attr in ("function", "effect", "definition"):
                key = (org, attr)
                if key not in seen_org_attr and len(val) <= 140:
                    seen_org_attr.add(key)
                    organ_func_entries.append(
                        {
                            "entry_id": f"en_organ#{idx}",
                            "source_file": "KG/en_organ.csv",
                            "source_row": idx,
                            "subject": org,
                            "attribute": attr,
                            "value": val,
                            "verification": "source_asserted",
                        }
                    )

        # Select representative disease descriptions by anatomical topic for Knear distractors (<= 24 entries)
        dis_rows = list(csv.DictReader(io.StringIO(zf.read("KG/en_disease.csv").decode("utf-8-sig")), delimiter="#"))
        dis_by_name: dict[str, dict[str, Any]] = {}
        for idx, r in enumerate(dis_rows, start=1):
            name = r["disease"].strip()
            attr = r["attribute"].strip()
            val = r["value"].strip()
            rec = dis_by_name.setdefault(name, {"location": None, "description": None, "symptom": None, "rows": []})
            if attr in ("location", "description", "symptom") and rec[attr] is None:
                rec[attr] = val
                rec["rows"].append(idx)

        topic_quotas = {
            "Lung": 3,
            "Liver": 3,
            "Kidney": 3,
            "Brain": 3,
            "Heart": 3,
            "Stomach": 2,
            "Spleen": 2,
            "Spine": 2,
        }
        topic_counts = {k: 0 for k in topic_quotas}
        for name, info in dis_by_name.items():
            loc = str(info.get("location") or "")
            desc = str(info.get("description") or "")
            if not loc or not desc or len(desc) > 200:
                continue
            for topic, cap in topic_quotas.items():
                if topic_counts[topic] < cap and topic.lower() in loc.lower():
                    topic_counts[topic] += 1
                    disease_entries.append(
                        {
                            "entry_id": f"en_disease#{info['rows'][0]}",
                            "source_file": "KG/en_disease.csv",
                            "source_rows": info["rows"],
                            "subject": name,
                            "topic": topic,
                            "attribute": "reference_note",
                            "value": desc,
                            "symptom": info.get("symptom") or "",
                            "verification": "source_asserted",
                        }
                    )
                    break

    return {
        "generator_version": CONTEXT_GENERATOR_VERSION,
        "organ_rel_entries": organ_rel_entries,
        "organ_func_entries": organ_func_entries,
        "disease_entries": disease_entries,
        "total_entries": len(organ_rel_entries) + len(organ_func_entries) + len(disease_entries),
    }


def _render_entries(
    entries: list[dict[str, Any]],
    *,
    rendering_version: str = PROMPT_VERSION_TRAIN,
    reverse_order: bool = False,
) -> str:
    items = list(reversed(entries)) if reverse_order else list(entries)
    lines: list[str] = []
    for e in items:
        subj = e["subject"]
        attr = e["attribute"].replace("_", " ")
        val = e["value"]
        if rendering_version == PROMPT_VERSION_STRESS:
            lines.append(f"- Reference entry: regarding {subj}, {attr} is {val}.")
        else:
            lines.append(f"- {subj} | {attr}: {val}")
    return "\n".join(lines)


def _extract_topic_from_question_only(question: str) -> str:
    """Deterministically infer anatomical topic ONLY from question text (NEVER from gold/location)."""
    q = question.lower()
    topic_map = [
        ("lung", "Lung"),
        ("pulmonary", "Lung"),
        ("respiratory", "Lung"),
        ("chest", "Lung"),
        ("breathe", "Lung"),
        ("breathing", "Lung"),
        ("liver", "Liver"),
        ("hepatic", "Liver"),
        ("kidney", "Kidney"),
        ("renal", "Kidney"),
        ("urinary", "Kidney"),
        ("bladder", "Kidney"),
        ("brain", "Brain"),
        ("cerebral", "Brain"),
        ("nervous", "Brain"),
        ("head", "Brain"),
        ("heart", "Heart"),
        ("cardiac", "Heart"),
        ("circulatory", "Heart"),
        ("spleen", "Spleen"),
        ("stomach", "Stomach"),
        ("bowel", "Stomach"),
        ("intestine", "Stomach"),
        ("digestive", "Stomach"),
        ("abdomen", "Liver"),
        ("spine", "Spine"),
    ]
    for kw, topic in topic_map:
        if re.search(rf"\b{re.escape(kw)}\b", q):
            return topic
    return "General"


def build_context_pack(
    *,
    question: str,
    task_type: str,
    question_family: str | None,
    condition: str,
    kg_subset: dict[str, Any],
    rendering_version: str = PROMPT_VERSION_TRAIN,
    reverse_order: bool = False,
) -> tuple[str, str, list[dict[str, Any]]]:
    """Build (context_pack_id, context_text, context_source_entries) without gold/location/triple leakage."""
    if condition == "K0":
        return "pack_K0", "", []

    if condition == "Kirrel":
        # Balanced 3-line unrelated anatomical pack (Ear / Tooth / Eye)
        rel_entries = [
            e
            for e in kg_subset["organ_func_entries"]
            if e["subject"] in ("Tooth", "Eye", "Spine") and e["attribute"] in ("function", "effect", "definition")
        ][:3]
        if len(rel_entries) < 3:
            rel_entries = kg_subset["organ_rel_entries"][:3]
        pack_id = f"pack_Kirrel_{rendering_version}{'_rev' if reverse_order else ''}"
        return pack_id, _render_entries(rel_entries, rendering_version=rendering_version, reverse_order=reverse_order), rel_entries

    if condition == "Knear":
        topic = _extract_topic_from_question_only(question)
        if topic == "General":
            # Balanced cross-region reference notes (no negative cue words, matched 3-line length)
            selected: list[dict[str, Any]] = []
            seen_t: set[str] = set()
            for e in kg_subset["disease_entries"]:
                if e["topic"] not in seen_t:
                    selected.append(e)
                    seen_t.add(e["topic"])
                if len(selected) == 3:
                    break
        else:
            selected = [e for e in kg_subset["disease_entries"] if e["topic"] == topic][:3]
            if len(selected) < 3:
                for e in kg_subset["disease_entries"]:
                    if e not in selected:
                        selected.append(e)
                    if len(selected) == 3:
                        break
        pack_id = f"pack_Knear_{topic}_{rendering_version}{'_rev' if reverse_order else ''}"
        return pack_id, _render_entries(selected, rendering_version=rendering_version, reverse_order=reverse_order), selected

    if condition == "Khelp":
        # Multi-organ controlled replay pack per question family (never a single oracle answer line!)
        if question_family == "organ_system":
            wanted = {"Lung", "Heart", "Liver", "Kidney", "Brain", "Spleen", "Stomach", "Spine"}
            selected = [e for e in kg_subset["organ_rel_entries"] if e["subject"] in wanted][:8]
            pack_id = f"pack_Khelp_organ_system_{rendering_version}{'_rev' if reverse_order else ''}"
        else:
            wanted = {"Lung", "Heart", "Liver", "Kidney", "Brain", "Small Intestine", "Rectum", "Bladder", "Esophagus", "Larynx", "Trachea", "Spleen", "Tooth", "Stomach"}
            selected = [
                e
                for e in kg_subset["organ_func_entries"]
                if e["subject"] in wanted and e["attribute"] in ("function", "effect")
            ][:10]
            pack_id = f"pack_Khelp_organ_function_{rendering_version}{'_rev' if reverse_order else ''}"
        return pack_id, _render_entries(selected, rendering_version=rendering_version, reverse_order=reverse_order), selected

    raise ValueError(f"Unsupported condition for controlled pack builder: {condition}")
