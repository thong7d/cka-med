"""CLI to prepare train/dev/test manifests, pixel-deduplicated datasets, and KG subset."""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
import random
from typing import Any

import yaml

from cka_med.data.contexts import CONTEXT_GENERATOR_VERSION, build_context_pack, load_curated_kg_subset
from cka_med.data.filter import classify_joint_family, classify_slake_row, is_cxr_candidate
from cka_med.data.ingest import (
    SLAKE_PINNED_REVISION,
    SLAKE_REPO_ID,
    audit_images_and_dedup,
    ensure_slake_images,
)
from cka_med.inference.prompts import PROMPT_VERSION_TRAIN
from cka_med.inference.scoring import DEFAULT_ANSWER_ALIASES, normalize_text
from cka_med.manifest import bangkok_now_iso, save_json, sha256_file
from cka_med.schema import DataRecord, write_jsonl


def _canonicalize_answer(raw_ans: str) -> str:
    norm = normalize_text(raw_ans)
    return DEFAULT_ANSWER_ALIASES.get(norm, norm)


def _select_groups_deterministic(
    eligible_groups: list[dict[str, Any]],
    *,
    target_v: int,
    target_j: int,
    seed: int,
    prefer_all_cxr: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select deterministic V and J subsets by seed without peeking at predictions."""
    rng = random.Random(seed)
    v_groups = [g for g in eligible_groups if g["task_type"] == "V"]
    j_groups = [g for g in eligible_groups if g["task_type"] == "J"]

    def _pick(pool: list[dict[str, Any]], k: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if len(pool) <= k:
            return list(pool), []
        if prefer_all_cxr:
            cxr_items = [g for g in pool if g["cxr_candidate"]]
            non_cxr = [g for g in pool if not g["cxr_candidate"]]
            rng.shuffle(cxr_items)
            rng.shuffle(non_cxr)
            if len(cxr_items) >= k:
                chosen = cxr_items[:k]
                rest = cxr_items[k:] + non_cxr
            else:
                need = k - len(cxr_items)
                chosen = cxr_items + non_cxr[:need]
                rest = non_cxr[need:]
            return chosen, rest
        shuffled = list(pool)
        rng.shuffle(shuffled)
        return shuffled[:k], shuffled[k:]

    # Sort before shuffle for 100% determinism independent of dict order
    v_groups.sort(key=lambda g: g["group_id"])
    j_groups.sort(key=lambda g: g["group_id"])
    v_keep, v_drop = _pick(v_groups, target_v)
    j_keep, j_drop = _pick(j_groups, target_j)
    return v_keep + j_keep, v_drop + j_drop


def prepare_dataset(config: dict[str, Any]) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    raw_meta_dir = Path(config["raw_metadata_dir"])
    raw_img_dir = Path(config["raw_images_dir"])
    out_dir = Path(config["processed_dir"])
    manifest_path = Path(config["manifest_path"])
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    imgs_dir = ensure_slake_images(
        raw_img_dir,
        repo_id=config.get("dataset_repo", SLAKE_REPO_ID),
        revision=config.get("dataset_revision", SLAKE_PINNED_REVISION),
    )

    split_raw: dict[str, list[dict[str, Any]]] = {}
    for sp in ("train", "validation", "test"):
        raw_rows = json.loads((raw_meta_dir / f"{sp}.json").read_text(encoding="utf-8"))
        split_raw[sp] = [r for r in raw_rows if r.get("q_lang") == "en"]

    dedup_audit = audit_images_and_dedup(
        split_raw,
        imgs_dir,
        near_dup_hamming_threshold=int(config.get("near_dup_hamming_threshold", 2)),
    )
    excluded_imgs = dedup_audit["excluded_images"]
    image_info = dedup_audit["image_info"]

    kg_zip = raw_meta_dir / "KG.zip"
    kg_subset = load_curated_kg_subset(kg_zip)
    kg_subset_path = out_dir / "kg_subset.json"
    save_json(kg_subset_path, kg_subset)

    # First pass on train ONLY to build locked J answer vocabulary.
    train_j_vocab: set[str] = set()
    for row in split_raw["train"]:
        img_key = f"train::{row['img_name']}"
        if img_key in excluded_imgs:
            continue
        t_tmp, _, excl_tmp = classify_slake_row(row, train_j_vocab=None)
        if t_tmp == "J" and excl_tmp is None:
            train_j_vocab.add(_canonicalize_answer(str(row["answer"])))

    # Filter all splits using locked train_j_vocab
    exclusions: list[dict[str, Any]] = []
    eligible_by_split: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}

    for sp in ("train", "validation", "test"):
        for row in split_raw[sp]:
            img_name = str(row["img_name"])
            img_key = f"{sp}::{img_name}"
            qid = str(row["qid"])
            if img_key in excluded_imgs:
                exclusions.append(
                    {
                        "split": sp,
                        "source_qid": qid,
                        "img_name": img_name,
                        "task_type_raw": row.get("base_type"),
                        "question": row.get("question"),
                        "exclusion_reason": excluded_imgs[img_key],
                    }
                )
                continue

            task_type, rationale, excl_reason = classify_slake_row(row, train_j_vocab=train_j_vocab)
            if excl_reason is not None or task_type not in ("V", "J"):
                exclusions.append(
                    {
                        "split": sp,
                        "source_qid": qid,
                        "img_name": img_name,
                        "task_type_raw": row.get("base_type"),
                        "question": row.get("question"),
                        "exclusion_reason": excl_reason or "non_vj_task",
                    }
                )
                continue

            info = image_info[img_key]
            group_id = f"slake_{sp}_q{qid}"
            eligible_by_split[sp].append(
                {
                    "group_id": group_id,
                    "source_qid": qid,
                    "image_id": str(row["img_id"]),
                    "img_name": img_name,
                    "image_path": info["path"],
                    "image_sha256": info["sha256"],
                    "image_dhash": info["dhash"],
                    "modality": str(row["modality"]),
                    "content_type": str(row["content_type"]),
                    "cxr_candidate": is_cxr_candidate(row),
                    "original_question": str(row["question"]).strip(),
                    "original_answer": str(row["answer"]).strip(),
                    "normalized_answer": _canonicalize_answer(str(row["answer"])),
                    "task_type": task_type,
                    "question_family": classify_joint_family(row) if task_type == "J" else None,
                    "task_eligibility_rationale": rationale,
                    "split": "dev" if sp == "validation" else sp,
                }
            )

    budgets = config["budgets"]
    split_map = {"train": "train", "validation": "dev", "test": "test"}
    selected_records_by_split: dict[str, list[dict[str, Any]]] = {}
    j_vocab_sorted = sorted(train_j_vocab)

    for raw_sp, out_sp in split_map.items():
        sp_cfg = budgets[out_sp]
        chosen, dropped_budget = _select_groups_deterministic(
            eligible_by_split[raw_sp],
            target_v=int(sp_cfg["target_v_groups"]),
            target_j=int(sp_cfg["target_j_groups"]),
            seed=seed + {"train": 0, "dev": 1, "test": 2}[out_sp],
            prefer_all_cxr=bool(sp_cfg.get("prefer_all_cxr", False)),
        )
        for d in dropped_budget:
            exclusions.append(
                {
                    "split": raw_sp,
                    "source_qid": d["source_qid"],
                    "img_name": d["img_name"],
                    "task_type_raw": d["task_type"],
                    "question": d["original_question"],
                    "exclusion_reason": "deterministic_budget_cap",
                }
            )

        records: list[dict[str, Any]] = []
        for g in chosen:
            conditions = ("K0", "Kirrel", "Knear") if g["task_type"] == "V" else ("K0", "Khelp", "Kirrel")
            for cond in conditions:
                pack_id, ctx_text, ctx_entries = build_context_pack(
                    question=g["original_question"],
                    task_type=g["task_type"],
                    question_family=g["question_family"],
                    condition=cond,
                    kg_subset=kg_subset,
                    rendering_version=PROMPT_VERSION_TRAIN,
                    reverse_order=False,
                )
                rec = DataRecord(
                    sample_id=f"{g['group_id']}__{cond}",
                    group_id=g["group_id"],
                    image_id=g["image_id"],
                    img_name=g["img_name"],
                    source_dataset="BoKelvin/SLAKE",
                    source_version=SLAKE_PINNED_REVISION,
                    source_qid=g["source_qid"],
                    image_path=g["image_path"],
                    image_sha256=g["image_sha256"],
                    image_dhash=g["image_dhash"],
                    modality=g["modality"],
                    original_question=g["original_question"],
                    original_answer=g["original_answer"],
                    normalized_answer=g["normalized_answer"],
                    task_type=g["task_type"],
                    task_eligibility_rationale=g["task_eligibility_rationale"],
                    split=out_sp,
                    condition=cond,
                    context_pack_id=pack_id,
                    context_text=ctx_text,
                    context_source_entries=ctx_entries,
                    rendering_version=PROMPT_VERSION_TRAIN,
                    gold_invariant=True,
                    provenance_quality="accepted",
                    reviewer_type="agent_audited",
                    exclusion_reason=None,
                    patient_id=None,
                    premise_label=None,
                    verification="not_established",
                    oracle_context=False,
                    synthetic_fixture=False,
                    content_type=g["content_type"],
                    cxr_candidate=g["cxr_candidate"],
                    question_family=g["question_family"],
                    answer_choices=["yes", "no"] if g["task_type"] == "V" else j_vocab_sorted,
                )
                records.append(rec.to_dict())

        selected_records_by_split[out_sp] = records
        write_jsonl(out_dir / f"{out_sp}.jsonl", records)

    # Save exclusions CSV
    excl_csv_path = out_dir / "exclusions.csv"
    with excl_csv_path.open("w", newline="", encoding="utf-8") as fh:
        fieldnames = ["split", "source_qid", "img_name", "task_type_raw", "question", "exclusion_reason"]
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(exclusions)

    # Save dedup audit details
    dedup_path = out_dir / "split_dedup_audit.json"
    save_json(
        dedup_path,
        {
            "total_unique_split_images": dedup_audit["total_unique_split_images"],
            "exact_cross_split_dups": dedup_audit["exact_cross_split_dups"],
            "near_cross_split_dups": dedup_audit["near_cross_split_dups"],
            "excluded_images": dedup_audit["excluded_images"],
        },
    )

    # Build manifest
    split_stats = {}
    for sp, recs in selected_records_by_split.items():
        groups = {r["group_id"]: r for r in recs}
        v_g = [g for g in groups.values() if g["task_type"] == "V"]
        j_g = [g for g in groups.values() if g["task_type"] == "J"]
        split_stats[sp] = {
            "n_records": len(recs),
            "n_groups": len(groups),
            "n_v_groups": len(v_g),
            "n_j_groups": len(j_g),
            "n_unique_images": len({r["image_id"] for r in recs}),
            "n_cxr_groups": sum(1 for g in groups.values() if g["cxr_candidate"]),
            "n_cxr_images": len({r["image_id"] for r in recs if r["cxr_candidate"]}),
            "conditions": dict(collections.Counter(r["condition"] for r in recs)),
            "file_sha256": sha256_file(out_dir / f"{sp}.jsonl"),
        }

    manifest = {
        "manifest_version": "poc_v1",
        "created_at_bangkok": bangkok_now_iso(),
        "seed": seed,
        "source_dataset": "BoKelvin/SLAKE",
        "source_revision": SLAKE_PINNED_REVISION,
        "context_generator_version": CONTEXT_GENERATOR_VERSION,
        "train_j_vocab": j_vocab_sorted,
        "kg_subset_path": kg_subset_path.as_posix(),
        "kg_subset_sha256": sha256_file(kg_subset_path),
        "kg_subset_entries": kg_subset["total_entries"],
        "split_dedup_audit_path": dedup_path.as_posix(),
        "exclusions_csv_path": excl_csv_path.as_posix(),
        "exclusions_count": len(exclusions),
        "files": {sp: (out_dir / f"{sp}.jsonl").as_posix() for sp in ("train", "dev", "test")},
        "splits": split_stats,
    }
    save_json(manifest_path, manifest)
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare SLAKE PoC datasets and manifests.")
    parser.add_argument("--config", required=True, help="Path to configs/data.yaml")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    manifest = prepare_dataset(cfg)
    print(json.dumps(manifest["splits"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
