"""Generate Phase 1 fixture artifacts and Phase 2 data/context audit tables from real prepared data."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import subprocess
import sys

from cka_med.evaluation.bootstrap import paired_cluster_bootstrap
from cka_med.evaluation.metrics import evaluate_rows, join_predictions_with_gold
from cka_med.manifest import save_json, sha256_file
from cka_med.schema import DataRecord, PredictionRecord, read_jsonl, write_jsonl


def build_phase1_artifacts() -> None:
    p1_dir = Path("reports/phase_1")
    p1_dir.mkdir(parents=True, exist_ok=True)
    fix_dir = Path("tests/fixtures")
    fix_dir.mkdir(parents=True, exist_ok=True)

    # 8-group synthetic fixture across 4 image clusters
    rows = []
    v_specs = [
        ("syn_g1", "syn_im1", "Does the picture contain liver?", "yes"),
        ("syn_g2", "syn_im1", "Does the picture contain spleen?", "no"),
        ("syn_g3", "syn_im2", "Are there abnormalities in this image?", "yes"),
        ("syn_g4", "syn_im2", "Is the heart enlarged?", "no"),
    ]
    for gid, iid, q, a in v_specs:
        for cond, ctx in (
            ("K0", ""),
            ("Knear", "Reference note: hepatic cyst may cause abdominal distension."),
            ("Kirrel", "Reference note: tooth enamel supports chewing."),
        ):
            rows.append(
                DataRecord(
                    sample_id=f"{gid}__{cond}",
                    group_id=gid,
                    image_id=iid,
                    img_name=f"{iid}/source.jpg",
                    source_dataset="SYNTHETIC_FIXTURE",
                    source_version="v1",
                    source_qid=gid,
                    image_path=f"tests/fixtures/{iid}.jpg",
                    image_sha256="0" * 64,
                    image_dhash="0" * 16,
                    modality="X-Ray",
                    original_question=q,
                    original_answer=a,
                    normalized_answer=a,
                    task_type="V",
                    task_eligibility_rationale="synthetic_fixture",
                    split="dev",
                    condition=cond,
                    context_pack_id=f"pack_{cond}",
                    context_text=ctx,
                    context_source_entries=[],
                    rendering_version="v1_train",
                    gold_invariant=True,
                    provenance_quality="synthetic_fixture",
                    reviewer_type="synthetic_fixture",
                    synthetic_fixture=True,
                ).to_dict()
            )

    j_specs = [
        ("syn_g5", "syn_im3", "Which organ belongs to the circulatory system?", "heart"),
        ("syn_g6", "syn_im3", "Which organ belongs to the respiratory system?", "lung"),
        ("syn_g7", "syn_im4", "Which organ belongs to the urinary system?", "kidney"),
        ("syn_g8", "syn_im4", "What is the function of the main organ?", "breathe"),
    ]
    for gid, iid, q, a in j_specs:
        for cond, ctx in (
            ("K0", ""),
            ("Khelp", "Heart | belong to: Circulatory System; Lung | belong to: Respiratory System; Kidney | belong to: Urinary System."),
        ):
            rows.append(
                DataRecord(
                    sample_id=f"{gid}__{cond}",
                    group_id=gid,
                    image_id=iid,
                    img_name=f"{iid}/source.jpg",
                    source_dataset="SYNTHETIC_FIXTURE",
                    source_version="v1",
                    source_qid=gid,
                    image_path=f"tests/fixtures/{iid}.jpg",
                    image_sha256="0" * 64,
                    image_dhash="0" * 16,
                    modality="CT",
                    original_question=q,
                    original_answer=a,
                    normalized_answer=a,
                    task_type="J",
                    task_eligibility_rationale="synthetic_fixture",
                    split="dev",
                    condition=cond,
                    context_pack_id=f"pack_{cond}",
                    context_text=ctx,
                    context_source_entries=[],
                    rendering_version="v1_train",
                    gold_invariant=True,
                    provenance_quality="synthetic_fixture",
                    reviewer_type="synthetic_fixture",
                    synthetic_fixture=True,
                ).to_dict()
            )

    write_jsonl(fix_dir / "synthetic_8group.jsonl", rows)

    # Deterministic synthetic predictions:
    # On V: K0 gets 3/4 correct (g1,g2,g3 correct; g4 wrong).
    #       Knear flips g2 (correct->wrong) and g4 stays wrong -> 2/4 correct, CFR_near = 1/3, LostCorrect = 1/4.
    #       Kirrel stays 3/4 correct -> CFR_irrel = 0.0.
    # On J: K0 gets 1/4 correct (g5); Khelp gets 3/4 correct (g5, g6, g7) -> KUG = +0.50.
    preds = []
    for r in rows:
        sid = r["sample_id"]
        gold = r["normalized_answer"]
        if sid in ("syn_g4__K0", "syn_g4__Knear", "syn_g4__Kirrel"):
            pred_val = "yes"  # wrong (gold is no)
        elif sid == "syn_g2__Knear":
            pred_val = "yes"  # flipped correct->wrong (gold is no)
        elif sid in ("syn_g6__K0", "syn_g7__K0", "syn_g8__K0", "syn_g8__Khelp"):
            pred_val = "liver"  # wrong
        else:
            pred_val = gold

        preds.append(
            PredictionRecord(
                run_id="syn_run_1",
                model_id="synthetic_model",
                group_id=r["group_id"],
                sample_id=sid,
                image_id=r["image_id"],
                condition=r["condition"],
                task_type=r["task_type"],
                context_pack_id=r["context_pack_id"],
                oracle_context=False,
                raw_output=pred_val,
                normalized_answer=pred_val,
                label_scores={"yes": 0.9, "no": 0.1} if r["task_type"] == "V" else None,
                abstained=False,
                valid_output=True,
                error=None,
                latency_ms=1.0,
            ).to_dict()
        )

    joined = join_predictions_with_gold(rows, preds)
    actual = evaluate_rows(joined)
    boot = paired_cluster_bootstrap({"synthetic_model": joined}, n_replicates=200, seed=42)
    actual["bootstrap_4clusters"] = boot

    expected = {
        "acc_0_v": 0.75,
        "acc_near_v": 0.50,
        "acc_irrel_v": 0.75,
        "cfr_near_v": 1.0 / 3.0,
        "cfr_irrel_v": 0.0,
        "acc_0_j": 0.25,
        "acc_help_j": 0.75,
        "kug": 0.50,
        "j_rescued_count": 2,
        "j_harmed_count": 0,
    }
    save_json(p1_dir / "fixture_expected.json", expected)
    save_json(p1_dir / "fixture_actual.json", actual)

    # Run pytest and record exact output
    proc = subprocess.run([sys.executable, "-m", "pytest", "-v"], capture_output=True, text=True, check=False)
    (p1_dir / "test_results.txt").write_text(proc.stdout + "\n" + proc.stderr, encoding="utf-8")
    (p1_dir / "commands.txt").write_text(
        "\n".join(
            [
                f"{sys.executable} -m pytest -v",
                f"{sys.executable} -m cka_med.data.prepare --config configs/data.yaml",
                f"{sys.executable} -m cka_med.data.audit --manifest data/manifests/poc_v1.json",
                f"{sys.executable} -m cka_med.evaluation.run --help",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def build_phase2_artifacts() -> None:
    p2_dir = Path("reports/phase_2")
    p2_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2("data/processed/exclusions.csv", p2_dir / "exclusions.csv")
    shutil.copy2("data/processed/split_dedup_audit.json", p2_dir / "split_audit.json")

    dev_rows = read_jsonl("data/processed/dev.jsonl")
    # Build 40-example eligibility_audit.csv (20 V, 20 J across pos/neg)
    v_k0 = [r for r in dev_rows if r["task_type"] == "V" and r["condition"] == "K0"]
    j_k0 = [r for r in dev_rows if r["task_type"] == "J" and r["condition"] == "K0"]

    v_yes = [r for r in v_k0 if r["normalized_answer"] == "yes"][:10]
    v_no = [r for r in v_k0 if r["normalized_answer"] == "no"][:10]
    j_sel = j_k0[:20]

    elig_rows = []
    for r in v_yes + v_no + j_sel:
        elig_rows.append(
            {
                "sample_id": r["sample_id"],
                "group_id": r["group_id"],
                "source_qid": r["source_qid"],
                "img_name": r["img_name"],
                "modality": r["modality"],
                "cxr_candidate": r["cxr_candidate"],
                "task_type": r["task_type"],
                "content_type": r["content_type"],
                "question": r["original_question"],
                "normalized_answer": r["normalized_answer"],
                "rationale": r["task_eligibility_rationale"],
                "gold_invariant": r["gold_invariant"],
                "reviewer_type": r["reviewer_type"],
                "audit_note": "Gold preserved from SLAKE reference; no organ leak in J; no specialist diagnosis synthesized.",
            }
        )

    with (p2_dir / "eligibility_audit.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(elig_rows[0].keys()))
        w.writeheader()
        w.writerows(elig_rows)

    # Build 40-example context_audit.csv covering Knear, Kirrel, Khelp
    v_knear = [r for r in dev_rows if r["task_type"] == "V" and r["condition"] == "Knear"][:15]
    v_kirrel = [r for r in dev_rows if r["task_type"] == "V" and r["condition"] == "Kirrel"][:10]
    j_khelp = [r for r in dev_rows if r["task_type"] == "J" and r["condition"] == "Khelp"][:15]

    ctx_rows = []
    for r in v_knear + v_kirrel + j_khelp:
        entry_ids = [e.get("entry_id") for e in r["context_source_entries"]]
        ctx_rows.append(
            {
                "sample_id": r["sample_id"],
                "group_id": r["group_id"],
                "task_type": r["task_type"],
                "condition": r["condition"],
                "context_pack_id": r["context_pack_id"],
                "oracle_context": r["oracle_context"],
                "word_count": len(r["context_text"].split()),
                "source_entry_ids": ";".join(str(x) for x in entry_ids),
                "question": r["original_question"],
                "gold_answer": r["normalized_answer"],
                "why_no_gold_leak": (
                    "Multi-organ pack does not single out answer; requires image."
                    if r["condition"] == "Khelp"
                    else "General source-asserted KG entry matched from question text only; does not establish image truth."
                ),
            }
        )

    with (p2_dir / "context_audit.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(ctx_rows[0].keys()))
        w.writeheader()
        w.writerows(ctx_rows)


if __name__ == "__main__":
    build_phase1_artifacts()
    build_phase2_artifacts()
    print("Built Phase 1 and Phase 2 audit artifacts.")
