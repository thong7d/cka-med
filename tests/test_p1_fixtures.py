"""P1 CPU fixture tests covering metrics, transitions, clustering, schema isolation, and CLI."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from cka_med.evaluation.bootstrap import paired_cluster_bootstrap
from cka_med.evaluation.metrics import evaluate_rows, join_predictions_with_gold
from cka_med.evaluation.run import evaluate_prediction_files
from cka_med.inference.prompts import render_prompt
from cka_med.inference.scoring import normalize_answer
from cka_med.schema import (
    DataRecord,
    DuplicatePredictionError,
    InputLeakageError,
    PredictionRecord,
    PredictionWriter,
    read_jsonl,
    write_jsonl,
)


def _make_record(
    *,
    sample_id: str,
    group_id: str,
    image_id: str,
    task_type: str,
    condition: str,
    question: str,
    answer: str,
    context_text: str = "",
    oracle_context: bool = False,
) -> dict:
    rec = DataRecord(
        sample_id=sample_id,
        group_id=group_id,
        image_id=image_id,
        img_name=f"{image_id}/source.jpg",
        source_dataset="SYNTHETIC_FIXTURE",
        source_version="v1",
        source_qid=group_id,
        image_path=f"tests/fixtures/{image_id}.jpg",
        image_sha256="0" * 64,
        image_dhash="0" * 16,
        modality="X-Ray",
        original_question=question,
        original_answer=answer,
        normalized_answer=answer.lower(),
        task_type=task_type,
        task_eligibility_rationale="synthetic fixture",
        split="dev",
        condition=condition,
        context_pack_id=f"pack_{condition}",
        context_text=context_text,
        context_source_entries=[],
        rendering_version="v1_train",
        gold_invariant=True,
        provenance_quality="synthetic_fixture",
        reviewer_type="synthetic_fixture",
        oracle_context=oracle_context,
        synthetic_fixture=True,
    )
    return rec.to_dict()


def _make_pred(
    *,
    run_id: str,
    model_id: str,
    rec: dict,
    raw_output: str,
    norm: str | None,
    abstained: bool = False,
    valid: bool = True,
) -> PredictionRecord:
    return PredictionRecord(
        run_id=run_id,
        model_id=model_id,
        group_id=rec["group_id"],
        sample_id=rec["sample_id"],
        image_id=rec["image_id"],
        condition=rec["condition"],
        task_type=rec["task_type"],
        context_pack_id=rec["context_pack_id"],
        oracle_context=rec["oracle_context"],
        raw_output=raw_output,
        normalized_answer=norm,
        label_scores={"yes": 0.6, "no": 0.4} if rec["task_type"] == "V" else None,
        abstained=abstained,
        valid_output=valid,
        error=None,
        latency_ms=12.5,
    )


def _build_fixture_dataset() -> list[dict]:
    # 8 groups across 5 images (img1 has 2 groups, img2 has 2 groups, img3..5 have 1-2 groups)
    rows = []
    v_specs = [
        ("g1", "img1", "Does the picture contain liver?", "yes"),
        ("g2", "img1", "Does the picture contain spleen?", "no"),
        ("g3", "img2", "Are there abnormalities in this image?", "yes"),
        ("g4", "img2", "Is the heart enlarged?", "no"),
    ]
    for gid, iid, q, a in v_specs:
        rows.append(_make_record(sample_id=f"{gid}_K0", group_id=gid, image_id=iid, task_type="V", condition="K0", question=q, answer=a))
        rows.append(_make_record(sample_id=f"{gid}_Knear", group_id=gid, image_id=iid, task_type="V", condition="Knear", question=q, answer=a, context_text="Note: Heart relates to blood flow."))
        rows.append(_make_record(sample_id=f"{gid}_Kirrel", group_id=gid, image_id=iid, task_type="V", condition="Kirrel", question=q, answer=a, context_text="Note: Tooth helps chew food."))

    j_specs = [
        ("g5", "img3", "Which organ belongs to the circulatory system?", "heart"),
        ("g6", "img4", "Which organ belongs to the respiratory system?", "lung"),
        ("g7", "img5", "Which organ belongs to the urinary system?", "kidney"),
        ("g8", "img5", "What is the function of the main organ?", "breathe"),
    ]
    for gid, iid, q, a in j_specs:
        rows.append(_make_record(sample_id=f"{gid}_K0", group_id=gid, image_id=iid, task_type="J", condition="K0", question=q, answer=a))
        rows.append(_make_record(sample_id=f"{gid}_Khelp", group_id=gid, image_id=iid, task_type="J", condition="Khelp", question=q, answer=a, context_text="Heart -> circulatory; Lung -> respiratory/breathe; Kidney -> urinary."))
    return rows


def test_poc_plan_exact_4_group_fixture():
    # Matches POC_PLAN.md line 220: z0=[1,1,0,0], zk=[1,0,1,0] -> CFR=1/2, LostCorrect=1/4, acc=1/2
    data = []
    for i, ans in enumerate(["yes", "yes", "no", "no"], start=1):
        gid = f"g{i}"
        data.append(_make_record(sample_id=f"{gid}_K0", group_id=gid, image_id=f"im{i}", task_type="V", condition="K0", question="Q?", answer=ans))
        data.append(_make_record(sample_id=f"{gid}_Knear", group_id=gid, image_id=f"im{i}", task_type="V", condition="Knear", question="Q?", answer=ans, context_text="K"))

    preds = [
        _make_pred(run_id="r", model_id="m", rec=data[0], raw_output="A", norm="yes").to_dict(),  # z0=1
        _make_pred(run_id="r", model_id="m", rec=data[1], raw_output="A", norm="yes").to_dict(),  # zk=1
        _make_pred(run_id="r", model_id="m", rec=data[2], raw_output="A", norm="yes").to_dict(),  # z0=1
        _make_pred(run_id="r", model_id="m", rec=data[3], raw_output="B", norm="no").to_dict(),   # zk=0
        _make_pred(run_id="r", model_id="m", rec=data[4], raw_output="A", norm="yes").to_dict(),  # z0=0
        _make_pred(run_id="r", model_id="m", rec=data[5], raw_output="B", norm="no").to_dict(),   # zk=1
        _make_pred(run_id="r", model_id="m", rec=data[6], raw_output="A", norm="yes").to_dict(),  # z0=0
        _make_pred(run_id="r", model_id="m", rec=data[7], raw_output="A", norm="yes").to_dict(),  # zk=0
    ]
    joined = join_predictions_with_gold(data, preds)
    ev = evaluate_rows(joined)
    assert ev["summary"]["acc_0_v"] == pytest.approx(0.5)
    assert ev["summary"]["acc_near_v"] == pytest.approx(0.5)
    tr = ev["transitions"]["V:K0->Knear"]
    assert tr["cfr"] == pytest.approx(0.5)
    assert tr["lost_correct_rate"] == pytest.approx(0.25)
    assert tr["harmful_wrong_flip_rate"] == pytest.approx(0.25)
    assert tr["positive_flip_rate"] == pytest.approx(0.25)


def test_all_correct_all_wrong_and_zero_cfr_denominator():
    data = [
        _make_record(sample_id="g1_K0", group_id="g1", image_id="im1", task_type="V", condition="K0", question="Q?", answer="yes"),
        _make_record(sample_id="g1_Knear", group_id="g1", image_id="im1", task_type="V", condition="Knear", question="Q?", answer="yes", context_text="K"),
    ]
    # All-correct
    preds_ok = [
        _make_pred(run_id="r", model_id="m", rec=data[0], raw_output="A", norm="yes").to_dict(),
        _make_pred(run_id="r", model_id="m", rec=data[1], raw_output="A", norm="yes").to_dict(),
    ]
    ev_ok = evaluate_rows(join_predictions_with_gold(data, preds_ok))
    assert ev_ok["summary"]["acc_0_v"] == 1.0
    assert ev_ok["summary"]["cfr_near_v"] == 0.0

    # All-wrong -> z0_correct == 0 -> CFR MUST be None (null)
    preds_bad = [
        _make_pred(run_id="r", model_id="m", rec=data[0], raw_output="B", norm="no").to_dict(),
        _make_pred(run_id="r", model_id="m", rec=data[1], raw_output="B", norm="no").to_dict(),
    ]
    ev_bad = evaluate_rows(join_predictions_with_gold(data, preds_bad))
    assert ev_bad["summary"]["acc_0_v"] == 0.0
    assert ev_bad["summary"]["cfr_near_v"] is None
    assert ev_bad["transitions"]["V:K0->Knear"]["cfr_den"] == 0


def test_semantic_unknown_vs_abstain_vs_invalid():
    norm_ab, abst_ab, val_ab = normalize_answer("Abstain", task_type="V")
    assert norm_ab is None and abst_ab is True and val_ab is True

    norm_inv, abst_inv, val_inv = normalize_answer("maybe kidney", task_type="V")
    assert norm_inv is None and abst_inv is False and val_inv is False

    # Semantic unknown in J/C stays distinct from abstain action
    norm_sem, abst_sem, val_sem = normalize_answer("insufficient_evidence", task_type="C", allowed_vocab=["insufficient_evidence", "yes", "no"])
    assert norm_sem == "insufficient_evidence" and abst_sem is False and val_sem is True


def test_missing_variant_and_incomplete_coverage_not_dropped(tmp_path: Path):
    data = _build_fixture_dataset()
    # Drop one prediction intentionally
    preds = []
    for r in data[:-1]:
        preds.append(_make_pred(run_id="r1", model_id="m1", rec=r, raw_output=r["normalized_answer"], norm=r["normalized_answer"]).to_dict())
    pred_path = tmp_path / "preds.jsonl"
    write_jsonl(pred_path, preds)

    # Without allow_incomplete, evaluate_prediction_files must raise RuntimeError
    with pytest.raises(RuntimeError, match="Incomplete prediction coverage"):
        evaluate_prediction_files(data_records=data, prediction_paths=[pred_path], allow_incomplete=False)

    res = evaluate_prediction_files(data_records=data, prediction_paths=[pred_path], n_replicates=50, allow_incomplete=True)
    assert res["evaluations"]["m1"]["summary"]["missing_predictions"] == 1
    assert res["evaluations"]["m1"]["summary"]["incomplete_coverage"] is True
    assert res["evaluations"]["m1"]["condition_metrics"]["J:Khelp"]["n_eligible"] == 4


def test_duplicate_prediction_key_rejected(tmp_path: Path):
    data = _build_fixture_dataset()[:2]
    writer = PredictionWriter(tmp_path / "p.jsonl", overwrite=True)
    p0 = _make_pred(run_id="run1", model_id="m1", rec=data[0], raw_output="A", norm="yes")
    writer.write(p0)
    with pytest.raises(DuplicatePredictionError):
        writer.write(p0)


def test_forbidden_fields_rejected_in_prompt_and_model_input():
    data = _build_fixture_dataset()
    rec = DataRecord.from_dict(data[0])
    model_inp = rec.to_model_input_dict()
    prompt, meta = render_prompt(model_inp, mode="B-R")
    assert "Does the picture contain liver?" in prompt
    assert "External medical knowledge:" not in prompt  # K0 has empty context
    assert meta["uses_image"] is True

    for forbidden_key in ("triple", "location", "gold_entity", "role", "original_answer", "premise_label"):
        bad_inp = dict(model_inp)
        bad_inp[forbidden_key] = "LEAK"
        with pytest.raises(InputLeakageError):
            render_prompt(bad_inp, mode="B-R")

    # Oracle context rejected by default
    oracle_rec = DataRecord.from_dict(
        _make_record(sample_id="o1", group_id="go", image_id="im9", task_type="J", condition="O-H", question="Q?", answer="lung", context_text="Lung", oracle_context=True)
    )
    with pytest.raises(InputLeakageError):
        oracle_rec.to_model_input_dict(allow_oracle=False)


def test_cluster_bootstrap_and_roundtrip(tmp_path: Path):
    data = _build_fixture_dataset()
    # Test JSONL & CSV roundtrip
    jsonl_p = write_jsonl(tmp_path / "fixture.jsonl", data)
    reloaded = read_jsonl(jsonl_p)
    assert len(reloaded) == len(data)
    csv_p = tmp_path / "fixture.csv"
    with csv_p.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["sample_id", "group_id", "image_id", "task_type", "condition", "normalized_answer"])
        w.writeheader()
        for r in data:
            w.writerow({k: r[k] for k in ("sample_id", "group_id", "image_id", "task_type", "condition", "normalized_answer")})
    with csv_p.open("r", newline="", encoding="utf-8") as fh:
        csv_rows = list(csv.DictReader(fh))
    assert len(csv_rows) == len(data)

    # Create two models on the same 5 image clusters
    p_base = [_make_pred(run_id="rb", model_id="B-S", rec=r, raw_output="no" if r["condition"] == "Knear" else r["normalized_answer"], norm="no" if r["condition"] == "Knear" else r["normalized_answer"]).to_dict() for r in data]
    p_cand = [_make_pred(run_id="rc", model_id="M-C", rec=r, raw_output=r["normalized_answer"], norm=r["normalized_answer"]).to_dict() for r in data]

    j_base = join_predictions_with_gold(data, p_base)
    j_cand = join_predictions_with_gold(data, p_cand)
    boot = paired_cluster_bootstrap({"B-S": j_base, "M-C": j_cand}, comparator_model="B-S", n_replicates=200, seed=42)
    assert boot["n_image_clusters"] == 5
    delta = boot["paired_deltas"]["M-C__minus__B-S"]["acc_near_v"]
    assert delta["point_delta"] > 0.0
    assert 150 <= delta["n_valid_replicates"] <= 200
