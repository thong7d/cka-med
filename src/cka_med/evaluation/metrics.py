"""Evaluation metrics, paired transitions, and missingness accounting."""

from __future__ import annotations

import collections
from typing import Any


def _safe_div(num: int | float, den: int | float) -> float | None:
    if den == 0:
        return None
    return float(num) / float(den)


def compute_binary_macro_f1(y_true: list[str], y_pred: list[str | None]) -> dict[str, Any]:
    """Compute yes/no macro-F1 and confusion counts; invalid/missing predictions count as errors."""
    tp = fp = tn = fn = invalid_or_missing = 0
    for yt, yp in zip(y_true, y_pred):
        if yp not in ("yes", "no"):
            invalid_or_missing += 1
            if yt == "yes":
                fn += 1
            else:
                fp += 1
            continue
        if yt == "yes" and yp == "yes":
            tp += 1
        elif yt == "no" and yp == "yes":
            fp += 1
        elif yt == "no" and yp == "no":
            tn += 1
        elif yt == "yes" and yp == "no":
            fn += 1

    f1_yes = _safe_div(2 * tp, 2 * tp + fp + fn)
    f1_no = _safe_div(2 * tn, 2 * tn + fn + fp)
    if f1_yes is not None and f1_no is not None:
        macro_f1: float | None = 0.5 * (f1_yes + f1_no)
    elif f1_yes is not None:
        macro_f1 = f1_yes
    elif f1_no is not None:
        macro_f1 = f1_no
    else:
        macro_f1 = None

    return {
        "macro_f1": macro_f1,
        "f1_yes": f1_yes,
        "f1_no": f1_no,
        "tp_yes": tp,
        "fp_yes": fp,
        "tn_no": tn,
        "fn_yes": fn,
        "invalid_or_missing_in_confusion": invalid_or_missing,
    }


def evaluate_rows(
    joined_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Evaluate joined (reference + prediction) rows.

    Each row must already contain:
      sample_id, group_id, image_id, task_type, condition,
      gold_normalized, pred_normalized, valid_output, abstained, missing_prediction
    """
    by_task_cond: dict[tuple[str, str], list[dict[str, Any]]] = collections.defaultdict(list)
    for row in joined_rows:
        by_task_cond[(row["task_type"], row["condition"])].append(row)

    condition_metrics: dict[str, dict[str, Any]] = {}
    for (task_type, condition), rows in sorted(by_task_cond.items()):
        n = len(rows)
        correct = 0
        missing = 0
        invalid = 0
        abstained = 0
        y_true: list[str] = []
        y_pred: list[str | None] = []
        for r in rows:
            is_missing = bool(r.get("missing_prediction", False))
            is_abstained = bool(r.get("abstained", False))
            is_valid = bool(r.get("valid_output", False))
            gold = r["gold_normalized"]
            pred = r.get("pred_normalized")
            if is_missing:
                missing += 1
            elif is_abstained:
                abstained += 1
            elif not is_valid:
                invalid += 1
            is_correct = (not is_missing) and (not is_abstained) and is_valid and (pred == gold)
            if is_correct:
                correct += 1
            y_true.append(gold)
            y_pred.append(pred if (not is_missing and not is_abstained and is_valid) else None)

        entry: dict[str, Any] = {
            "task_type": task_type,
            "condition": condition,
            "n_eligible": n,
            "n_images": len({r["image_id"] for r in rows}),
            "correct": correct,
            "accuracy": _safe_div(correct, n),
            "missing": missing,
            "invalid": invalid,
            "abstained": abstained,
        }
        if task_type == "V":
            entry["binary_stats"] = compute_binary_macro_f1(y_true, y_pred)
        condition_metrics[f"{task_type}:{condition}"] = entry

    # Paired transitions relative to K0 within each (task_type, group_id)
    grouped: dict[tuple[str, str], dict[str, dict[str, Any]]] = collections.defaultdict(dict)
    for r in joined_rows:
        grouped[(r["task_type"], r["group_id"])][r["condition"]] = r

    transitions: dict[str, dict[str, Any]] = {}
    all_pairs = sorted({(r["task_type"], r["condition"]) for r in joined_rows if r["condition"] != "K0"})
    for task_type, cond in all_pairs:
        n_paired = 0
        n_images_set: set[str] = set()
        z0_correct = 0
        zk_correct = 0
        cfr_num = 0  # z0=1 and zk=0
        harmful_wrong_num = 0  # z0=1 and wrong substantive answer at k
        pos_flip_num = 0  # z0=0 and zk=1
        both_correct = 0
        both_wrong = 0
        for (t, gid), cond_map in grouped.items():
            if t != task_type or "K0" not in cond_map or cond not in cond_map:
                continue
            r0 = cond_map["K0"]
            rk = cond_map[cond]
            n_paired += 1
            n_images_set.add(r0["image_id"])
            z0 = (not r0.get("missing_prediction", False)) and (not r0.get("abstained", False)) and bool(r0.get("valid_output", False)) and (r0.get("pred_normalized") == r0["gold_normalized"])
            zk = (not rk.get("missing_prediction", False)) and (not rk.get("abstained", False)) and bool(rk.get("valid_output", False)) and (rk.get("pred_normalized") == rk["gold_normalized"])
            substantive_wrong_k = (not rk.get("missing_prediction", False)) and (not rk.get("abstained", False)) and bool(rk.get("valid_output", False)) and (not zk)

            if z0:
                z0_correct += 1
            if zk:
                zk_correct += 1
            if z0 and zk:
                both_correct += 1
            elif z0 and not zk:
                cfr_num += 1
                if substantive_wrong_k:
                    harmful_wrong_num += 1
            elif (not z0) and zk:
                pos_flip_num += 1
            else:
                both_wrong += 1

        transitions[f"{task_type}:K0->{cond}"] = {
            "task_type": task_type,
            "condition": cond,
            "n_paired": n_paired,
            "n_images": len(n_images_set),
            "z0_correct": z0_correct,
            "zk_correct": zk_correct,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "cfr_num": cfr_num,
            "cfr_den": z0_correct,
            "cfr": _safe_div(cfr_num, z0_correct),
            "lost_correct_rate": _safe_div(cfr_num, n_paired),
            "harmful_wrong_flip_num": harmful_wrong_num,
            "harmful_wrong_flip_rate": _safe_div(harmful_wrong_num, n_paired),
            "positive_flip_num": pos_flip_num,
            "positive_flip_rate": _safe_div(pos_flip_num, n_paired),
            "acc_0": _safe_div(z0_correct, n_paired),
            "acc_k": _safe_div(zk_correct, n_paired),
            "delta_acc_k_minus_0": (
                (_safe_div(zk_correct, n_paired) - _safe_div(z0_correct, n_paired))
                if n_paired > 0
                else None
            ),
        }

    # Primary PoC endpoint shortcuts
    v_k0 = condition_metrics.get("V:K0", {})
    v_knear = condition_metrics.get("V:Knear", {})
    v_kirrel = condition_metrics.get("V:Kirrel", {})
    j_k0 = condition_metrics.get("J:K0", {})
    j_khelp = condition_metrics.get("J:Khelp", {})

    kug = None
    if j_k0.get("accuracy") is not None and j_khelp.get("accuracy") is not None:
        kug = j_khelp["accuracy"] - j_k0["accuracy"]

    j_help_trans = transitions.get("J:K0->Khelp", {})

    total_missing = sum(1 for r in joined_rows if r.get("missing_prediction", False))
    summary = {
        "n_eligible_records": len(joined_rows),
        "n_unique_images": len({r["image_id"] for r in joined_rows}),
        "missing_predictions": total_missing,
        "incomplete_coverage": total_missing > 0,
        "acc_0_v": v_k0.get("accuracy"),
        "acc_near_v": v_knear.get("accuracy"),
        "acc_irrel_v": v_kirrel.get("accuracy"),
        "macro_f1_near_v": (v_knear.get("binary_stats") or {}).get("macro_f1"),
        "cfr_near_v": transitions.get("V:K0->Knear", {}).get("cfr"),
        "cfr_irrel_v": transitions.get("V:K0->Kirrel", {}).get("cfr"),
        "acc_0_j": j_k0.get("accuracy"),
        "acc_help_j": j_khelp.get("accuracy"),
        "kug": kug,
        "kug_paired": j_help_trans.get("delta_acc_k_minus_0"),
        "j_rescued_count": j_help_trans.get("positive_flip_num"),
        "j_harmed_count": j_help_trans.get("cfr_num"),
    }

    return {
        "summary": summary,
        "condition_metrics": condition_metrics,
        "transitions": transitions,
    }


def join_predictions_with_gold(
    data_records: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join predictions onto eligible reference records by sample_id without dropping missing predictions."""
    pred_by_sample: dict[str, dict[str, Any]] = {}
    for p in predictions:
        sid = str(p["sample_id"])
        if sid in pred_by_sample:
            raise ValueError(f"Duplicate sample_id in prediction file: {sid}")
        pred_by_sample[sid] = p

    joined: list[dict[str, Any]] = []
    for rec in data_records:
        if rec.get("exclusion_reason") is not None:
            continue
        sid = str(rec["sample_id"])
        pred = pred_by_sample.get(sid)
        if pred is None:
            joined.append(
                {
                    "sample_id": sid,
                    "group_id": rec["group_id"],
                    "image_id": rec["image_id"],
                    "task_type": rec["task_type"],
                    "condition": rec["condition"],
                    "cxr_candidate": bool(rec.get("cxr_candidate", False)),
                    "gold_normalized": rec["normalized_answer"],
                    "pred_normalized": None,
                    "valid_output": False,
                    "abstained": False,
                    "missing_prediction": True,
                }
            )
        else:
            joined.append(
                {
                    "sample_id": sid,
                    "group_id": rec["group_id"],
                    "image_id": rec["image_id"],
                    "task_type": rec["task_type"],
                    "condition": rec["condition"],
                    "cxr_candidate": bool(rec.get("cxr_candidate", False)),
                    "gold_normalized": rec["normalized_answer"],
                    "pred_normalized": pred.get("normalized_answer"),
                    "valid_output": bool(pred.get("valid_output", False)),
                    "abstained": bool(pred.get("abstained", False)),
                    "missing_prediction": False,
                }
            )
    return joined
