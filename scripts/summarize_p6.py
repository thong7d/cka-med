"""Summarize P6 final test evaluations, paired deltas, clustered CIs, CXR slice, stress tests, and error inventory."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from cka_med.evaluation.bootstrap import paired_cluster_bootstrap
from cka_med.evaluation.metrics import evaluate_rows, join_predictions_with_gold
from cka_med.manifest import save_json
from cka_med.schema import read_jsonl


def _build_main_row(model_id: str, scope: str, ev: dict) -> dict:
    s = ev["summary"]
    v_kn = ev["condition_metrics"].get("V:Knear", {})
    bstats = v_kn.get("binary_stats") or {}
    return {
        "model_id": model_id,
        "scope": scope,
        "n_eligible": s["n_eligible_records"],
        "n_images": s["n_unique_images"],
        "missing": s["missing_predictions"],
        "acc_0_v": s["acc_0_v"],
        "acc_irrel_v": s["acc_irrel_v"],
        "acc_near_v": s["acc_near_v"],
        "macro_f1_near_v": s["macro_f1_near_v"],
        "cfr_irrel_v": s["cfr_irrel_v"],
        "cfr_near_v": s["cfr_near_v"],
        "v_knear_tp": bstats.get("tp_yes"),
        "v_knear_fp": bstats.get("fp_yes"),
        "v_knear_tn": bstats.get("tn_no"),
        "v_knear_fn": bstats.get("fn_yes"),
        "acc_0_j": s["acc_0_j"],
        "acc_help_j": s["acc_help_j"],
        "kug": s["kug"],
        "j_rescued": s["j_rescued_count"],
        "j_harmed": s["j_harmed_count"],
    }


def main() -> None:
    p6_dir = Path("reports/phase_6")
    p6_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = p6_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    test_records = read_jsonl("data/processed/test.jsonl")
    test_cxr = [r for r in test_records if r.get("cxr_candidate")]
    test_j = [r for r in test_records if r["task_type"] == "J"]

    # Discover all *_test prediction runs
    runs_dir = Path("artifacts/runs")
    main_order = ["B-N", "B-R", "B-P_v1", "B-F", "B-S_seed42"]
    for cand in sorted(p.name[:-5] for p in runs_dir.glob("*_test") if not p.name.endswith("_test_stress")):
        if cand not in main_order and cand not in ("D-Q", "D-K"):
            main_order.append(cand)

    main_rows = []
    trans_rows = []
    joined_full = {}
    joined_cxr = {}
    eval_full = {}

    for mid in main_order:
        pred_p = runs_dir / f"{mid}_test" / "predictions.jsonl"
        if not pred_p.exists():
            continue
        preds = read_jsonl(pred_p)
        joined = join_predictions_with_gold(test_records, preds)
        ev = evaluate_rows(joined)
        joined_full[mid] = joined
        eval_full[mid] = ev
        main_rows.append(_build_main_row(mid, "full_test", ev))

        for tkey, tval in ev["transitions"].items():
            trans_rows.append({"model_id": mid, "transition": tkey, **tval})

        # CXR exploratory slice
        preds_cxr = [p for p in preds if any(r["sample_id"] == p["sample_id"] for r in test_cxr)]
        j_cxr = join_predictions_with_gold(test_cxr, preds_cxr)
        ev_cxr = evaluate_rows(j_cxr)
        joined_cxr[mid] = j_cxr
        main_rows.append(_build_main_row(mid, "cxr_slice", ev_cxr))

    # Dependency controls D-Q / D-K on test J
    for dep in ("D-Q", "D-K"):
        pred_p = runs_dir / f"{dep}_test" / "predictions.jsonl"
        if pred_p.exists():
            preds = read_jsonl(pred_p)
            j_dep = join_predictions_with_gold(test_j, preds)
            ev_dep = evaluate_rows(j_dep)
            main_rows.append(_build_main_row(dep, "J_only_test", ev_dep))

    with (p6_dir / "main_results.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(main_rows[0].keys()))
        w.writeheader()
        w.writerows(main_rows)

    with (p6_dir / "transitions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(trans_rows[0].keys()))
        w.writeheader()
        w.writerows(trans_rows)

    # Clustered bootstrap CIs paired vs B-S_seed42
    comp = "B-S_seed42" if "B-S_seed42" in joined_full else "B-R"
    boot_full = paired_cluster_bootstrap(joined_full, comparator_model=comp, n_replicates=2000, seed=20261121)
    save_json(p6_dir / "confidence_intervals.json", boot_full)

    delta_rows = []
    for pair_name, metrics_map in boot_full["paired_deltas"].items():
        for mname, stats in metrics_map.items():
            delta_rows.append(
                {
                    "comparison": pair_name,
                    "metric": mname,
                    "point_delta": stats["point_delta"],
                    "boot_mean": stats["mean"],
                    "ci_low": stats["ci_low"],
                    "ci_high": stats["ci_high"],
                    "n_valid_replicates": stats["n_valid_replicates"],
                }
            )
    if delta_rows:
        with (p6_dir / "paired_deltas.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(delta_rows[0].keys()))
            w.writeheader()
            w.writerows(delta_rows)

    # Stress test results (*_test_stress)
    stress_rows = []
    for sdir in sorted(runs_dir.glob("*_test_stress")):
        preds = read_jsonl(sdir / "predictions.jsonl")
        joined = join_predictions_with_gold(test_records, preds)
        ev = evaluate_rows(joined)
        stress_rows.append(_build_main_row(sdir.name, "stress_test", ev))
    if stress_rows:
        with (p6_dir / "stress_results.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(stress_rows[0].keys()))
            w.writeheader()
            w.writerows(stress_rows)

    # Error inventory across strata on full test (comparing B-R vs B-S_seed42 vs selected M-C)
    test_by_sid = {r["sample_id"]: r for r in test_records}
    bs_map = {r["sample_id"]: r for r in joined_full.get("B-S_seed42", [])}
    mc_name = next((m for m in main_order if m.startswith("M-C_") and m.endswith("seed42")), None)
    mc_map = {r["sample_id"]: r for r in joined_full.get(mc_name, [])} if mc_name else {}
    br_map = {r["sample_id"]: r for r in joined_full.get("B-R", [])}

    err_rows = []
    for gid in sorted({r["group_id"] for r in test_records}):
        if f"{gid}__Knear" in test_by_sid:
            sid = f"{gid}__Knear"
            rec = test_by_sid[sid]
            r_br = br_map.get(sid, {})
            r_bs = bs_map.get(sid, {})
            r_mc = mc_map.get(sid, {})
            err_rows.append(
                {
                    "group_id": gid,
                    "task_type": "V",
                    "condition": "Knear",
                    "cxr": rec["cxr_candidate"],
                    "question": rec["original_question"],
                    "gold": rec["normalized_answer"],
                    "pred_BR": r_br.get("pred_normalized"),
                    "pred_BS": r_bs.get("pred_normalized"),
                    "pred_MC": r_mc.get("pred_normalized"),
                }
            )
        elif f"{gid}__Khelp" in test_by_sid:
            sid = f"{gid}__Khelp"
            rec = test_by_sid[sid]
            r_br = br_map.get(sid, {})
            r_bs = bs_map.get(sid, {})
            r_mc = mc_map.get(sid, {})
            err_rows.append(
                {
                    "group_id": gid,
                    "task_type": "J",
                    "condition": "Khelp",
                    "cxr": rec["cxr_candidate"],
                    "question": rec["original_question"],
                    "gold": rec["normalized_answer"],
                    "pred_BR": r_br.get("pred_normalized"),
                    "pred_BS": r_bs.get("pred_normalized"),
                    "pred_MC": r_mc.get("pred_normalized"),
                }
            )
    with (p6_dir / "error_inventory.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(err_rows[0].keys()))
        w.writeheader()
        w.writerows(err_rows)

    # Generate comparison bar figure (Acc_0(V), Acc_near(V), KUG(J))
    full_rows = [r for r in main_rows if r["scope"] == "full_test"]
    if full_rows:
        labels = [r["model_id"] for r in full_rows]
        acc0 = [100.0 * (r["acc_0_v"] or 0.0) for r in full_rows]
        accn = [100.0 * (r["acc_near_v"] or 0.0) for r in full_rows]
        kug = [100.0 * (r["kug"] or 0.0) for r in full_rows]

        x = range(len(labels))
        width = 0.26
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar([i - width for i in x], acc0, width, label="Acc_0 (V, %)")
        ax.bar(x, accn, width, label="Acc_near (V, %)")
        ax.bar([i + width for i in x], kug, width, label="KUG (J, pp)")
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=20, ha="right")
        ax.set_ylabel("Percentage / Percentage Points")
        ax.set_title("CKA-MedVQA Final Test Results (SLAKE Pilot)")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.legend()
        fig.tight_layout()
        fig.savefig(fig_dir / "final_test_tradeoff.png", dpi=160)
        plt.close(fig)

    print(json.dumps([r for r in main_rows if r["scope"] != "cxr_slice"], indent=2))


if __name__ == "__main__":
    main()
