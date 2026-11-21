"""Summarize P3 dev baseline predictions, dependency controls, profiling, and compute forecast."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from cka_med.evaluation.bootstrap import paired_cluster_bootstrap
from cka_med.evaluation.metrics import evaluate_rows, join_predictions_with_gold
from cka_med.manifest import save_json
from cka_med.schema import read_jsonl


def main() -> None:
    p3_dir = Path("reports/phase_3")
    p3_dir.mkdir(parents=True, exist_ok=True)

    dev_records = read_jsonl("data/processed/dev.jsonl")
    dev_j_only = [r for r in dev_records if r["task_type"] == "J"]

    full_models = ["B-N", "B-R", "B-P_v1", "B-P_v2", "B-F"]
    dep_models = ["D-Q", "D-K"]

    summary_rows = []
    transition_rows = []
    joined_full = {}
    eval_full = {}

    for m in full_models:
        pred_path = Path(f"artifacts/runs/{m}_dev/predictions.jsonl")
        preds = read_jsonl(pred_path)
        joined = join_predictions_with_gold(dev_records, preds)
        ev = evaluate_rows(joined)
        joined_full[m] = joined
        eval_full[m] = ev
        s = ev["summary"]
        v_k0 = ev["condition_metrics"].get("V:K0", {})
        v_kn = ev["condition_metrics"].get("V:Knear", {})
        v_ki = ev["condition_metrics"].get("V:Kirrel", {})
        summary_rows.append(
            {
                "model_id": m,
                "scope": "full_dev",
                "n_eligible": s["n_eligible_records"],
                "missing": s["missing_predictions"],
                "acc_0_v": s["acc_0_v"],
                "acc_irrel_v": s["acc_irrel_v"],
                "acc_near_v": s["acc_near_v"],
                "macro_f1_near_v": s["macro_f1_near_v"],
                "cfr_irrel_v": s["cfr_irrel_v"],
                "cfr_near_v": s["cfr_near_v"],
                "v_knear_tp": (v_kn.get("binary_stats") or {}).get("tp_yes"),
                "v_knear_fp": (v_kn.get("binary_stats") or {}).get("fp_yes"),
                "v_knear_tn": (v_kn.get("binary_stats") or {}).get("tn_no"),
                "v_knear_fn": (v_kn.get("binary_stats") or {}).get("fn_yes"),
                "acc_0_j": s["acc_0_j"],
                "acc_help_j": s["acc_help_j"],
                "kug": s["kug"],
                "j_rescued": s["j_rescued_count"],
                "j_harmed": s["j_harmed_count"],
            }
        )
        for tkey, tval in ev["transitions"].items():
            transition_rows.append({"model_id": m, "transition": tkey, **tval})

    # Dependency controls on J
    dep_evals = {}
    for m in dep_models:
        preds = read_jsonl(f"artifacts/runs/{m}_dev/predictions.jsonl")
        joined = join_predictions_with_gold(dev_j_only, preds)
        ev = evaluate_rows(joined)
        dep_evals[m] = ev
        s = ev["summary"]
        summary_rows.append(
            {
                "model_id": m,
                "scope": "J_only_dev",
                "n_eligible": s["n_eligible_records"],
                "missing": s["missing_predictions"],
                "acc_0_v": None,
                "acc_irrel_v": None,
                "acc_near_v": None,
                "macro_f1_near_v": None,
                "cfr_irrel_v": None,
                "cfr_near_v": None,
                "v_knear_tp": None,
                "v_knear_fp": None,
                "v_knear_tn": None,
                "v_knear_fn": None,
                "acc_0_j": s["acc_0_j"],
                "acc_help_j": s["acc_help_j"],
                "kug": s["kug"],
                "j_rescued": s["j_rescued_count"],
                "j_harmed": s["j_harmed_count"],
            }
        )

    with (p3_dir / "baseline_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    with (p3_dir / "transitions.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(transition_rows[0].keys()))
        w.writeheader()
        w.writerows(transition_rows)

    boot = paired_cluster_bootstrap(joined_full, comparator_model="B-R", n_replicates=2000, seed=20261121)
    save_json(p3_dir / "dev_bootstrap.json", boot)

    # Profile and cost forecast
    prof_src = Path("artifacts/runs/p3_profile.json")
    if prof_src.exists():
        prof = json.loads(prof_src.read_text(encoding="utf-8"))
        save_json(p3_dir / "profile.json", prof)
        inf_eps = float(prof.get("inference_examples_per_s", 2.0))
        tr_10_s = float(prof["train_10_opt_steps_profile"].get("wall_time_s", 60.0))
        # 1 epoch on train: 520 groups = 1040 pairs, grad_accum=8 -> 130 opt steps
        est_1epoch_train_s = tr_10_s * (130.0 / 10.0) * 1.3
        est_eval_s = (480.0 / max(0.5, inf_eps)) * 1.3
        n_train_models = 5  # B-S_s42, M-C_lam05_s42, M-C_lam02_s42, A-U_s42, + optional s43 pair
        n_eval_runs = 12
        total_est_hr = (n_train_models * est_1epoch_train_s + n_eval_runs * est_eval_s) / 3600.0
        forecast = {
            "gpu_type": prof.get("gpu_name", "L4"),
            "modal_l4_rate_usd_per_hr": 0.80,
            "safety_factor": 1.3,
            "est_single_1epoch_train_s": est_1epoch_train_s,
            "est_single_test_eval_s": est_eval_s,
            "planned_train_runs": n_train_models,
            "planned_eval_runs": n_eval_runs,
            "total_estimated_gpu_hours": total_est_hr,
            "estimated_modal_cost_usd": total_est_hr * 0.80,
            "remaining_free_credit_usd": 10.09,
            "fits_within_free_credit": (total_est_hr * 0.80) <= 6.5,
        }
        save_json(p3_dir / "cost_forecast.json", forecast)

    # Casebook: sample V K0->Knear flips and J K0->Khelp rescued/harmed under B-R
    br_by_sid = {r["sample_id"]: r for r in joined_full["B-R"]}
    dev_by_sid = {r["sample_id"]: r for r in dev_records}
    case_rows = []
    for gid in sorted({r["group_id"] for r in dev_records}):
        k0_sid = f"{gid}__K0"
        r0 = br_by_sid.get(k0_sid)
        if not r0:
            continue
        if r0["task_type"] == "V":
            rn = br_by_sid.get(f"{gid}__Knear")
            if not rn:
                continue
            z0 = int(r0["pred_normalized"] == r0["gold_normalized"] and r0["valid_output"])
            zn = int(rn["pred_normalized"] == rn["gold_normalized"] and rn["valid_output"])
            strata = f"V_z0={z0}_zn={zn}"
            raw_rec = dev_by_sid[f"{gid}__Knear"]
            case_rows.append(
                {
                    "group_id": gid,
                    "strata": strata,
                    "question": raw_rec["original_question"],
                    "gold": r0["gold_normalized"],
                    "pred_k0": r0["pred_normalized"],
                    "pred_kk": rn["pred_normalized"],
                    "context_snippet": raw_rec["context_text"][:140].replace("\n", " "),
                }
            )
        else:
            rh = br_by_sid.get(f"{gid}__Khelp")
            if not rh:
                continue
            z0 = int(r0["pred_normalized"] == r0["gold_normalized"] and r0["valid_output"])
            zh = int(rh["pred_normalized"] == rh["gold_normalized"] and rh["valid_output"])
            strata = f"J_z0={z0}_zh={zh}"
            raw_rec = dev_by_sid[f"{gid}__Khelp"]
            case_rows.append(
                {
                    "group_id": gid,
                    "strata": strata,
                    "question": raw_rec["original_question"],
                    "gold": r0["gold_normalized"],
                    "pred_k0": r0["pred_normalized"],
                    "pred_kk": rh["pred_normalized"],
                    "context_snippet": raw_rec["context_text"][:140].replace("\n", " "),
                }
            )

    with (p3_dir / "casebook.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(case_rows[0].keys()))
        w.writeheader()
        w.writerows(case_rows)

    print(json.dumps(summary_rows, indent=2))


if __name__ == "__main__":
    main()
