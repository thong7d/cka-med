"""Summarize P5 training runs, fairness audit, and dev checkpoint selection."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from cka_med.evaluation.bootstrap import paired_cluster_bootstrap
from cka_med.evaluation.metrics import evaluate_rows, join_predictions_with_gold
from cka_med.manifest import save_json
from cka_med.schema import read_jsonl


def main() -> None:
    p5_dir = Path("reports/phase_5")
    p5_dir.mkdir(parents=True, exist_ok=True)
    dev_records = read_jsonl("data/processed/dev.jsonl")

    run_ids = []
    for cand in ["B-S_seed42", "M-C_lam05_seed42", "M-C_lam20_seed42", "A-U_seed42", "B-S_seed43", "M-C_lam05_seed43"]:
        if Path(f"artifacts/runs/{cand}_dev/predictions.jsonl").exists():
            run_ids.append(cand)

    train_rows = []
    dev_rows = []
    joined_by_model = {}
    fairness = {}

    for rid in run_ids:
        man_path = Path(f"artifacts/runs/{rid}/run_manifest.json")
        if man_path.exists():
            m = json.loads(man_path.read_text(encoding="utf-8"))
            train_rows.append(
                {
                    "run_id": rid,
                    "objective_mode": m.get("objective_mode"),
                    "lambda_cond": m.get("lambda_cond"),
                    "seed": m.get("seed"),
                    "opt_steps": m.get("opt_steps"),
                    "exposure_records": m.get("exposure_records"),
                    "n_pairs": m.get("n_pairs"),
                    "wall_time_s": m.get("wall_time_s"),
                    "peak_vram_mb": m.get("peak_vram_mb"),
                    "final_ans_loss": m.get("final_ans_loss"),
                    "final_cond_loss": m.get("final_cond_loss"),
                }
            )
            fairness[rid] = {
                "seed": m.get("seed"),
                "opt_steps": m.get("opt_steps"),
                "exposure_records": m.get("exposure_records"),
                "wall_time_s": m.get("wall_time_s"),
                "lora_rank": m.get("config", {}).get("lora_rank"),
                "base_model_revision": m.get("model_revision"),
            }

        preds = read_jsonl(f"artifacts/runs/{rid}_dev/predictions.jsonl")
        joined = join_predictions_with_gold(dev_records, preds)
        joined_by_model[rid] = joined
        ev = evaluate_rows(joined)
        s = ev["summary"]
        dev_rows.append(
            {
                "model_id": rid,
                "n_eligible": s["n_eligible_records"],
                "missing": s["missing_predictions"],
                "acc_0_v": s["acc_0_v"],
                "acc_irrel_v": s["acc_irrel_v"],
                "acc_near_v": s["acc_near_v"],
                "macro_f1_near_v": s["macro_f1_near_v"],
                "cfr_irrel_v": s["cfr_irrel_v"],
                "cfr_near_v": s["cfr_near_v"],
                "acc_0_j": s["acc_0_j"],
                "acc_help_j": s["acc_help_j"],
                "kug": s["kug"],
                "j_rescued": s["j_rescued_count"],
                "j_harmed": s["j_harmed_count"],
            }
        )

    if train_rows:
        with (p5_dir / "train_summary.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(train_rows[0].keys()))
            w.writeheader()
            w.writerows(train_rows)

    if dev_rows:
        with (p5_dir / "dev_results.csv").open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(dev_rows[0].keys()))
            w.writeheader()
            w.writerows(dev_rows)

    save_json(p5_dir / "fairness_audit.json", fairness)
    if "B-S_seed42" in joined_by_model:
        boot = paired_cluster_bootstrap(joined_by_model, comparator_model="B-S_seed42", n_replicates=2000, seed=20261121)
        save_json(p5_dir / "dev_bootstrap.json", boot)

    print(json.dumps(dev_rows, indent=2))


if __name__ == "__main__":
    main()
