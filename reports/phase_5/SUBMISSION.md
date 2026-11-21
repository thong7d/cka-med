# Phase 5 — Same-data SFT và conditional candidate

- Ngày/giờ Asia/Bangkok: 2026-11-21T10:50+07:00
- Plan version: `docs/POC_PLAN.md` v1.0
- Selected candidate: `M-C_lam05_seed42` (locked on `dev` per P4 rules)

## 1. Mục tiêu và việc thực hiện

Completed:
- Verified P5 unit checks (`tests/test_p5_objectives.py`, 5/5 passed -> `reports/phase_5/gradient_checks.txt`).
- Trained `B-S_seed42`, `M-C_lam05_seed42`, `M-C_lam20_seed42`, `A-U_seed42`, `B-S_seed43`, `M-C_lam05_seed43` on identical paired training items (`1,040` pairs = `2,080` record exposures, `130` optimizer steps, `LoRA r=8` on LLM layers only, frozen vision encoder).
- Verified identical exposure, optimizer steps, and model revision across all runs (`reports/phase_5/fairness_audit.json`).
- Evaluated all 6 trained adapters on `dev` (`315` records, 0 missing) and selected `M-C_lam05_seed42` prior to Phase 6 `test` evaluation.

## 2. Artifact index

- `reports/phase_5/{train_summary.csv,dev_results.csv,fairness_audit.json,gradient_checks.txt,selection_decision.md,dev_bootstrap.json,SUBMISSION.md}`
- `artifacts/runs/{B-S_seed42,M-C_lam05_seed42,M-C_lam20_seed42,A-U_seed42,B-S_seed43,M-C_lam05_seed43}/` (adapters + train logs)
- `artifacts/runs/*_dev/predictions.jsonl`
