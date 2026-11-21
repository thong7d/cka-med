# Phase 6 — Final evaluation và ablation

- Ngày/giờ Asia/Bangkok: 2026-11-21T11:45+07:00
- Plan version: `docs/POC_PLAN.md` v1.0
- Engineering status: Complete (`0` missing predictions across all runs)
- Scientific verdict: **`REFUTE` for `M-C > B-S` method superiority** (`delta Acc_near = 0.0000` on seed 42 and seed 43; robustness gain is already captured by same-data SFT `B-S`).

## 1. Mục tiêu và việc thực hiện

Completed:
- Evaluated all mandatory baselines (`D-Q`, `D-K`, `B-N`, `B-R`, `B-P_v1`, `B-F`), comparator (`B-S_seed42`, `B-S_seed43`), candidate (`M-C_lam05_seed42`, `M-C_lam05_seed43`), and unconditional ablation (`A-U_seed42`) on frozen `test.jsonl` (`160` groups / `480` records / `63` unique images).
- Evaluated held-out rendering template + reversed pack order stress tests (`B-S_seed42_test_stress`, `M-C_lam05_seed42_test_stress`, `B-S_seed43_test_stress`).
- Computed exploratory CXR slice (`112` groups / `336` records across `30` CXR images).
- Computed 2,000-replicate paired image-cluster bootstrap CIs (`seed=20261121`) against `B-S_seed42`.
- Generated error inventory (`error_inventory.csv`) and summary figure (`figures/final_test_tradeoff.png`).

## 2. Artifact index

- `reports/phase_6/{main_results.csv,paired_deltas.csv,confidence_intervals.json,transitions.csv,stress_results.csv,error_inventory.csv,compute_actual.json,final_access_log.md,figures/final_test_tradeoff.png,SUBMISSION.md}`
- `artifacts/runs/*_test/predictions.jsonl` and `artifacts/runs/*_test_stress/predictions.jsonl`
