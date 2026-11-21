# Phase 4 — Khóa data và protocol trước training/final evaluation

- Ngày/giờ Asia/Bangkok: 2026-11-21T09:16+07:00
- Plan version: `docs/POC_PLAN.md` v1.0
- Pre-registration status: `FROZEN` (`final_test_predictions_exist: false`)

## 1. Mục tiêu và việc thực hiện

- Frozen `protocol_v1.yaml` and `protocol_v1.md` specifying primary endpoints (`Acc_near(V)`, `Acc_0(V)`, `KUG(J)`), non-inferiority margins (`-3.0 pp`), paired image-cluster bootstrap (`2,000` replicates, `seed=20261121`), and comparator `B-S_seed42`.
- Locked `B-P_v1` over `B-P_v2` from dev results and locked two nonzero `lambda_cond` candidates (`0.5`, `2.0`).
- Generated `freeze_manifest.json` recording SHA-256 hashes of all `src/`, `configs/`, `data/processed/`, `data/manifests/poc_v1.json`, and confirming no final test predictions exist.

## 2. Artifact index

- `reports/phase_4/{protocol_v1.yaml,protocol_v1.md,dataset_manifest.json,freeze_manifest.json,run_matrix.csv,selection_rules.md,SUBMISSION.md}`
