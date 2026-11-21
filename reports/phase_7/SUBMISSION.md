# Phase 7 — Tái lập, báo cáo và quyết định khóa luận

- Ngày/giờ Asia/Bangkok: 2026-11-21T11:48+07:00
- Plan version: `docs/POC_PLAN.md` v1.0
- Final PoC Decision: **`GO — benchmark/analysis`** (`M-C > B-S` method claim **REFUTED**)

## 1. Mục tiêu và việc thực hiện

Completed:
- Re-ran CPU test suite (`12 passed in 6.68s`), manifest audit (`PASS`, `2,355` records), and raw-JSONL metric recomputation (`exact match = true` across all models).
- Verified P4 data and config SHA-256 hashes unchanged; documented post-P4 `A-U` J-branch edit in `src/cka_med/training/trainer.py`.
- Reloaded `M-C_lam05_seed42` LoRA adapter on Modal L4 GPU and verified `15/15` exact prediction matches (`reports/phase_7/gpu_reload_repro.json`).
- Generated `bundle_checksums.json`, `POC_REPORT.md`, `REPRODUCE.md`, `CLAIM_EVIDENCE.csv`, `FINAL_DECISION.md`, and `limitations.md`.

## 2. Artifact index

- `reports/phase_7/{SUBMISSION.md,POC_REPORT.md,REPRODUCE.md,CLAIM_EVIDENCE.csv,FINAL_DECISION.md,limitations.md,bundle_checksums.json,repro_verification.json,gpu_reload_repro.json}`
