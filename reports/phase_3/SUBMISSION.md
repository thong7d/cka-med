# Phase 3 — Competence, baseline, dependency và compute gate

- Ngày/giờ Asia/Bangkok: 2026-11-21T09:15+07:00
- Plan version: `docs/POC_PLAN.md` v1.0
- Gate verdict: **`TRAIN`**

## 1. Mục tiêu và việc thực hiện

Completed:
- Profiled `Qwen/Qwen2.5-VL-3B-Instruct` (`66285546d2b821cf421d4f5eb2576359d3770cd3`) in `bfloat16` on Modal NVIDIA L4 (`8.13 GB` inference peak VRAM, `1.96 ex/s`; `8.35 GB` training peak VRAM, `32.1 s` per 10 opt steps).
- Caught and fixed a LoRA target-module leak during P3 smoke check where `gate_proj/up_proj/down_proj` matched `model.visual.blocks.*.mlp`; restricted LoRA strictly to language-model layers (`visual` has 0 trainable parameters).
- Ran all mandatory dev baselines (`D-Q`, `D-K`, `B-N`, `B-R`, `B-P_v1`, `B-P_v2`, `B-F`) with 0 missing/invalid predictions (`artifacts/runs/*_dev/predictions.jsonl`).
- Computed paired transitions, `dev_bootstrap.json` (2,000 image-cluster replicates), `casebook.csv`, and `cost_forecast.json`.

## 2. Artifact index

- `reports/phase_3/{baseline_summary.csv,transitions.csv,dependency_report.md,profile.json,cost_forecast.json,train_gate.md,dev_bootstrap.json,casebook.csv}`
- `artifacts/runs/{D-Q,D-K,B-N,B-R,B-P_v1,B-P_v2,B-F}_dev/`
