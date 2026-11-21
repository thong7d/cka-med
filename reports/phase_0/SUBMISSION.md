# Phase 0 — Scope, sources and access (updated 2026-11-21)

- Date/time Asia/Bangkok: 2026-11-21T08:35+07:00
- Plan version: `docs/POC_PLAN.md` 1.0
- Protocol/data/model/config versions: P0 updated with live Modal/Kaggle probes & local SLAKE images

## 1. Objective and work performed

Completed:
- Confirmed scope: SLAKE English + bundled KG, `Qwen/Qwen2.5-VL-3B-Instruct` (`66285546d2b821cf421d4f5eb2576359d3770cd3`) default backbone, frozen vision encoder, CXR reported as exploratory slice only.
- Verified SLAKE (`a9083ce6c34ac3ffb17671a605962924d8a8f9e9`, CC BY 4.0) & downloaded `imgs.zip` (642 image folders).
- Verified live Modal profile `thongpham796` and billing summary (`$20.58` metered covered by credits, `$0.00` billed, ~$10.09 free Starter credit remaining, L4 at `$0.80/hr`).
- Verified Kaggle CLI config (`phamtohng`, `ACCESS_TOKEN`); weekly 30h refresh window (`2026-11-21 07:00 Asia/Bangkok`) passed.
- Maintained `prior_art_matrix.csv`, `scope_contract.md`, `decision_log.md`, `protocol_draft.md`.

## 2. Artifact index

- `reports/phase_0/{SUBMISSION.md,scope_contract.md,decision_log.md,resource_access.md,compute_budget.json,protocol_draft.md,prior_art_matrix.csv,source_manifest.json,probe_log.txt}`

## 3. Observed status

- Paid spend authorized: `$0.00` (enforced).
- Training gate: ready for P1–P3 profiling before full training.
