# P0 — Resource access and provenance (updated 2026-11-21)

## SLAKE

- Dataset API revision: `a9083ce6c34ac3ffb17671a605962924d8a8f9e9`
- License: CC BY 4.0, according to the Hugging Face dataset card.
- Downloaded & extracted locally on 2026-11-21 (`data/raw/slake/imgs/`): 642 image folders verified.
- Metadata hash manifest: `artifacts/source_audit/slake_metadata/audit_summary.json`

## Qwen2.5-VL-3B-Instruct

- Model revision: `66285546d2b821cf421d4f5eb2576359d3770cd3`
- Public model API & `config.json`: HTTP 200.
- License: Qwen Research License Agreement (non-commercial research/evaluation grant; not Apache 2.0).

## MedGemma 4B IT

- Status: Gated (HTTP 401 unauthenticated); kept out of critical path per `POC_PLAN.md`.

## Modal (Primary Live Free GPU Runner)

- Profile verified via CLI on 2026-11-21: `thongpham796`.
- `modal billing summary`: Metered Cost `$20.58` (`$19.91` ephemeral apps, `$0.67` volumes), Credits `-$19.91`, Free Storage `-$0.67`, **Billed Cost `$0.00`**.
- Remaining Starter monthly free credit: ~$10.09 USD. L4 GPU rate `$0.80/hr` (~12.6 L4 hours available within free credit).
- Paid spend authorized: `$0.00`.

## Kaggle (Backup / Thin Notebook Target)

- CLI verified via `kaggle config view`: username `phamtohng`, `ACCESS_TOKEN` configured.
- Snapshot dated 2026-11-16 had 27.28 GPU-hours remaining with weekly refresh `2026-11-21 07:00 Asia/Bangkok`.
- Thin runner notebook provided at `notebooks/kaggle_runner.ipynb`.
