# Phase 2 — Development data và knowledge packs

- Ngày/giờ Asia/Bangkok: 2026-11-21T08:44+07:00
- Plan version: `docs/POC_PLAN.md` v1.0
- Manifest: `data/manifests/poc_v1.json` (`status: PASS`, 2,355 records checked, 0 errors)

## 1. Mục tiêu và việc thực hiện

Completed:
- Downloaded and extracted `BoKelvin/SLAKE` `imgs.zip` (pinned revision `a9083ce6c34ac3ffb17671a605962924d8a8f9e9`, 642 image folders).
- Performed SHA-256 and 64-bit dHash pixel deduplication across `train`/`validation`/`test`, excluding near-duplicate images from `train`/`validation` to protect `test` independence.
- Curated 69 provenance-backed KG entries (`data/processed/kg_subset.json`) and generated deterministic `K0`/`Khelp`/`Kirrel`/`Knear` controlled context packs (`cka_ctx_v1`).
- Built `train.jsonl` (520 groups / 1,560 records), `dev.jsonl` (105 groups / 315 records), `test.jsonl` (160 groups / 480 records).
- Audited 40 eligibility records (`eligibility_audit.csv`) and 40 context packs (`context_audit.csv`), confirming gold invariance, no organ leaks in J, and no `mask.png` inputs.
- Verified `python -m cka_med.data.audit --manifest data/manifests/poc_v1.json` -> `PASS`.

## 2. Artifact index

- `data/manifests/poc_v1.json`
- `data/processed/{train,dev,test}.jsonl`, `data/processed/kg_subset.json`
- `reports/phase_2/{data_card.md,eligibility_audit.csv,context_audit.csv,split_audit.json,exclusions.csv}`

## 3. Commands thực sự đã chạy

- `.venv\Scripts\python.exe -m cka_med.data.prepare --config configs/data.yaml` -> Exit 0
- `.venv\Scripts\python.exe -m cka_med.data.audit --manifest data/manifests/poc_v1.json` -> Exit 0 (`PASS`, 2355 records, 0 errors)
- `.venv\Scripts\python.exe scripts\build_p1_p2_artifacts.py` -> Exit 0
