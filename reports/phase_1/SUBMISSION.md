# Phase 1 — Repo, schema, evaluator và runner skeleton

- Ngày/giờ Asia/Bangkok: 2026-11-21T08:44+07:00
- Plan version: `docs/POC_PLAN.md` v1.0
- Protocol/data/model/config versions: `cka_med` v0.1.0, `requirements-lock.txt` pinned via `uv`

## 1. Mục tiêu và việc thực hiện

Completed:
- Created editable Python package `src/cka_med/` with `data/`, `inference/`, `training/`, `evaluation/`, `schema.py`, and `manifest.py`.
- Implemented all required CLIs (`cka_med.data.prepare`, `cka_med.data.audit`, `cka_med.inference.run`, `cka_med.training.sft`, `cka_med.training.conditional`, `cka_med.evaluation.run`).
- Built strict input-isolation guards (`FORBIDDEN_MODEL_INPUT_KEYS`, `oracle_context` rejection, duplicate `(run_id, sample_id)` rejection).
- Created synthetic 8-group fixture (`tests/fixtures/synthetic_8group.jsonl`) and ran full pytest suite (`tests/test_p1_fixtures.py` + `tests/test_p5_objectives.py`, 12/12 passed).
- Created Kaggle thin runner (`notebooks/kaggle_runner.ipynb`) and Modal GPU runner (`scripts/modal_runner.py`).

## 2. Artifact index

| Artifact path | Mục đích | Cách kiểm tra |
|---|---|---|
| `pyproject.toml`, `requirements-lock.txt` | Pinned CPU dependencies | `uv pip list` |
| `src/cka_med/` | Core pipeline package | `pytest -v` |
| `tests/fixtures/synthetic_8group.jsonl` | 8-group synthetic fixture (`synthetic_fixture=true`) | `reports/phase_1/fixture_actual.json` |
| `reports/phase_1/fixture_expected.json` | Hand-calculated expected metrics | Match against `fixture_actual.json` |
| `reports/phase_1/test_results.txt` | Full pytest run log (12/12 passed) | Re-run `.venv\Scripts\python.exe -m pytest -v` |
| `notebooks/kaggle_runner.ipynb` | Kaggle thin runner calling `cka_med` | Inspect notebook cells |

## 3. Commands thực sự đã chạy

- `.venv\Scripts\python.exe -m pytest -v` -> Exit 0 (`12 passed in 13.38s`)
- `.venv\Scripts\python.exe scripts\build_p1_p2_artifacts.py` -> Exit 0

## 4. Evidence theo acceptance criteria

- All-correct / all-wrong / `CFR denominator=0 -> null` verified (`test_all_correct_all_wrong_and_zero_cfr_denominator`).
- Exact 4-group hand fixture (`z0=[1,1,0,0], zk=[1,0,1,0]`) verified (`test_poc_plan_exact_4_group_fixture`).
- Semantic unknown != abstain != invalid verified (`test_semantic_unknown_vs_abstain_vs_invalid`).
- Missing variant / incomplete coverage stays in denominator and fails strict evaluation (`test_missing_variant_and_incomplete_coverage_not_dropped`).
- Duplicate `(run_id, sample_id)` rejected (`test_duplicate_prediction_key_rejected`).
- Forbidden gold/triple/location/role/oracle keys rejected (`test_forbidden_fields_rejected_in_prompt_and_model_input`).
- Image-cluster bootstrap & CSV/JSON roundtrip verified (`test_cluster_bootstrap_and_roundtrip`).
- Answer-only mask, empty conditional mask zero, dual-branch gradient, and `lambda=0` numerical equivalence with SFT verified (`tests/test_p5_objectives.py`).
