# Reproducibility Guide — CKA-MedVQA PoC

## 1. Environment & Pinned Revisions

- Python 3.12 (`uv` venv) with pinned dependencies in `requirements-lock.txt`
- Dataset: `BoKelvin/SLAKE` revision `a9083ce6c34ac3ffb17671a605962924d8a8f9e9` (CC BY 4.0)
- Model: `Qwen/Qwen2.5-VL-3B-Instruct` revision `66285546d2b821cf421d4f5eb2576359d3770cd3` (Qwen Research License)
- Checksums index: `reports/phase_7/bundle_checksums.json`

## 2. Clean CPU Verification Commands

```powershell
.venv\Scripts\python.exe -m pytest -v
.venv\Scripts\python.exe -m cka_med.data.audit --manifest data/manifests/poc_v1.json
.venv\Scripts\python.exe scripts\verify_p7_repro.py
.venv\Scripts\python.exe scripts\summarize_p6.py
```

Single-model evaluation from raw predictions:

```powershell
.venv\Scripts\python.exe -m cka_med.evaluation.run `
  --predictions artifacts/runs/B-S_seed42_test/predictions.jsonl artifacts/runs/M-C_lam05_seed42_test/predictions.jsonl `
  --protocol reports/phase_4/protocol_v1.yaml `
  --comparator B-S_seed42
```

## 3. GPU Training & Inference Commands (Modal L4 / Kaggle Thin Runner)

- Modal L4 runner: `scripts/modal_runner.py` (`--action p3`, `--action p5`, `--action p6`, `--action p7`)
- Kaggle thin runner: `notebooks/kaggle_runner.ipynb` calling `cka_med` CLIs (`cka_med.inference.run`, `cka_med.training.sft`, `cka_med.training.conditional`)

## 4. Known Nondeterminism & Post-Freeze Diff

- `bfloat16` GPU kernels and `gradient_checkpointing` introduce minor run-to-run floating point variance across seeds (`42` vs `43` documented separately).
- Post-P4-freeze source change: `src/cka_med/training/trainer.py` was updated before launching `A-U_seed42` so `objective_mode="unconditional"` applies JS divergence on `J` first-answer logits (otherwise `A-U` would have been identical to `M-C`); `B-S_seed42`, `M-C_lam05_seed42`, and `M-C_lam20_seed42` were unaffected (`objective_mode != "unconditional"`).
