# Phase 6 — Final Access & Independence Log

- Frozen test dataset: `data/processed/test.jsonl` (`sha256: 0ff76000836a6e8fb365af38aa214f19f00f1af318fe565cf0d15c2efd9fa8a6`)
- Pre-registration freeze: `reports/phase_4/freeze_manifest.json` (`2026-11-21T09:14:19+07:00`)
- All `P5` checkpoints (`B-S_seed42`, `M-C_lam05_seed42`, `M-C_lam20_seed42`, `A-U_seed42`, `B-S_seed43`, `M-C_lam05_seed43`) were trained and selected solely on `train.jsonl` + `dev.jsonl` before any `test.jsonl` prediction was generated.
- No hyperparameters, prompt templates, answer aliases, or checkpoints were modified after running `test.jsonl`.
- Prediction coverage: `0` missing predictions across all `11` test runs and `3` stress test runs.
- Paid compute used: `$0.00` (`$4.47` of free Modal Starter credits used; `$5.62` remaining).
