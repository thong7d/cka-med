# Phase 5 — Checkpoint & Candidate Selection Decision (Dev Only)

- **Selected `M-C` candidate:** `M-C_lam05_seed42` (`lambda_cond = 0.5`)
- **Main comparator:** `B-S_seed42` (same-data, same-sampler, same-exposure SFT)
- **Ablation:** `A-U_seed42` (`lambda_cond = 0.5`, unconditional)
- **Second seed pair:** `B-S_seed43` and `M-C_lam05_seed43`

## Dev Selection Table (N=315, 0 missing)

| Model | `Acc_0(V)` | `Acc_near(V)` | `CFR_near(V)` | `Acc_0(J)` | `Acc_help(J)` | `KUG(J)` | P4 Constraints vs `B-S_seed42` |
|---|---:|---:|---:|---:|---:|---:|---|
| `B-S_seed42` | 0.8429 | 0.8571 | 0.0169 | 0.8286 | 0.8571 | +0.0286 | Reference comparator |
| `M-C_lam05_seed42` | 0.8429 | 0.8571 | **0.0000** | 0.8286 | 0.8571 | +0.0286 | **FEASIBLE (Selected)** |
| `M-C_lam20_seed42` | 0.8429 | 0.8429 | 0.0169 | 0.8857 | 0.8571 | -0.0286 | Violates KUG margin (-5.71 pp < -3.0 pp) |
| `A-U_seed42` | 0.8429 | 0.8571 | 0.0000 | 0.8000 | 0.8571 | +0.0571 | Ablation |
| `B-S_seed43` | 0.8286 | 0.8429 | 0.0000 | 0.8571 | 0.8571 | 0.0000 | Seed 43 comparator |
| `M-C_lam05_seed43` | 0.8286 | 0.8429 | 0.0000 | 0.8571 | 0.8571 | 0.0000 | Seed 43 candidate |

Selection strictly followed `reports/phase_4/selection_rules.md` before any evaluation on `test.jsonl`.
