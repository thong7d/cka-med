# Phase 4 — Model & Comparator Selection Rules

1. **Prompt baseline comparator (`B-P`)**: Selected `B-P_v1` over `B-P_v2` on `dev` (`Acc_0(V)` `0.8286` vs `0.8000`, `Acc_irrel(V)` `0.8143` vs `0.7857`, identical `Acc_near(V)` `0.7286` and `KUG` `+0.0286`).
2. **Main comparator for candidate `M-C`**: `B-S_seed42` (same-data, same-sampler, same-exposure SFT).
3. **Candidate `lambda_cond` selection on `dev` ONLY**:
   - Evaluate `lambda_cond in {0.5, 2.0}` at `seed=42`.
   - Feasibility constraints relative to `B-S_seed42` on `dev`:
     - `Acc_0(V) >= Acc_0(B-S_seed42) - 0.03`
     - `Acc_help(J) >= Acc_help(B-S_seed42) - 0.03`
     - `KUG(J) >= KUG(B-S_seed42) - 0.03`
   - Among feasible candidates, select the one maximizing `Acc_near(V)` (tie-break: lower `CFR_near(V)`, then higher `KUG(J)`).
   - If neither candidate meets all constraints, record constraint violation transparently and still evaluate `M-C` (highest `Acc_near(V)` among candidates) on `test` per protocol without loosening the rule post-hoc.
4. **Final `test` split rule**: No hyperparameter, prompt, alias, or checkpoint may be tuned after evaluating `test.jsonl`.
