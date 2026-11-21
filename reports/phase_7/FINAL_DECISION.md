# Phase 7 — Final PoC Decision

- **Date:** 2026-11-21 Asia/Bangkok
- **PoC Verdict (per `docs/POC_PLAN.md` §9 P7):** **`GO — benchmark/analysis`**
- **Scientific Verdict on Conditional Consistency (`M-C` vs `B-S`):** **`REFUTE`** (no measurable superiority over same-data multi-context SFT)

## Decision Basis

1. **Engineering & Protocol Integrity (PASS):** Zero data leakage (`cka_med.data.audit` PASS on 2,355 records after perceptual dHash cross-split deduplication), zero missing predictions across all 14 `test`/`stress` runs, exact metric recomputation from raw JSONL predictions, and 15/15 exact adapter-reload verification.
2. **Failure Mode & Baseline Trade-off Confirmed:** On frozen SLAKE `test` (`160` groups / `63` images), raw context injection (`B-R`) causes `-12.17 pp` drop under `Knear` (`CFR_near = 15.69%`) and `-14.78 pp` drop under `Kirrel` (`CFR_irrel = 19.61%`). Keyword filtering (`B-F`) blocks `Kirrel` (`CFR_irrel = 0%`) but fails on near-topic `Knear` (`CFR_near = 14.71%`).
3. **Method Pilot Outcome (`M-C` vs `B-S`):** Same-data multi-context SFT (`B-S`) already reduces `CFR_near` to `1.03%` (seed 42) / `2.06%` (seed 43), while conditional JS consistency (`M-C`, `lambda=0.5`) yields `0.0000` paired delta on `Acc_near(V)` (`83.48%` vs `83.48%`), `0.0000` on `Acc_0(V)` (`84.35%` vs `84.35%`), and identical `84.35%` on stress rendering.
4. **Why `M-C` Did Not Separate From `B-S` in This Pilot:**
   - Under `B-S`, the residual JS divergence between `K0` and `{Kirrel, Knear}` was already near zero by epoch end (`final_cond_loss = 3.5e-4` on seed 42, `4.6e-5` on seed 43).
   - Closed-vocabulary SLAKE `J` (`45` test groups) reached `86.67–88.89%` under `K0` after SFT (`Acc_0(J)`), leaving almost no headroom for `Khelp` (`KUG <= 0` after SFT) and turning remaining errors into visual organ localization errors (`Heart` vs `Lung`).

## Prioritized Graduation Thesis Backlog

1. **Drop unsupported `M-C > B-S` superiority claim**; refocus thesis contribution on **diagnostic protocol, failure taxonomy, and evidence-applicability evaluation**.
2. **Acquire credentialed CXR data + clinician annotations** (e.g., VinDr-CXR / specialist rubric) where `premise_label in {supported, contradicted, unknown}` is genuinely grounded rather than `null`, and where unseen medical knowledge cannot be memorized from a 37-answer closed vocabulary.
3. **Investigate the clean-accuracy vs robustness trade-off of SFT** (`B-S` dropped clean `Acc_0(V)` by `-4.35 pp` vs `B-N`, increasing `FN` from `3` to `14` while cutting `FP` from `10` to `5`).
4. **Test true held-out rule families / unseen external KG sources** before proposing any new conditional learning objective.
