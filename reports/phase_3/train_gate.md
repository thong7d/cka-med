# Phase 3 — Train Gate Decision

- **Decision:** **`TRAIN`**
- **Date:** 2026-11-21 Asia/Bangkok

## Gate Checklist

1. **Data & label integrity:** `data/manifests/poc_v1.json` passed `cka_med.data.audit` with 0 errors (`2,355` records, 0 missing predictions across all 7 dev runs).
2. **Image usage & task validity:** On J (`N=35`), `B-R` (`77.14%`) outperforms text-only `D-K` (`45.71%`) by `+31.43 pp` and `D-Q` (`22.86%`) by `+54.29 pp`. On V (`N=70`), `B-N` reaches `81.43%` (`57` no-K-correct opportunities).
3. **Observed failure mode & trade-off:** `B-R` suffers `-10.00 pp` accuracy drop under `Knear` (`81.43% -> 71.43%`, `CFR_near = 17.54%`). `B-P_v1`/`B-P_v2` fail to recover `Acc_near` (`72.86%`) and hurt `KUG` (`+2.86 pp`). `B-F` eliminates `KUG` (`0.00 pp`).
4. **Compute budget check:** P3 profiling + 7 dev runs consumed ~`$0.40` Modal free credit (`$9.69` remaining of Starter free credits, `$0.00` billed). Full P5 + P6 plan (`5` 1-epoch L4 training runs + test/stress evals) fits comfortably under `$3.00`, preserving >70% of remaining credits.

Proceeding to Phase 4 pre-registration & freeze before launching Phase 5 training.
