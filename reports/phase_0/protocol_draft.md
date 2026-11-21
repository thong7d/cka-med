# P0 — Protocol draft

## Input contract

Allowed input:
- image tensor,
- question,
- frozen context pack,
- condition label only in the evaluator/sampler.

Forbidden model/retriever input:
- final test answer,
- gold finding,
- gold entity,
- location label,
- patient report,
- knowledge source role or applicability label used as an oracle,
- context triple selected because it contains the test answer.

## Context variants

- `K0`: no external knowledge.
- `Khelp`: source-backed knowledge family used for a fixed joint candidate.
- `Kirrel`: source-backed entry irrelevant to the question family.
- `Knear`: source-backed near-topic distractor not established as an applicable finding.
- `Kfalse`: explicit false-assertion diagnostic control, optional and isolated.
- `O-H`: gold-selected oracle pack, optional diagnostic upper bound only.

## Baselines

- D-Q, D-K, B-N, B-R, B-P, B-F are mandatory only according to later approved phases.
- B-S and M-C require P4/P5 gates and sufficient free quota.

## Primary task

- Primary visual output: constrained yes/no.
- Knowledge short output: fixed train-vocabulary or prelocked choices.
- Missing/invalid prediction remains in the eligible denominator; it is not silently dropped.

## Primary endpoints

- `Acc_near`
- `Acc0`
- `CFR`
- `KUG = Acc_help(J) - Acc0(J)`
- paired clustered image bootstrap for confidence intervals

## Decision outcomes

- `SUPPORT` only if paired robustness and utility evidence are both valid.
- `REFUTE` if candidate does not improve, while engineering can still pass.
- `INCONCLUSIVE` if J is too small, denominators are unstable, or controls fail.
- Clinical applicability is always out of scope.
