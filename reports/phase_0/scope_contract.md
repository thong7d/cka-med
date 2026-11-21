# P0 — Scope contract

## Core scope

- Dataset: SLAKE English metadata and KG.
- Pilot radiology: English SLAKE plus its bundled KG.
- CXR: report as an exploratory slice only.
- Primary model: `Qwen/Qwen2.5-VL-3B-Instruct`, frozen vision encoder, LoRA/QLoRA only if training becomes approved.
- Optional alternative: `google/medgemma-4b-it`; not a default dependency and not selected unless access and development competence are confirmed before P3.
- No clinical support, disease, absence, unknown, or premise labels may be synthesized from images.
- No patient-specific diagnosis or clinical safety claim.

## Allowed task

1. Test whether context interventions change answers on fixed SLAKE reference groups.
2. Measure visual competence and knowledge utility on audited visual and joint groups.
3. If training is approved, compare same-data SFT and one conditional consistency candidate against prompt/filter baselines.

## Explicit exclusions

- VinDr-CXR and MIMIC-CXR-JPG credentialing.
- External clinical labeling or radiologist adjudication.
- RL/DPO/GRPO, multiple full backbones, large ontology integration, support head training.
- Paid compute or paid credentials.
- Turning benchmark reference results into clinical applicability claims.

## Positive claim threshold

A claim requires paired image clusters, fixed final protocol, non-leaking context, raw predictions, missingness, clustered confidence intervals, and review of the exact denominators.
