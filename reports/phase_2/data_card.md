# Phase 2 — Pilot data card (SLAKE English + Curated KG)

- Source dataset: `BoKelvin/SLAKE` (CC BY 4.0), revision `a9083ce6c34ac3ffb17671a605962924d8a8f9e9`
- Context generator version: `cka_ctx_v1`
- Reviewer type: `agent_audited` (no specialist clinician review in PoC scope)

## 1. Split & Image Deduplication Summary

- Total unique split images inspected across `train`, `validation`, `test`: **642** (`source.jpg` only; `mask.png` explicitly excluded and never loaded).
- Exact SHA-256 cross-split duplicates: **0**.
- Near-duplicate perceptual dHash (`hamming_distance <= 2`) cross-split pairs detected and resolved by deterministic exclusion from `train`/`validation` (keeping `test` intact): logged in `reports/phase_2/split_audit.json`.

## 2. Dataset Counts (`data/manifests/poc_v1.json`)

| Split | Total Records | Total Groups | V Groups | J Groups | Unique Images | CXR Groups | CXR Images |
|---|---:|---:|---:|---:|---:|---:|---:|
| `train` | 1,560 | 520 | 360 | 160 | 283 | 153 | 90 |
| `dev` (`validation`) | 315 | 105 | 70 | 35 | 50 | 57 | 23 |
| `test` | 480 | 160 | 115 | 45 | 63 | 112 | 30 |

## 3. Task Eligibility & Exclusions

- **V (Visual primary):** `base_type == "vqa"`, `answer_type == "CLOSED"`, `normalized_answer in {"yes", "no"}`, `content_type in {"Abnormality", "Organ", "Position"}`. `Modality/Plane/Color/Size/Quantity/Shape` excluded from primary V core (`exclusions.csv`).
- **J (Joint candidate):** `base_type == "kvqa"`, anatomy/system/function relations (`belong to`, `part of`, `function`, `effect`, `be used for`, `could`, `can`, `role`), zero organ-name leakage in question, clinical `treat/prevent/cause/symptom/damage` excluded, `train_j_vocab` locked solely from `train` (37 canonical answers).
- Total exclusions logged in `reports/phase_2/exclusions.csv`: **6,248** rows (non-English, open VQA, clinical KVQA, near-duplicate cross-split images, organ-leaked questions, budget cap).

## 4. Curated KG Subset & Context Variants

- `data/processed/kg_subset.json`: **69** provenance-backed entries from `KG/en_organ_rel.csv`, `KG/en_organ.csv`, and `KG/en_disease.csv` (`source_asserted`).
- `K0`: empty external context.
- `Khelp` (for J): multi-organ controlled replay packs (`pack_Khelp_organ_system` 8 entries, `pack_Khelp_organ_function` 10 entries); `q + K` alone cannot isolate the image-present organ.
- `Knear` (for V): 3-line near-topic disease notes from `en_disease.csv` selected solely from question keywords (never `location`, `triple`, or gold answer), without negative cue words.
- `Kirrel` (for V & J): 3-line unrelated anatomical facts (`Tooth`, `Eye`, `Spine`).
- `premise_label = null`, `verification = not_established` throughout.
