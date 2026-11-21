# Protocol v1 — Pre-Registered PoC Specification

- Frozen at: `2026-11-21T09:16:00+07:00` (before P5 training and before any model runs on `test.jsonl`)
- Backbone: `Qwen/Qwen2.5-VL-3B-Instruct` (`66285546d2b821cf421d4f5eb2576359d3770cd3`), frozen vision encoder, `bfloat16`, pixel bounds `[200704, 401408]`.
- Data splits (`data/manifests/poc_v1.json`):
  - `train`: `520` groups (`360 V`, `160 J`), `1,560` records, `283` unique images
  - `dev`: `105` groups (`70 V`, `35 J`), `315` records, `50` unique images
  - `test`: `160` groups (`115 V`, `45 J`), `480` records, `63` unique images (`112` CXR groups across `30` CXR images)

## Primary Endpoints & Hypothesis

- Compare conditional consistency (`M-C`) against same-data SFT (`B-S`) under identical paired sampler, exposure (`2,080` variant-record forwards per epoch), LoRA config, and seed.
- Practical comparators: `B-N`, `B-R`, `B-P_v1` (selected on dev over `B-P_v2`), `B-F`.
- Primary endpoints: `Acc_near(V)`, `Acc_0(V)`, `KUG(J) = Acc_help(J) - Acc_0(J)`.
- Non-inferiority margins (PoC-specific, not community standard): `-3.0 pp` (`-0.03`) on `Acc_0(V)`, `Acc_help(J)`, and `KUG(J)`.
- Statistical inference: paired image-cluster bootstrap (`2,000` replicates, `seed=20261121`, `image_id` cluster). Method superiority requires paired `Acc_near(V)` delta 95% CI lower bound `> 0`, non-inferiority lower CIs `> -0.03`, and valid `J` / dependency controls.
