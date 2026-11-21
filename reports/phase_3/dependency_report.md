# Phase 3 — Dependency and Competence Report (Dev Split)

- Split: `dev` (`105` groups = `70 V` + `35 J`, `50` unique images, `315` variant-records)
- Backbone: `Qwen/Qwen2.5-VL-3B-Instruct` (`66285546d2b821cf421d4f5eb2576359d3770cd3`), `torch.bfloat16`, NVIDIA L4.

## 1. Visual Competence (V, N=70)

- `B-N` (`I + q`, no-K): `Acc_0(V) = 0.8143` (`57/70`), `Macro-F1 = 0.7918` (`TP=40, TN=17, FP=9, FN=4`).
- No-K ceiling is NOT reached (`13/70` errors remain, and `57/70` correct opportunities exist for `CFR` denominator).

## 2. Knowledge-Induced Conflict Failure Mode (V, N=70)

- Adding `Knear` (`B-R`): `Acc_near(V)` drops from `0.8143` to `0.7143` (**-10.00 pp**), `Macro-F1` drops from `0.7918` to `0.6500`, `CFR_near = 0.1754` (`10/57` correct answers flipped to wrong, `FP` rises `9 -> 16`, `TN` drops `17 -> 10`).
- Adding `Kirrel` (`B-R`): `Acc_irrel(V) = 0.8429`, `CFR_irrel = 0.0702` (`4/57` flipped).
- Evidence-priority prompt (`B-P_v1`): `Acc_near = 0.7286` (`CFR_near = 0.1724`), while reducing `KUG` on J from `+8.57 pp` to `+2.86 pp`.
- Deterministic relevance filter (`B-F`): `Acc_near = 0.7429` (`CFR_near = 0.1228`), `CFR_irrel = 0.0000`, but drops useful `Khelp` on J so `KUG = 0.0000` (`0` net gain).

## 3. Joint Image + Knowledge Dependence Controls (J, N=35)

| Model | Input | `Acc_0(J)` | `Acc_help(J)` | `KUG` | Rescued / Harmed |
|---|---|---:|---:|---:|---:|
| `D-Q` | `q` only | 0.2286 (8/35) | 0.2286 (8/35) | 0.0000 | 0 / 0 |
| `D-K` | `q + K` (no image) | 0.2286 (8/35) | 0.4571 (16/35) | +0.2286 | 8 / 0 |
| `B-N` | `I + q` (no K) | 0.6857 (24/35) | 0.6857 (24/35) | 0.0000 | 0 / 0 |
| `B-R` | `I + q + K` | 0.6857 (24/35) | **0.7714 (27/35)** | **+0.0857** | 4 / 1 |

Interpretation:
- `D-Q` (`22.86%`) and `D-K` (`45.71%` with `Khelp`) cannot solve `J` without the image (`B-R` reaches `77.14%`, **+31.43 pp** above `D-K`).
- `Khelp` helps `B-R` over `B-N` (`68.57% -> 77.14%`, **+8.57 pp**, `4` rescued vs `1` harmed).
- Image tensors were verified (`pixel_values` non-empty for all multimodal runs; `uses_image=False` for `D-Q`/`D-K`).
