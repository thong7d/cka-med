# Báo cáo PoC CKA-MedVQA: Khả năng phục hồi trước tri thức gây nhiễu và tính hữu dụng trên SLAKE English

**Ngày hoàn thành:** 2026-11-21 (Asia/Bangkok)  
**Backbone:** `Qwen/Qwen2.5-VL-3B-Instruct` (revision `66285546d2b821cf421d4f5eb2576359d3770cd3`, Qwen Research License)  
**Benchmark:** `BoKelvin/SLAKE` English + curated KG subset (revision `a9083ce6c34ac3ffb17671a605962924d8a8f9e9`, CC BY 4.0)  
**Quyết định cuối:** **`GO — benchmark/analysis`** (Kết quả âm đối với giả thuyết phương pháp `M-C > B-S`: `REFUTE`)

---

## 1. Câu hỏi nghiên cứu và phạm vi

PoC kiểm định ba câu hỏi theo `docs/POC_PLAN.md`:
1. Trên tập câu hỏi thị giác (`V`, binary yes/no), việc đưa tri thức ngoài từ KG (`Kirrel`, `Knear`) có làm mô hình đổi đáp án đúng thành sai (`CFR`) không?
2. Trên tập câu hỏi cần kết hợp ảnh và tri thức đã được audit (`J`), context phù hợp (`Khelp`) có đem lại lợi ích quan sát được (`KUG`) không, và mô hình có thực sự phụ thuộc đồng thời vào cả ảnh lẫn tri thức không?
3. Với cùng backbone, dữ liệu, paired sampler và số lần tiếp xúc (`2,080` record exposures/epoch), regularization nhất quán có điều kiện (`M-C`) có cải thiện trade-off tốt hơn same-data SFT (`B-S`), prompt ưu tiên ảnh (`B-P_v1`) và bộ lọc từ khóa (`B-F`) không?

Không có bác sĩ tham gia review trong PoC; do đó `premise_label=null`, `verification=not_established`, và không đưa ra kết luận về độ an toàn lâm sàng thực tế.

---

## 2. Dữ liệu, deduplication và cách tạo context

- **Kiểm tra trùng lặp ảnh (SHA-256 + 64-bit dHash):** Kiểm tra toàn bộ `642` ảnh nguồn (`source.jpg`; tuyệt đối không nạp `mask.png`). Không có trùng byte chính xác giữa các split, nhưng phát hiện các cặp near-duplicate perceptual (`hamming_distance <= 2`) giữa `train`/`validation` và `test` (như `xmlab395`/`xmlab401`, `xmlab602`/`xmlab604`, `xmlab624`/`xmlab622`). Các ảnh trùng ở `train`/`validation` đã bị loại tất định để bảo vệ tính độc lập của `test` (`reports/phase_2/split_audit.json`).
- **Quy mô các tập đã khóa (`data/manifests/poc_v1.json`):**
  - `train`: `520` groups (`360 V`, `160 J`), `1,560` variant-records, `283` unique images (`90` CXR images).
  - `dev` (`validation`): `105` groups (`70 V`, `35 J`), `315` variant-records, `50` unique images (`23` CXR images).
  - `test`: `160` groups (`115 V`, `45 J`), `480` variant-records, `63` unique images (`112` CXR groups trên toàn bộ `30` CXR images).
- **Controlled context replay (`data/processed/kg_subset.json`, `69` entries):**
  - `Khelp` (J): gói đa-cơ quan theo `question_family` (`organ_system` 8 dòng, `organ_function` 10 dòng), không có đáp án đơn lẻ kiểu oracle.
  - `Knear` (V): 3 dòng mô tả bệnh lý từ `KG/en_disease.csv` chọn chỉ từ từ khóa trong câu hỏi `q` (không đọc `location`, `triple` hay `answer`).
  - `Kirrel` (V & J): 3 dòng giải phẫu không liên quan (`Tooth`, `Eye`, `Spine`).

---

## 3. Thiết kế phương pháp và tính công bằng

- **Frozen vision encoder:** Mọi lớp `visual.*` bị đóng băng (`0` tham số trainable). LoRA (`r=8, alpha=16, dropout=0.05`) chỉ gắn vào `q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj` của phần ngôn ngữ (`bfloat16`, `min_pixels=200704`, `max_pixels=401408`).
- **Loss có điều kiện (`M-C`):** Trên `V`, `p_theta` là softmax trên 2 token cố định `A` (`yes`) / `B` (`no`). `L_cond = mean JS(p_theta(.|I,q,K0), p_theta(.|I,q,Kk))` chỉ áp dụng khi `task==V`, `gold_invariant`, `accepted`, và `k in {Kirrel, Knear}` (giữ gradient ở cả hai nhánh). Trên `J`, chỉ tính answer-only token NLL.
- **Công bằng tuyệt đối giữa `B-S`, `M-C`, `A-U`:** Chạy trên cùng `1,040` cặp (`2,080` exposures), `130` optimizer steps, cùng seed (`42` và `43`). Thời gian training thực tế: `B-S_seed42` = `1,947.8s`, `M-C_lam05_seed42` = `2,042.8s` (+4.9%), cùng peak VRAM `9,197.5 MB`.
- **Chọn mô hình trên `dev` trước khi mở `test` (`reports/phase_5/selection_decision.md`):** `M-C_lam05_seed42` (`Acc_near=85.71%`, `CFR_near=0.00%`, `Acc_0=84.29%`, `KUG=+2.86 pp`) đạt mọi ràng buộc P4 và được chọn; `M-C_lam20_seed42` bị loại vì vi phạm margin `KUG` (`-2.86 pp` so với `+2.86 pp` của `B-S_seed42`).

---

## 4. Kết quả chính trên Final Test (`N=480` records, `63` image clusters, `0` missing)

### 4.1. Bảng kết quả tổng hợp (`reports/phase_6/main_results.csv`)

| Model | `Acc_0(V)` | `Acc_irrel(V)` | `Acc_near(V)` | `F1_near(V)` | `CFR_irrel` | `CFR_near` | `Acc_0(J)` | `Acc_help(J)` | `KUG(J)` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `D-Q` (q only, J) | — | — | — | — | — | — | 0.2889 | 0.2889 | 0.0000 |
| `D-K` (q+K, J) | — | — | — | — | — | — | 0.2889 | 0.4889 | +0.2000 |
| `B-N` (I+q, no-K) | 0.8870 | 0.8870 | 0.8870 | 0.8857 | 0.0000 | 0.0000 | 0.7111 | 0.7111 | 0.0000 |
| `B-R` (I+q+K raw) | 0.8870 | 0.7391 | 0.7652 | 0.7593 | 0.1961 | 0.1569 | 0.7111 | 0.7556 | +0.0444 |
| `B-P_v1` (priority prompt) | **0.9043** | 0.8087 | 0.7913 | 0.7854 | 0.1154 | 0.1346 | 0.7333 | 0.8000 | **+0.0667** |
| `B-F` (keyword filter) | 0.8870 | **0.8870** | 0.7652 | 0.7593 | **0.0000** | 0.1471 | 0.7111 | 0.7333 | +0.0222 |
| `B-S_seed42` (same-data SFT) | 0.8435 | 0.8435 | 0.8348 | 0.8346 | **0.0000** | **0.0103** | 0.8667 | **0.8667** | 0.0000 |
| `M-C_lam05_seed42` | 0.8435 | 0.8435 | 0.8348 | 0.8346 | **0.0000** | **0.0103** | **0.8889** | **0.8667** | -0.0222 |
| `A-U_seed42` (uncond. abl.) | 0.8522 | 0.8522 | 0.8348 | 0.8346 | **0.0000** | 0.0204 | 0.8222 | **0.8667** | +0.0444 |
| `B-S_seed43` | 0.8435 | 0.8261 | 0.8348 | 0.8346 | 0.0206 | 0.0206 | 0.8667 | 0.8444 | -0.0222 |
| `M-C_lam05_seed43` | 0.8348 | 0.8261 | 0.8348 | 0.8346 | 0.0104 | 0.0208 | 0.8667 | 0.8444 | -0.0222 |

### 4.2. So sánh theo cặp và Bootstrap CI (`2,000` replicates theo `image_id`)

- **`B-R` so với `B-S_seed42` (`reports/phase_6/paired_deltas.csv`):**
  - `Acc_irrel(V)` delta: `-10.43 pp` (95% CI `[-18.18, -3.25] pp`); `CFR_irrel` delta: `+19.61 pp` (95% CI `[+12.36, +26.67] pp`).
  - `Acc_near(V)` delta: `-6.96 pp` (95% CI `[-15.74, +0.90] pp`); `CFR_near` delta: `+14.66 pp` (95% CI `[+7.46, +22.22] pp`).
  - `Acc_help(J)` delta: `-11.11 pp` (95% CI `[-20.51, -2.63] pp`).
- **`M-C_lam05_seed42` so với `B-S_seed42`:**
  - `Acc_near(V)` delta = **`0.0000`** (95% CI `[0.0000, 0.0000]`).
  - `Acc_0(V)` delta = **`0.0000`** (95% CI `[0.0000, 0.0000]`).
  - `CFR_near(V)` delta = **`0.0000`** (95% CI `[0.0000, 0.0000]`).
  - `KUG(J)` delta = `-2.22 pp` (95% CI `[-7.84, 0.00] pp`).
- **Stress test (held-out rendering template + đảo thứ tự pack, `reports/phase_6/stress_results.csv`):** `B-S_seed42_test_stress` và `M-C_lam05_seed42_test_stress` đạt kết quả giống hệt nhau (`Acc_0=85.22%`, `Acc_near=84.35%`, `CFR_near=2.04%`, `Acc_help=84.44%`).
- **CXR exploratory slice (`336` records / `30` images):** Trên base `B-N`, `Acc_0(V)=94.94%`, `B-R` giảm xuống `72.15%` dưới `Kirrel` (`CFR_irrel=25.33%`) và `87.34%` dưới `Knear`; cả `B-S_seed42` và `M-C_lam05_seed42` đều đạt `Acc_near=86.08%`, `CFR_near=0.00%`, trong khi `Acc_0(J)` tăng lên `96.97–100%`.

---

## 5. Phân tích nguyên nhân và bài học

1. **Failure mode là có thật và `B-F` không đủ:** Dưới `Knear`, `B-R` hay đổi đáp án đúng `"no"` sang `"yes"` (FP tăng gấp đôi `10 -> 20`, ví dụ câu `"Does the picture contain kidney?"`). Bộ lọc từ khóa `B-F` chặn hoàn toàn `Kirrel` (`CFR_irrel=0%`) vì khác từ khóa câu hỏi, nhưng cho lọt toàn bộ `Knear` (`CFR_near=14.71%`).
2. **Vì sao `M-C` không vượt `B-S`:**
   - Khi huấn luyện trên cùng dữ liệu đa ngữ cảnh với gold answer bất biến (`K0`, `Kirrel`, `Knear`), `B-S` tự động học cách giữ phân phối đầu ra ổn định trước `Kirrel`/`Knear`, khiến JS divergence giữa nhánh `K0` và `Kk` vốn đã tiến sát `0` (`3.5e-4` ở seed 42; `4.6e-5` ở seed 43). Thêm `lambda * L_cond` không tạo biên độ cải thiện bổ sung.
   - Đồng thời, `B-S`/`M-C` làm `Acc_0(V)` giảm nhẹ `-4.35 pp` so với `B-N` (`88.70% -> 84.35%`, `FN` tăng `3 -> 14`, `FP` giảm `10 -> 5`), cho thấy SFT chuyển mô hình sang thiên hướng bảo thủ (`no` nhiều hơn).
3. **Điểm nghẽn ở `J` sau fine-tune:** Trên bộ 37 đáp án đóng của SLAKE, SFT ghi nhớ liên hệ hệ cơ quan/chức năng (`Acc_0(J)` tăng từ `71.11%` lên `86.7–88.9%`), làm giảm dư địa đo `KUG` sau train. Các lỗi còn lại trên `J` chủ yếu là định vị thị giác (nhầm `Heart` thành `Lung` trên CXR với câu hỏi `"center organ"`).

---

## 6. Chi phí thực tế và Tái lập

- **Chi phí Modal L4 (`bfloat16`):** Tổng chi phí PoC trên Modal = **`$4.48`** trừ trực tiếp vào free Starter credit (`$0.00` billed; còn lại `$5.61` credit). Kaggle GPU chưa sử dụng (`0` giờ; thin runner nằm ở `notebooks/kaggle_runner.ipynb`).
- **Tái lập đã kiểm chứng (`reports/phase_7/repro_verification.json`, `gpu_reload_repro.json`):** `pytest` đạt `12/12`, `cka_med.data.audit` đạt `PASS`, metric tính lại từ raw JSONL khớp `100%`, và nạp lại adapter `M-C_lam05_seed42` trên GPU cho kết quả khớp `15/15` mẫu.
