# Kế hoạch PoC CKA-MedVQA: hoàn thành trước 23/11/2026

Phiên bản 1.0 — 16/11/2026 — Asia/Bangkok. Gemini triển khai; Codex nghiệm thu từng phase; người dùng quyết định thay đổi scope hoặc bổ sung tài nguyên. Các con số **mục tiêu** dưới đây không phải dữ liệu/kết quả đã có.

## 1. Mục tiêu và mức kết luận cho phép

PoC trả lời ba câu hỏi theo thứ tự:

1. Trên benchmark có reference answer, thêm tri thức ngoài có làm mô hình đổi một đáp án đúng thành sai không? Hiện tượng còn tồn tại khi kiểm soát format, độ dài, chủ đề và năng lực đọc ảnh không?
2. Trên câu hỏi cần kết hợp ảnh và tri thức đã được audit, context phù hợp có đem lại lợi ích quan sát được không? Nếu không, nguyên nhân là ceiling, task không dùng ảnh, context kém hay model không dùng K?
3. Với cùng backbone/dữ liệu/sampler, regularization có điều kiện có tạo trade-off tốt hơn same-data SFT và prompt/filter đơn giản không?

Đầu ra bắt buộc là **một quyết định có căn cứ về hướng khóa luận** kèm code/config/raw predictions tái lập được. Kết quả âm hoặc chưa đủ bằng chứng là đầu ra hợp lệ. Một candidate không thắng baseline không đồng nghĩa công việc kỹ thuật thất bại.

Trong tuần này, không có bác sĩ review. Vì vậy chỉ được kết luận về hành vi trên **benchmark references và context interventions đã audit**, chưa được nói có benchmark clinical applicability được xác nhận hoặc mô hình an toàn hơn trong thực hành y tế.

## 2. Ràng buộc đã xác nhận và lựa chọn kỹ thuật

| Hạng mục | Quyết định |
|---|---|
| Hạn chót | 23/11/2026; giao bản cuối trong ngày theo Asia/Bangkok |
| Ngân sách | Không phát sinh chi trả; Kaggle và credits Modal còn thực tế |
| Dữ liệu | SLAKE English làm primary pilot radiology; báo riêng CXR candidates, không trộn hai scope trong claim |
| Ground truth | Reference answer sẵn có; không tự đọc ảnh rồi gán bệnh/absence/unknown mới |
| Backbone | Qwen2.5-VL-3B-Instruct mặc định; MedGemma 4B nếu có quyền và đổi sớm có lý do |
| Train | Một backbone, frozen vision encoder, LoRA/QLoRA; SFT trước CKA; không RL |
| Tri thức | Tập nhỏ từ KG đi kèm, giữ provenance; relation nghi vấn loại bỏ; không ontology toàn ngành |
| Primary output | Binary visual QA có đáp án yes/no, constrained label scoring |
| Knowledge output | Short answers trong vocabulary cố định từ train, hoặc task-specific choices đã khóa trước test; giữ mọi thay đổi format công bằng |
| Uncertainty/support head | Để backlog vì chưa có gold chuyên môn phù hợp; không bắt buộc train ba trạng thái trong PoC |
| Phạm vi đóng góp | Pipeline/protocol + failure analysis + method pilot; không hứa novelty/SOTA |

Tham khảo [phân tích nguồn](SOURCE_ANALYSIS.md) và [access](RESOURCE_ACCESS.md). Model/dataset revision, thư viện và runtime phải được pin bằng các giá trị thực sự chạy được; không dùng `latest` làm bằng chứng tái lập.

## 3. Timeline và gate

| Phase | Ngày mục tiêu | Đầu ra chính | Gate trước bước tiếp theo |
|---|---|---|---|
| P0 — Chốt scope, access, protocol sơ bộ | 16/11 | Resource probes, decision log, experiment contract | Tài nguyên và task đủ rõ, không có dependency cấp quyền dài |
| P1 — Repo và evaluator | 17/11 sáng | CPU fixtures, schema, CLI, manifest, runner skeleton | Metric/packing/input isolation đúng |
| P2 — Pilot data và context | 17/11 chiều–18/11 sáng | Development groups, KG subset, label audit | Giữ gold, không oracle input hoặc nhãn y khoa tự sinh |
| P3 — Baseline và profiling | 18/11 | No-K/raw/prompt/filter + dependency controls | Task đủ hợp lệ; biết compute; quyết định có nên train |
| P4 — Khóa dataset và pre-registration | 19/11 | Train/dev/test manifests, protocol v1, configs | Không chọn theo final test; đủ quota cho cặp so sánh |
| P5 — SFT và candidate | 20–21/11 | Hai adapters chính, logs, dev comparison | Tính công bằng và objective có tác động thực |
| P6 — Final evaluation và stress | 22/11 | Raw predictions, clustered CI, ablation, error slices | Kết quả khớp protocol; không tune bằng final test |
| P7 — Tái lập và quyết định | 23/11 | Báo cáo PoC, reproducibility bundle, go/pivot | Codex recompute được và claim khớp evidence |

Đây là lịch đích, không bảo đảm thời gian chờ GPU/review. Sau mỗi phase Gemini đóng gói hồ sơ và dừng để Codex nghiệm thu. Chưa PASS thì chỉ sửa finding hoặc làm việc CPU độc lập được reviewer nêu rõ, không tự chạy phase GPU phụ thuộc phía sau.

Nếu trễ một ngày, bỏ extension trước: backbone thứ hai, bilingual, external dataset, support head, nhiều seed. Không bỏ same-data SFT, leakage audit hoặc raw predictions. Đến hết 21/11 phải khóa candidate; 22–23/11 ưu tiên đánh giá và giao kết quả, không đổi kiến trúc lớn.

## 4. Định nghĩa dữ liệu và điều kiện thông tin

### 4.1. Đơn vị và split

`group_id = một image + một question gốc`; mỗi group có các variant K, giữ nguyên answer reference nếu intervention chỉ thay context không mang observation mới. Nếu có nhiều QA trên một ảnh, chúng cùng split và cùng statistical cluster.

Giữ official train/validation/test của SLAKE làm nguồn. Chọn subset deterministic bằng seed, rule và budget khóa trước; không re-split QA để tăng số ảnh. Không thấy patient ID thì cluster theo image, ghi rõ limitation. Sau tải ảnh, kiểm tra file hash và near duplicates giữa split; nếu phát hiện trùng, áp rule loại cố định, lưu exclusion log và manifest version mới.

Không tính QA variants hay bản dịch là observations độc lập. Không nói patient-disjoint khi chỉ có image IDs. Không chứng minh được model chưa từng thấy benchmark trong pretraining.

### 4.2. Ba tầng QA

- **V — Visual primary:** source `vqa`, CLOSED, canonical answer yes/no có reference rõ; ưu tiên finding, organ visibility và position có nghĩa. Modality recognition dễ chỉ làm competence/control, không là toàn bộ headline result.
- **J — Joint candidate:** source `kvqa`, anatomy/function/body-system với reference và entity association đủ rõ từ annotation. Audit không có tên target organ tự lộ trong q, không chỉ generic fact, không cần chẩn đoán nguyên nhân/điều trị. Đạt tiêu chí mới vào J; không chọn J vì no-K của model đang sai.
- **C — Controls/không hợp lệ cho claim core:** generic knowledge-only, modality-only, câu hỏi điều trị/phòng bệnh/etiology không có chuyên môn xác nhận, ambiguous hoặc annotation conflict. Báo số lượng; không trộn với V/J để làm đẹp accuracy.

Không đủ J đáng tin thì báo utility **chưa kiểm chứng**, không sửa câu hỏi/answer để ép K có lợi. Anatomy joint task có thể khác failure mode clinical ban đầu; phải nói rõ khoảng cách đó trong kết luận.

### 4.3. Context variants

| Condition | Nội dung | Nhãn/giới hạn |
|---|---|---|
| `K0` | Không external context | Mốc I+q; không gọi là chỉ ảnh |
| `Khelp` | Knowledge pack phù hợp loại câu hỏi, chứa nhiều candidate hợp lý khi cần chọn bằng ảnh | Source-backed, không là một câu đáp án được chọn bằng gold test |
| `Kirrel` | Entry nguồn đúng theo benchmark nhưng không liên quan chủ đề | Control cơ bản, không gộp với near-topic |
| `Knear` | Knowledge tổng quát gần chủ đề, dễ gây association nhưng không tự xác nhận observation trên ảnh | Gọi là near-topic/source-supported distractor; chưa được gọi clinical not-applicable đã xác minh |
| `Kfalse` | False patient-specific assertion đối lập reference | Chỉ diagnostic adversarial control, optional; giữ metadata riêng, không trộn thành K thật |

Tối thiểu V có K0/Kirrel/Knear; J có K0/Khelp/Kirrel nếu hợp lệ. Không phải mọi group có mọi variant; evaluator dùng denominator tương ứng, báo missingness và không im lặng inner-join mất mẫu.

**Đối với premise status:** chỉ dùng `supported/contradicted/unknown` khi nguồn annotation thực sự gắn nhãn cho đúng proposition. Thiếu field không là semantic unknown. Trong PoC thường lưu `premise_label=null`, `verification=not_established`.

### 4.4. Controlled replay và retrieval thực tế

Đường chính là **controlled context replay**: knowledge pack nhỏ, xác định bằng question family và corpus đã khóa; cùng family dùng pack giống nhau trên nhiều ảnh. Ví dụ q về chức năng organ có một pack mô tả nhiều organ với độ dài tương tự; model phải chọn organ từ ảnh. Không chọn duy nhất triple chứa đáp án dựa trên test reference.

Nếu dùng gold để tạo một oracle-helpful pack nhằm kiểm tra upper bound, đánh dấu `oracle_context=true`; chỉ diagnostic, không tính thành main utility/deployable retrieval. Metadata `location`, gold entity, `triple` hay `content_type` không được lén đi vào model/retriever input.

Retriever BM25/alias theo q là đường phụ nếu làm kịp; dùng cùng corpus/top-k/budget cho mọi method. Không cần Neo4j, dense retriever hoặc external verifier. Không gọi controlled replay là đã xây RAG dùng được cho bệnh nhân.

### 4.5. Chất lượng và chống shortcut

Hai thành viên nếu có có thể review provenance, logic câu hỏi, formatting và rule mapping; review kỹ thuật không thay review ảnh chuyên môn. Nếu chỉ Gemini làm, ghi `agent_audited`, không ghi human/expert-reviewed. Codex audit mẫu và code độc lập; cũng không đóng vai bác sĩ.

Giữ nguyên source reference và citation row. Entry dịch sai/quan hệ quá mạnh hoặc QA khó hiểu: loại khỏi core, lưu lý do, không tự sửa bệnh học. Cân bằng độ dài, số triples, thứ tự và lexical overlap giữa Knear/Khelp khi có thể; dùng nhiều cách diễn đạt do rule/template, không chỉ negative chứa chữ “not”.

Train và final stress dùng template render khác nhau, source entry provenance giữ nguyên. Khác template chưa phải external knowledge source. Không hứa unseen-rule split nếu KG và số group không đủ; báo coverage thực tế.

## 5. Quy mô, compute và quy tắc cắt giảm

### 5.1. Data budget mục tiêu

| Tập | Mục tiêu trước lọc chất lượng | Lưu ý |
|---|---:|---|
| CPU fixture | 8–12 group giả lập, không ảnh bệnh nhân | Chỉ để kiểm tra code/metric |
| Development pilot | 60–100 group trên khoảng 25–40 ảnh từ validation | Tối thiểu có V và J candidates; không final claim |
| Train | 300–600 group từ official train, có V và J nếu J hợp lệ | 3–4 variants/group; đếm tokens và image clusters |
| Final test | 100–160 group, cố gắng phủ 60–96 ảnh official test | J mục tiêu 20–40; số thật sau audit có thể thấp hơn |
| CXR slice | Mọi group final thuộc CXR filter hợp lệ | Tối đa 30 ảnh nguồn trong metadata đã kiểm tra; chỉ exploratory |

Đây là mục tiêu cân bằng tính khả thi, không power analysis. Nếu J final dưới 20 group hoặc chỉ vài image clusters, KUG chỉ mô tả, không quyết định method superiority. Không tạo thêm synthetic gold để đạt quota. Báo cả số group và unique images theo task/condition.

### 5.2. Compute budget

Snapshot ngày 16/11: Kaggle còn khoảng 27,28 GPU-hours, refresh được báo 21/11 lúc 07:00. Quota có thể thay đổi khi tài khoản làm việc khác; recheck trước mọi lượt lớn. Modal credits chưa biết, không cộng vào ngân sách khả dụng.

Đặt ngân sách ban đầu tối đa **24 GPU-hours** từ phần Kaggle đã xác nhận; ít nhất **8 giờ trong số đó dành cho evaluation/export/retry**. Đây là trần điều phối của plan, không dự báo rằng mọi run chắc chắn cần bấy nhiêu. Sau profiling P3 tính lại từng job và có thể giảm scope.

| Nhóm việc | Trần lập lịch ban đầu |
|---|---:|
| Smoke test và baseline/dependency controls | 4 giờ |
| SFT + candidate + tuning nhỏ | 10 giờ |
| Ablation hoặc seed phụ nếu còn | 2 giờ |
| Final inference, statistics/export, retry | 8 giờ |

CPU statistics không cần GPU; bảng dành dự phòng cho inference và khắc phục run lỗi. Nếu inference chậm hơn dự báo, cắt seed/ablation trước. Không dựa vào quota refresh tương lai để khởi chạy một job hiện không đủ quota.

Profiling đo trên đúng GPU, resolution, prompt/context length và paired branches: warmup, 20–50 train steps ban đầu; tối đa 100 nếu cần ổn định tốc độ; 100–200 inference variant-records. Lưu peak VRAM, examples/s, tokens/s, overhead loading, checkpoint time và ước tính cả project với hệ số dự phòng 1,3. Giảm batch trước; không resize ảnh quá thấp mà không kiểm tra competence.

Một T4 16GB chạy được hay không là điều phải đo. Không cộng VRAM hai T4 rồi giả định có một GPU 32GB; chỉ dùng multi-GPU nếu code và budget hỗ trợ. Với T4 dùng dtype phần cứng hỗ trợ; không giả định bf16. Resume phải giữ optimizer/scheduler/RNG, không chỉ adapter.

## 6. Model, baselines và candidate

Mã dưới đây là **mã chuẩn của plan mới**, không lấy ID B0/B1 trong hai tài liệu cũ vì chúng không thống nhất.

| ID | Input/phương pháp | Bắt buộc |
|---|---|---|
| D-Q | q only | Có, diagnostic shortcut |
| D-K | q+K không ảnh | Có, diagnostic joint dependence |
| B-N | I+q, no-K | Có, competence và mốc utility |
| B-R | I+q+K raw | Có, quantification lợi/hại |
| B-P | B-R + evidence-priority prompt | Có, baseline rẻ |
| B-F | B-R + relevance/alias filter deterministic | Có; công khai rule và keep/drop counts |
| B-S | Same-data SFT, toàn bộ variants hợp lệ | Có nếu đi nhánh method |
| M-C | Cùng B-S + conditional consistency | Candidate duy nhất bắt buộc nếu đi nhánh method |
| A-U | Cùng B-S + consistency không điều kiện | Ưu tiên ablation nếu đủ quota |
| O-H | Gold-selected helpful context / support oracle | Optional, chỉ diagnostic upper bound |

B-F được phép dựa q và raw K, không đọc gold applicability/reference. B-S là hard-negative/multi-context SFT cùng data; không so candidate đã thấy Knear với baseline chưa thấy Knear rồi quy lợi ích cho loss.

Tối thiểu chạy B-S và M-C seed 42. Nếu đủ quota chạy seed 43 cho **cả cặp**, không chỉ seed có lợi. Một seed chỉ hỗ trợ kết luận thăm dò. Mọi model được chọn bằng validation; B-P tối đa hai prompt định sẵn, M-C tối đa hai lambda nonzero. Không full grid search.

### 6.1. Định nghĩa candidate có thể implement

Trên V, ánh xạ câu trả lời thành hai label tokens cùng độ dài, ví dụ A=yes/B=no; mapping cố định, tokenizer được kiểm tra. `p_theta` là softmax trên hai logits/completion scores hợp lệ ở answer position sau ảnh/q/K. Không dùng log-likelihood của cả prompt hoặc padding làm answer loss.

```text
L_ans = mean source-supervised answer NLL trên toàn bộ variant-records hợp lệ

mask(g,k) = task_is_visual
            AND gold_answer_is_unchanged_under_this_context
            AND label_provenance_is_accepted
            AND k in {Kirrel, Knear}

L_cond = mean JS(p_theta(. | I,q,K0), p_theta(. | I,q,Kk)) trên mask

L_total = L_ans + lambda_cond * L_cond
```

JS dùng hai nhánh còn gradient, gold CE ở cả nhánh theo sampler; không distill reference prediction sai, không `detach` toàn bộ student. Nếu mask rỗng, L_cond bằng scalar zero đúng device/dtype, log active_count=0; không NaN hoặc tính trung bình sai denominator.

J chỉ nhận supervised answer loss khi target có nguồn và format hợp lệ. **Không ép unknown/abstain khi không có external K**, vì model có thể biết kiến thức từ pretraining. Không thêm utility CE trùng L_ans rồi coi là một cơ chế khác.

M-C kiểm tra một regularizer đơn giản, chưa được coi là thuật toán mới. Nếu quan sát không vượt B-S thì giữ negative result. Support-supervised routing trong survey là nhánh nghiên cứu sau, khi có nhãn đáng tin; không tự lắp thêm head/gate giữa tuần để tạo claim.

### 6.2. Công bằng so sánh

SFT/candidate cùng base revision, train/dev/test, context text, tokenizer/processor, answer map, LoRA targets/rank, seed, effective batch và variant exposure. Candidate có thêm forward branches; phải log GPU-hours, forward tokens và số updates.

So sánh chính là **same-data/same-updates để cô lập objective**, kèm công bố compute chênh lệch; không gọi là same-compute nếu candidate tốn hơn. Nếu chi phí phụ đáng kể, thêm một B-S chạy trong cùng GPU-hour cap khi đủ budget, hoặc hạ claim efficiency. Không để same-data SFT có ít augmentation hơn.

Gợi ý khởi đầu để profile: QLoRA 4-bit nếu runtime hỗ trợ, LoRA rank 8, dropout 0,05, batch 1 + accumulation, tối đa 1–2 epochs, seed 42. Các giá trị thật phải khóa ở P4; không là cam kết phù hợp mọi backbone. Không train vision encoder hay lora_modules không tồn tại.

### 6.3. Unit checks có giá trị khoa học

Các test cần bảo vệ điểm dễ làm sai: answer-only token mask, image tensor thực sự được truyền, batch grouping, conditional mask, gradient đi vào adapter, lambda=0 tương đương B-S trong cùng batch, resume state, role metadata không rò vào prompt. Smoke overfit 8–16 training groups được phép để kiểm tra plumbing, không đưa thành kết quả PoC.

Trước khi train lớn, cho M-C có lambda=0 và B-S chạy cùng batch/RNG; loss/gradient phải khớp trong tolerance số học được ghi. Positive lambda phải có active pairs và gradient finite; nếu không, chưa triển khai đúng candidate.

## 7. Metrics và protocol thống kê

### 7.1. Correctness và missing records

`z_i^k=1` khi normalized answer đúng reference ở condition k. Primary V binary dùng constrained scoring để tránh lỗi parse. Knowledge short answers dùng mapping/vocabulary đã khóa từ train, normalize case/spacing và aliases có nguồn; không dùng LLM judge làm primary metric.

Ngoài tập vocabulary, answer không chuẩn hóa được hoặc timeout: lưu riêng. Với sample đã được đưa vào final denominator, missing prediction phải báo failed/missing; không tự loại để tăng điểm. Retry cùng config trước khi tổng hợp; nếu chưa đủ, báo incomplete coverage và không kết luận so sánh đủ cặp.

Clinical `insufficient_evidence` nếu có gold là semantic answer; `abstain` là model action; `invalid` là lỗi format. Không gộp chúng. PoC không có unknown gold mới thì không báo support-F1/clinical uncertainty như đã được đo.

### 7.2. Primary endpoints

1. `Acc_near(M)` trên cùng V test cố định dưới Knear; báo macro-F1 và per-class counts.
2. `Acc_0(M)` trên V no-K để xem giảm robustness có đổi lấy mất năng lực sạch không.
3. `KUG(M) = Acc_help(M | J) - Acc_0(M | J)` trên J annotation-selected cố định, kèm hai accuracy và counts được cứu/bị hại.

So M-C với B-S; B-P/B-F là đối chứng thực tế. Không chọn best baseline bằng final test rồi diễn giải như hypothesis được đăng ký trước; chọn main comparator ở P4 từ dev, vẫn công bố mọi baseline bắt buộc.

### 7.3. Transition và CFR phụ trợ

```text
CFR_k(M) = count(z0=1 and zk=0) / count(z0=1)
HarmfulWrongFlip_k = count(z0=1 and output_k is wrong substantive answer) / N_V
LostCorrect_k = count(z0=1 and zk=0) / N_V
PositiveFlip_help = count(z0=0 and zhelp=1) / N_J
```

`CFR` denominator=0 trả null/NA cùng counts, không trả 0. Denominator khác nhau giữa model; không xếp hạng chỉ bằng CFR. Tách wrong answer, abstain và invalid. Với forced binary, LostCorrect/HarmfulWrongFlip có thể trùng; ghi rõ protocol.

Ví dụ fixture 4 group: z0=[1,1,0,0], zk=[1,0,1,0] thì CFR=1/2, LostCorrect=1/4, accuracy cả hai=1/2. Đổi correctness của cùng group phải giữ pairing; average các rate không thay cho recompute pooled counts.

### 7.4. Dependency và generalization

Đo D-Q, D-K, no-K và ảnh thay thế có source label tương ứng. Khi đổi ảnh không mặc định giữ reference cũ; chỉ dùng QA/template compatible với nhãn nguồn và target được xác định lại. Nếu không có cặp hợp lệ, dùng no-image + same-pack/multi-image controls và ghi chưa kiểm tra image-swap.

J cần evidence hành vi rằng ảnh và K đóng vai trò; q+K giải được mọi câu hoặc q đã nêu target organ thì J không đạt joint claim. KUG=0 có thể do parametric knowledge/ceiling; không tự kết luận model ignore-K. Khelp chọn gold answer một cách oracle hoặc một dòng chứa sẵn đáp án không đủ làm utility proof.

Stress tối thiểu: cùng generic K trên nhiều ảnh, thay thứ tự pack, held-out rendering template, CXR slice nếu đủ. Không gọi held-out wording là external generalization.

### 7.5. CI và quy tắc diễn giải

Bootstrap paired theo image cluster: mỗi replicate lấy image IDs có hoàn lại, mang theo toàn bộ QA và context của ảnh đó, giữ cùng resample cho mọi model/condition. Dùng 2.000 replicates, seed cố định; báo CI 95% cho accuracy delta và KUG delta. Report cả số ảnh/group và replicate không xác định khi CFR denominator rỗng. Patient ID không có nên image CI vẫn có giới hạn.

Không bootstrap từng context variant độc lập. Nếu dùng nhiều seed, tách variability giữa seed khỏi CI theo test images. Không chạy hàng loạt significance tests ở mỗi slice nhỏ rồi chọn p-value có lợi.

P4 phải khóa **margin đề xuất của riêng PoC**, không gọi là chuẩn cộng đồng: clean accuracy không giảm quá 3 percentage points; Khelp accuracy không giảm quá 3 pp; KUG không giảm quá 3 pp. Có thể chỉnh dựa trên dev/ý nghĩa task trước final test, phải ghi lý do và version. `-3 pp` không có nghĩa `-3% relative`.

Để nói có bằng chứng method improvement: paired delta Acc_near so B-S có lower CI >0, các non-inferiority deltas có lower CI >-margin, và J/dependency controls hợp lệ. Nếu chỉ point estimates đạt nhưng CI rộng: **PROMISING/INCONCLUSIVE**, không PASS scientific superiority. Nếu không đủ J thì robustness-only result, chưa đạt mục tiêu kép.

No-K ceiling, fewer than roughly 20 meaningful no-K-correct opportunities, quá ít độc lập image clusters hoặc số errors quá nhỏ đều làm kết luận CFR kém ổn định. Không loại ảnh khó sau khi xem predictions; báo denominators và giới hạn.

## 8. Hợp đồng code và artifacts

```text
configs/                  # data, prompts, base, experiment configs
src/cka_med/
  data/                   # ingest, filter, group, contexts, audit
  inference/              # backbone adapter, prompt renderer, scoring
  training/               # sft, conditional objective, checkpoint/resume
  evaluation/             # correctness, transitions, clustered bootstrap
scripts/                  # runnable entry points, audit utility
tests/fixtures/           # synthetic cases only, marked as such
notebooks/                # Kaggle thin runner calling the same source
data/manifests/           # versioned IDs/hashes, no secret
data/processed/           # local derived data; governed by source terms
artifacts/runs/<run_id>/   # config, environment, logs, predictions, adapters
reports/phase_0 ... phase_7/
reports/reviews/           # Codex findings and decisions
```

Đây là contract cần Gemini hiện thực; ngoài audit script, các entry points dưới đây **chưa tồn tại** tại thời điểm lập plan:

```text
python -m cka_med.data.prepare --config configs/data.yaml
python -m cka_med.data.audit --manifest data/manifests/poc_v1.json
python -m cka_med.inference.run --config configs/experiments/baselines.yaml --split dev
python -m cka_med.training.sft --config configs/experiments/sft.yaml
python -m cka_med.training.conditional --config configs/experiments/conditional.yaml
python -m cka_med.evaluation.run --predictions <path> --protocol <path>
```

Gemini được đặt tên khác nếu có README mapping tương đương. CLI exit nonzero khi data invalid, leakage, thiếu checkpoint hoặc run thất bại. Không `except: pass`, sinh số ngẫu nhiên thay model, hoặc fallback từ ảnh sang text-only mà vẫn báo multimodal.

Mỗi run có `run_id`, source commit hoặc source SHA manifest, model revision, dataset/KG/prompt hashes, seed, GPU/dtype/dependency lock, LoRA config, max steps, context/pixel budget, train exposure counts, elapsed/quota, status và artifact checksums. Chưa có Git thì khởi tạo repo local ở P0/P1 hoặc cung cấp SHA manifest; không tạo remote/publish tự động.

Prediction record tách reference khỏi model input: `run_id, model_id, group_id, sample_id, image_id, condition, task_type, context_pack_id, oracle_context, raw_output, normalized_answer, label_scores, abstained, valid_output, error, latency`. Evaluator mới join gold theo sample_id. Writer phải ngăn duplicate key `(run_id,sample_id)`.

Data record có source dataset/version/qid, image path/hash, original q/answer, normalized answer, task eligibility/rationale, split, context source entries/rendering version, gold-invariant flag, provenance quality, reviewer type, exclusion reason. `patient_id` và `premise_label` cho phép null có nghĩa rõ ràng. Không phát minh reviewer IDs.

Notebook là runner, không phải nơi duy nhất chứa business logic. CPU audit và recompute metrics phải chạy ngoài Kaggle. Xuất logs/artifacts xuống local sau mỗi job; không để kết quả chỉ tồn tại trong session tạm.

## 9. Chi tiết từng phase và tiêu chí nghiệm thu

Chi tiết ở các mục tiếp theo. Mọi phase dùng cùng [template bàn giao/nghiệm thu](PHASE_REVIEW_TEMPLATE.md); reviewer phân biệt engineering status và scientific verdict.

### P0 — Scope, nguồn và access

**Input:** toàn bộ tài liệu hiện có, các xác nhận người dùng và source audit đã làm.

**Gemini thực hiện:**

1. Đọc plan/analysis/access; tạo `decision_log.md`, `scope_contract.md`, `resource_access.md`. Chốt pilot English radiology, CXR là slice và clinical support labels ngoài scope tuần này.
2. Access probe SLAKE images + Qwen; không tải model lớn lên local nếu chỉ cần chạy trên Kaggle. Nếu chọn MedGemma, ghi bằng chứng access đã có; không ký thay người dùng.
3. Kiểm tra quota Kaggle hiện tại, số dư Modal nếu dùng, chọn runner và budget trần. Không khởi chạy training.
4. Audit tập trung prior art: MedKInstruct, DEPO, CKRA, MR-RAG, CAER, mR²AG; bảng có URL/version, phần đã đọc, overlap, code/access và điều chưa biết. Timebox tối đa khoảng 2 giờ; không chờ đủ mọi full text mới viết code CPU, nhưng không được tuyên bố novelty chưa audit.
5. Ghi `protocol_draft.md`: task, allowed inputs, context modes, baseline IDs, candidate duy nhất, outcome khi utility không đo được.

**Deliverables:** `reports/phase_0/{SUBMISSION.md,scope_contract.md,decision_log.md,resource_access.md,compute_budget.json,prior_art_matrix.csv,protocol_draft.md}`; probe log đã che secrets; source hashes.

**Codex nghiệm thu:** đối chiếu scope với người dùng; kiểm tra link/revision và các claim mạnh; xem input contract có cấm gold metadata/report/role ở inference; kiểm tra miễn phí thực tế. **PASS** nếu có đường data+model+GPU khả thi hoặc fallback hợp lệ, không để dependency hạn chế chưa giải quyết làm critical path. Nếu không có ảnh/model access thì REVISE/NEEDS_INPUT, vẫn có thể làm P1 CPU theo chỉ định reviewer.

### P1 — Repo, schema, evaluator và runner skeleton

**Input:** P0 đã duyệt hoặc phần CPU được cho phép.

**Gemini thực hiện:** package tối thiểu, pinned environment, CLI, JSON schema, source manifest, prompt renderer, prediction writer, evaluator và bootstrap; synthetic fixtures 8–12 group. Kaggle notebook gọi code chung, config secrets qua môi trường; chưa cần GPU training.

**Fixtures phải kiểm tra:** all-correct/all-wrong; no-K-correct denominator=0; đúng→sai và sai→đúng; semantic unknown khác abstain; missing variant; duplicate IDs; nhiều QA cùng image; role/gold field bị cấm trong prompt; CSV/JSON roundtrip; test cho image-cluster resampling.

**Deliverables:** source, dependency lock, `reports/phase_1/{SUBMISSION.md,commands.txt,test_results.txt,fixture_expected.json,fixture_actual.json}`, CLI help, example run manifest. Fixture phải có `synthetic_fixture=true`, không trộn final data.

**Codex nghiệm thu:** chạy lại CPU checks, tính tay transition fixture, đọc cách cluster/missingness/join, thử thêm một record lỗi. **PASS** khi output đúng với fixture độc lập, failure exit đúng, và provenance/input isolation rõ. Green test tự viết nhưng sai metric vẫn REVISE.

### P2 — Development data và knowledge packs

**Input:** ingest/evaluator đã duyệt; official train/validation/test metadata và ảnh tải hợp lệ.

**Gemini thực hiện:**

1. Tải ảnh, kiểm tra đường dẫn, hash/dedup; không load `mask`/segmentation làm ảnh input mặc định. Giữ source split và IDs.
2. Tạo development V/J từ validation theo rule; record source qid/answer, exclusions, quality tier. Không sửa bệnh/absence gold.
3. Lọc KG 30–80 entries đủ phục vụ task; ưu tiên anatomy/body-system/function; giữ source row/raw relation. Không sinh KG mới từ trí nhớ LLM.
4. Tạo packs có nhiều candidates cho joint QA và Knear/Kirrel; đóng băng generator version. Dùng q/family và corpus được phép, không gold image label để retrieve main context.
5. Lập audit 30–50 development examples phủ V/J, positive/negative, Knear/Khelp; giải thích tại sao label còn đúng và K không tự tiết lộ đáp án. Phân biệt `source_asserted` với verified truth.

**Deliverables:** `reports/phase_2/{SUBMISSION.md,data_card.md,eligibility_audit.csv,context_audit.csv,split_audit.json,exclusions.csv}`, dev JSONL, KG subset/provenance, generator config và hash manifest. Source test chỉ kiểm tra integrity; chưa chạy model trên final test.

**Codex nghiệm thu:** trace ngẫu nhiên tối thiểu 10 V và 10 J candidate nếu đủ; xem 5 Knear khó; kiểm tra image input không chứa mask/gold overlay, q có lộ organ không, Khelp có là answer-only không, split và gold invariance. Codex chỉ xác nhận logic/provenance, không chứng nhận diễn giải ảnh y khoa. **PASS** kỹ thuật khi trace được và không rò gold; J bị loại vẫn có thể PASS với scientific warning/thu hẹp claim.

### P3 — Competence, baseline, dependency và compute gate

**Input:** development groups đã duyệt. Chỉ dùng development để chọn model/prompt/config.

**Gemini thực hiện:** B-N/B-R/B-P/B-F và D-Q/D-K; ghi paired transitions theo K; kiểm tra ảnh thật sự vào processor/model; so majority baseline trên V, no-image và các case nguyên nhân lỗi. Giới hạn model probes: backbone mặc định và tối đa một alternative khi access có sẵn/competence quá thấp.

Đo VRAM/tốc độ như mục 5.2; lập dự báo number-of-variant-records × models × forward branches × seeds, không dùng số group đơn thuần. Cài resume/checkpoint trước run dài. Báo những trường hợp prompt raw làm sai vì format/token truncation để không lầm thành knowledge conflict.

**Deliverables:** `reports/phase_3/{SUBMISSION.md,baseline_summary.csv,transitions.csv,dependency_report.md,profile.json,cost_forecast.json,train_gate.md}`; full dev predictions/configs; casebook có cả successes/failures ngẫu nhiên theo strata.

**Codex ra quyết định:**

- **TRAIN:** input/labels hợp lệ, model có evidence dùng ảnh, có cả vấn đề robustness đáng nghiên cứu và tập J có thể đo, baseline rẻ chưa giải thích hết; đủ compute cho SFT+candidate+eval.
- **ANALYSIS_ONLY:** task/pipeline hợp lệ nhưng no-K ceiling, không thấy failure mode hoặc K không có utility đáng đo; không cố train để có tên phương pháp.
- **REVISE_TASK:** q+K đủ trả lời phần lớn J, Knear không đúng theo nguồn, lỗi chủ yếu visual/format, hoặc thiếu điều kiện so sánh.

Không đặt mục tiêu phải tạo đủ wrong flips bằng một câu patient-specific sai rồi gọi đó là tri thức đúng. Nếu development còn ít cơ hội no-K-correct, chỉ báo mô tả; không tự dựng threshold thành clinical standard. Codex có thể cho phép nhánh robustness-only training thử, nhưng phải ghi rằng mục tiêu kép chưa được kiểm chứng.

### P4 — Khóa data và protocol trước training/final evaluation

**Input:** quyết định P3. Nếu ANALYSIS_ONLY, vẫn khóa protocol cho final analysis rồi bỏ training có lý do.

**Gemini thực hiện:** chọn train/dev/test subsets bằng rules/seed đã công bố, export source references và audit coverage; khóa prompt/context generators, answer vocabulary/aliases, task eligibility, pixel/token budget, baseline comparator, lambda candidates, max epochs/steps và checkpoint-selection rule. Chọn main comparator từ development, không final test.

Đối với final test, generator rule đã cố định trước khi join reference để audit. Human/agent exclusion do source lỗi phải dựa rule và không nhìn predictions; lưu rõ vì sao loại. Không cố match test answer bằng cách thêm option chỉ để chứa gold.

`protocol_v1.yaml` phải ghi endpoints, J definition, image clustering, NA/missing handling, confidence/abstention policy nếu có, margins, seeds, bootstrap seed/replicates, mọi run dự định và quy tắc dừng. Dùng argmax primary; risk-coverage chỉ exploratory nếu thêm confidence score, không dùng model tự nói “90%” làm calibration.

**Deliverables:** `reports/phase_4/{SUBMISSION.md,protocol_v1.yaml,protocol_v1.md,dataset_manifest.json,freeze_manifest.json,run_matrix.csv,selection_rules.md}`; sha256 của data/KG/prompts/code/configs. `freeze_manifest` ghi rõ chưa có final predictions.

**Codex nghiệm thu:** dò split leakage, reference-only fields, main-vs-oracle contexts, eligibility J, denominator/sample budget, compute và margin. **PASS** khi hypothesis/cách chọn model có thể kiểm chứng và artifacts đã khóa. Nếu sample không đủ, PASS với mức claim exploratory; không gọi là study powered cho vài pp.

### P5 — Same-data SFT và conditional candidate

**Input:** P4 đã duyệt, quota còn đủ. Nếu P3 quyết định ANALYSIS_ONLY, hồ sơ phase này ghi `NOT_RUN_BY_DESIGN` và chuyển P6 theo protocol; không làm checkpoint giả.

**Gemini thực hiện:** smoke overfit/gradient/mask check; B-S seed42; M-C seed42 cùng exposure, validation-only selection. Lambda tối đa hai lựa chọn nonzero, dừng khi hết timebox/budget; không tùy tiện thêm gate/head/loss.

Log QA NLL, JS, active mask count, learning rate, grad norm, optimizer steps, sample IDs/exposures, VRAM và wall time. Cặp no-K/Knear giữ cùng I/q/Y map; không pairing theo batch index giữa QA khác nhau. Giữ phiên bản package và base checkpoint giống nhau.

Chọn checkpoint theo rule P4, ví dụ tối ưu dev Acc_near dưới clean/help/utility constraints; nếu không checkpoint nào đáp ứng constraint thì ghi không đạt, không nới rule bằng test. Tuning dùng dev nên không dùng dev scores làm final superiority proof.

**Deliverables:** adapters, tokenizer/processor config, optimizer/RNG checkpoint nếu resume; `reports/phase_5/{SUBMISSION.md,train_summary.csv,dev_results.csv,fairness_audit.json,gradient_checks.txt,selection_decision.md}`, full logs và run manifests.

**Codex nghiệm thu:** đọc loss code, mask/sample pairing, verify lambda=0, kiểm tra active gradient và train exposures, reload adapter kiểm tra model output thay đổi; recompute dev metrics. **PASS** implementation nếu chạy đúng và fair, dù không cải thiện. Sai objective/leakage/không đủ bằng chứng run thì REVISE, không hợp thức hóa bằng loss giảm.

### P6 — Final evaluation và ablation

**Input:** source/model/protocol frozen và P5 đã duyệt hoặc NOT_RUN_BY_DESIGN.

**Gemini thực hiện:** chạy cùng final manifest cho B-N/B-R/B-P/B-F và B-S/M-C nếu có; D-Q/D-K trên J; stress rendering/pack order đã đăng ký. Nếu đủ quota, A-U hoặc seed43 cho cả B-S/M-C theo ưu tiên đã khóa; không bỏ final main test để thêm ablation.

Chỉ mở final evaluation sau khi đã chọn model. Nếu phát hiện bug, lưu bản cũ, mô tả ảnh hưởng; sửa chung và chạy lại mọi baseline bị ảnh hưởng. Không tune mới sau xem test mà vẫn gọi independent confirmation. Rerun một sample vì timeout cùng config là retry, không là model selection.

Tổng hợp clustered bootstrap, transitions, per-task/role/source label quality, CXR slice, no-K ceiling, missingness, GPU cost. Chọn error examples theo strata và seed; đưa cả model thắng/thua/không đổi và các trường hợp reference đáng nghi.

**Deliverables:** `reports/phase_6/{SUBMISSION.md,main_results.csv,paired_deltas.csv,confidence_intervals.json,transitions.csv,stress_results.csv,error_inventory.csv,compute_actual.json,final_access_log.md}`, raw prediction JSONL và figures xuất từ script.

**Codex nghiệm thu:** recompute từ raw, verify sample matching/denominators/no dropped errors, kiểm tra clustering bằng vài resamples; đối chiếu source/config hashes với P4/P5; audit casebook không cherry-pick. Engineering PASS không phụ thuộc dấu của delta. Scientific verdict có thể SUPPORT/REFUTE/INCONCLUSIVE.

### P7 — Tái lập, báo cáo và quyết định khóa luận

**Input:** P6 đã audit và toàn bộ run registry.

**Gemini thực hiện:** môi trường sạch tái chạy CPU audit + metrics, reload model và inference subset 10–20 samples nếu GPU còn; không cần full retrain. So checksum, score tolerance và known nondeterminism. Xuất bundle có README command, paths, manifest, dependency lock, adapter loading, predictions, code, source attribution và limitations.

Báo cáo 6–10 trang hoặc Markdown tương đương: vấn đề/phạm vi, data/access/quality, task differences so proposal, methods và fairness, raw/paired results+CI, dependency controls, cost, failures/negative results, limitations và recommendation. Mọi kết luận có bảng/run ID hỗ trợ; không đặt expected outcome thành observed result.

**Deliverables:** `reports/phase_7/{SUBMISSION.md,POC_REPORT.md,REPRODUCE.md,CLAIM_EVIDENCE.csv,FINAL_DECISION.md,limitations.md}`, bundle checksum/index, backlog khóa luận có ưu tiên.

**Codex kết luận cuối:**

| Kết luận | Điều kiện | Hướng tiếp theo |
|---|---|---|
| GO — method pilot | Task/J hợp lệ, robustness cải thiện và utility/clean giữ theo protocol, evidence đủ với giới hạn pilot | Mở rộng nhãn chuyên môn, CXR, prior-art reproduction và nhiều seed |
| GO — benchmark/analysis | Pipeline tốt, failure mode/insight đáng tin nhưng method không hơn SFT/filter | Bỏ method superiority claim, phát triển benchmark/phân tích có review |
| INCONCLUSIVE | CI rộng, ít J, ceiling/quota khiến thiếu evidence | Đề xuất đúng thí nghiệm cần bổ sung; không tự báo thành công/thất bại thuật toán |
| PIVOT | Joint task không hợp lệ, perception bottleneck hoặc inexpensive baseline đã giải quyết | Thu hẹp sang visual competence/evidence evaluation hoặc câu hỏi khác |

Không lấy demo đẹp, training loss giảm hay screenshot GPU làm điều kiện đủ hoàn thành PoC.

## 10. Xử lý sự cố và điều kiện phải hỏi ngay

| Vấn đề | Hành động trong scope | Khi nào báo người dùng/Codex ngay |
|---|---|---|
| HF access bị chặn | Dùng model/path fallback đã duyệt, lưu probe | Cả default và fallback đều không tải được |
| GPU không sẵn/oom | Giảm batch, bound pixels hợp lý, checkpoint, thử runner miễn phí khác trong budget | Cần chi trả, không giữ được competence hoặc phải bỏ cặp so sánh chính |
| Thiếu J hợp lệ | Báo audit counts và utility inconclusive; có thể analysis-only | Muốn thay task/dataset hoặc tự tạo clinical labels mới |
| Không có wrong flips với Knear | Kiểm tra format/pack/ceiling rồi ghi negative result | Muốn thay bằng false assertion để giữ claim true-knowledge conflict |
| Source label đáng nghi | Exclude bằng rule/log, không sửa theo model | Ảnh hưởng đáng kể scope hoặc cần người chuyên môn |
| Test đã bị dùng tuning | Version audit và hạ claim; giữ lịch sử | Cần test mới hoặc kéo dài deadline |
| SFT ngang/hơn M-C | Giữ negative result và so cost | Muốn đổi research objective/đánh giá sau xem test |
| Không đủ quota final eval | Bỏ extension, ưu tiên mandatory predictions | Không thể hoàn thành evaluation đã đăng ký trước 23/11 |

Các routine implementation choices không cần hỏi lại. Quyết định thay scope, cần chuyên môn không có, phát sinh tiền hoặc ảnh hưởng tính độc lập final test phải được nêu ngay với bằng chứng và phương án cụ thể. Không chờ đến báo cáo cuối mới nói task không hợp lệ.

## 11. Những việc chuyển sang sau PoC

Theo kết quả P7 mới quyết định: xin VinDr-CXR cho nhãn chuyên gia; bổ sung bác sĩ/review rubric cho premise support; xây J thực sự khó nhưng không dựa giả định chẩn đoán; train support-aware mechanism; tái lập đối thủ gần; cỡ test theo variance/power analysis; external CXR/source/rule holdouts; backbone/seed thứ hai; tiếng Việt/ViX-Ray.

Chưa có quyền và chuyên môn thì không khẳng định phần gốc “tri thức y khoa thật nhưng không áp dụng cho bệnh nhân” đã được kiểm chứng đầy đủ. Đầu ra tuần này phải làm rõ phần nào đã có evidence, phần nào còn thiếu và thiếu vì sao.
