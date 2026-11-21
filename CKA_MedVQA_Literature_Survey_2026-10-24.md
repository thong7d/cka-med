# Survey và kiểm định hướng khóa luận CKA-MedVQA

**Ngày tra cứu:** 24/10/2026. **Đối tượng:** Conditional Knowledge Alignment for Evidence-Grounded Medical Visual Question Answering. **Trạng thái:** khảo sát tài liệu và thiết kế nghiên cứu; chưa thực nghiệm.

## 1. Kết luận để ra quyết định

**Nên tiếp tục hướng nghiên cứu, nhưng phải thu hẹp tuyên bố đóng góp và đặt một giai đoạn kiểm chứng trước khi chốt phương pháp.** Bản đề xuất đã nhận diện đúng nguy cơ bỏ qua tri thức và thiếu grounding; tuy nhiên, danh sách related work ban đầu chưa đủ để xác nhận tính mới.

Ba kết luận chính:

1. **Không mới ở cấp ý tưởng chung:** chọn lọc tri thức; giữ lợi ích của retrieval đồng thời chống nhiễu; dùng gate; huấn luyện với hard negatives; tăng visual grounding; tạo benchmark xung đột. Các nhánh này đã có công trình tương ứng [W01–W09, W12–W15, P05, P14–P17].
2. **Khoảng trống ứng viên đáng kiểm chứng:** đánh giá và học cách sử dụng một mệnh đề y khoa đúng nhưng chỉ được phép áp dụng khi tiền đề cần thiết được bằng chứng của ca hỗ trợ; phân biệt *tiền đề bị phủ định* và *chưa xác định được tiền đề*; đồng thời đo lợi ích của tri thức hữu ích trên câu hỏi thực sự dùng cả ảnh và tri thức.
3. **Chưa thể xác nhận “chưa bất kỳ ai làm”.** Một số đối thủ mới tìm thấy mới đọc được abstract hoặc metadata, chưa truy cập được toàn văn. Không tìm thấy bài trùng tên không chứng minh tính mới. Không có cơ sở gán xác suất 90% hay 95% cho novelty.

**Tên làm việc đề xuất:** *Evidence-Conditioned Knowledge Use in Chest X-ray VQA: Applicability, Robustness, and Utility*. Tên này mô tả đối tượng kiểm nghiệm rõ hơn “alignment” chung chung; không tuyên bố tên chưa được sử dụng.

**Đóng góp chính nên là giao thức đánh giá và tập dữ liệu có kiểm định; đóng góp phương pháp chỉ được công nhận nếu vượt baseline mạnh trên cùng dữ liệu.** Không cam kết SOTA, chất lượng lâm sàng hoặc khả năng được nhận tại một hội nghị cụ thể.

## 2. Phạm vi và mức độ đầy đủ của survey

### 2.1. Những gì đã thực hiện

- Đọc bản proposal được cung cấp; trích xuất 17 PDF trong bộ tài liệu dự án để sàng lọc abstract, vấn đề và phạm vi. Đọc sâu hơn phần phương pháp ở các tài liệu trọng tâm, đặc biệt DEPO; đối chiếu MedKInstruct, MMCAP và các hướng grounding.
- Tìm kiếm qua hai công cụ tìm web; ưu tiên nguồn tác giả, arXiv, ACL Anthology, CVF, IJCAI, MICCAI, PubMed và trang dataset/model chính thức.
- Mở và kiểm tra nguồn chính cho các đối thủ quan trọng. Một số trang trả lỗi truy cập; các trường hợp đó được đánh dấu riêng, không suy diễn nội dung phương pháp.
- Lập danh mục **46 công trình/benchmark nghiên cứu**, gồm toàn bộ 17 PDF và 29 mục bổ sung, theo mức liên quan tới tính mới. Danh mục này không tính model card và trang dữ liệu bổ sung là paper mới.

Đây là **focused scoping survey và novelty audit**, không phải systematic review đã đăng ký protocol hoặc một bản kiểm kê toàn bộ y văn. Chưa xuất kết quả truy vấn Scopus/Web of Science/PubMed theo một giao thức có tổng số bản ghi, chưa sàng lọc độc lập bởi hai người, chưa hoàn thành forward/backward citation snowballing của mọi bài. Do đó không dựng số liệu PRISMA hoặc nói đã tìm hết các công trình đến ngày tra cứu.

### 2.2. Quy tắc đưa vào và xếp mức liên quan

Đưa vào nếu công trình nghiên cứu ít nhất một trong các thành phần: MedVQA có tri thức ngoài; xử lý xung đột ảnh–văn bản/tri thức; chọn lọc thông tin theo ngữ cảnh; robustness–utility của RAG; grounding/hallucination; benchmark và dữ liệu phục vụ kiểm chứng.

Ưu tiên theo thứ tự: **trùng cơ chế cần tuyên bố mới → trùng giao thức đánh giá → trùng miền MedVQA → nguồn dữ liệu/backbone hỗ trợ**. Thứ hạng là đánh giá phục vụ khóa luận, không phải thứ hạng chất lượng paper. Trong cùng tầng, thứ tự có thể thay đổi sau khi đọc toàn văn.

Các paper text-only được giữ vì có thể phủ định tính mới của objective hoặc cách tạo dữ liệu, dù không giải quyết toàn bộ bài toán ảnh y khoa. Các bài federated learning, multilingual và fusion được đặt thấp hơn vì khác failure mode.

### 2.3. Nhật ký truy vấn cốt lõi

Các truy vấn dưới đây đã được dùng trong phiên tra cứu; kết quả chủ yếu là nguồn tiếng Anh:

| Nhóm | Truy vấn tiêu biểu |
|---|---|
| MedVQA + conflict | `"medical visual question answering" "knowledge" "conflict"`; `"medical VQA" "true" "irrelevant" knowledge` |
| Chọn lọc tri thức | `"medical" "selective knowledge" "Fusion"`; `medical visual question answering knowledge conflict selective retrieval robustness irrelevant knowledge` |
| Khoảng trống hẹp | `"MedVQA" "applicability"`; `"medical" "knowledge applicability" OR "non-applicable knowledge" OR "inapplicable knowledge"`; `"medical visual question answering" "conditional knowledge"` |
| Multimodal RAG | `multimodal retrieval augmented generation knowledge conflict visual evidence selective reliance`; `"multimodal" "knowledge conflict" benchmark mitigation` |
| Truy đối thủ | Tìm chính xác tiêu đề DEPO, AMiF, CAER, MR-RAG, MedEvidence-RAG, CKRA, MedRGB, mR²AG |
| Grounding | `"GEMeX" medical visual question answering grounding`; `"medical visual question answering" "abstention"` |
| Tiền lệ text-RAG | Tìm chính xác Self-RAG, RAFT, Making Retrieval-Augmented Language Models Robust to Irrelevant Context |
| Tài nguyên | ViX-Ray; SLAKE; VinDr-CXR access; MedGemma official model card |

Truy vấn exact phrase có thể bỏ sót paper dùng tên khác. Kết quả rỗng hoặc không liên quan ở truy vấn “conditional knowledge” không phải bằng chứng phủ định prior art.

## 3. Danh mục xếp theo mức liên quan

**Mức đọc:** `M` = đã kiểm tra phần phương pháp liên quan trong toàn văn; `A` = abstract/giới thiệu chính thức hoặc PDF được cung cấp đủ để xác định phạm vi, chưa audit toàn bộ thí nghiệm; `L` = metadata/trích đoạn nguồn chính, chưa đủ để kết luận về phần thiếu. Mọi nhận định “còn khác” dưới đây là so sánh trong phần đã đọc, không mặc định paper không có nội dung đó ở phụ lục.

### Tầng I — đối thủ cần đọc trước khi chốt novelty

| Hạng | Công trình | Đã làm gì và kết quả liên quan | Hệ quả cho khóa luận | Nguồn; mức đọc |
|---:|---|---|---|---|
| 1 | CKRA — Beyond Static Knowledge | Lọc tri thức theo ngữ cảnh mẫu; knowledge–image contrastive learning và guided attention; báo cáo cải thiện trên SLAKE/VQA-RAD | “Chọn tri thức theo ảnh/câu hỏi” không đủ mới. Phải so trực tiếp relevance với premise support | W01; A |
| 2 | MR-RAG | Medical VQA với multimodal relevance-aware RAG; công trình CVPR 2026 đã xác minh | Đối thủ bắt buộc cho selective retrieval/generation; chưa đủ toàn văn để kết luận không xử lý applicability | W02; L |
| 3 | DEPO — Beyond Surface Features | Perturb câu hỏi/ảnh và resample đáp án không được ưu tiên để căn chỉnh bằng chứng | DPO, negative động và chống text-over-image không mới; đầu vào phương pháp mô tả là ảnh–query, không mặc định là KG truy xuất riêng | P14/W03; M |
| 4 | CAER | Router gắn claim với visual evidence, hai prefix experts cho supported/contradicted; backbone đóng băng | Gate theo trạng thái xung đột đã có. Phải làm rõ vai trò unknown và tri thức tổng quát đúng | W04; M |
| 5 | mR²AG | Quyết định khi nào retrieval cần thiết và định vị evidence hữu ích; giữ năng lực visual-dependent tasks | “Dùng kiến thức khi cần” đã có trong multimodal VQA | W05; A |
| 6 | AMiF — Alignment, Mining and Fusion | Alignment đa mức, hard-negative mining, gated fusion từ answer vocabulary | Gate/hard negatives không mới; không đồng nhất answer-vocabulary prior với tài liệu y khoa ngoại sinh | W06; M |
| 7 | CMC-Bench — When Image and Text Disagree | Can thiệp evidence theo image-correct/text-correct/both-wrong/aligned; tách abstention và fabrication | Benchmark xung đột đa phương thức có trước; cần một khác biệt ngữ nghĩa vượt “thay sang ảnh y khoa” | W07; A |
| 8 | Making Retrieval-Augmented Language Models Robust to Irrelevant Context | NLI filtering và SFT trộn relevant/irrelevant context; xét đồng thời lợi ích và tác hại retrieval | Phủ định novelty của mục tiêu robustness–utility tổng quát | W08; A |
| 9 | MedKInstruct | MMKG-guided instruction, multi-hop/hard negatives, path reward cho RL | KG + QA tổng hợp + hard negatives không đủ mới | P05; A |
| 10 | MedRGB | Đánh giá medical text-RAG về sufficiency, integration và robustness trước nhiễu/thông tin sai | Đánh giá sufficiency/reliability không mới; cần bổ sung bằng chứng ảnh và điều kiện áp dụng | W09; A |
| 11 | Fine-grained knowledge fusion for retrieval-augmented medical VQA | Nghiên cứu retrieval và knowledge fusion trong đúng miền | Cần lấy toàn văn trước khi chốt đóng góp fusion | W10; L |
| 12 | MedEvidence-RAG | Công trình về joint evidence retrieval và generator alignment cho VQA/report generation | Tên và metadata cho thấy overlap; chưa đủ dữ liệu để đánh giá chi tiết hoặc loại trừ trùng lặp | W11; L |

### Tầng II — tiền lệ cơ chế, robustness và grounding

| Hạng | Công trình | Đóng góp liên quan | Cách dùng trong khóa luận | Nguồn; mức đọc |
|---:|---|---|---|---|
| 13 | KCM — Enhancing Retrieval-Augmented LVLMs via Knowledge Conflict Mitigation | Điều chỉnh attention, neuron và context logits để giảm xung đột context–parametric memory | Baseline conflict mitigation; khác với việc mặc định tri thức truy xuất không đủ điều kiện áp dụng | W12; A |
| 14 | Utility-Oriented Visual Evidence Selection | Dùng information gain/latent helpfulness để chọn visual evidence | Phân biệt relevance và utility đã có; không tuyên bố đó là phát hiện đầu tiên | W13; A |
| 15 | Self-RAG | Adaptive retrieval và reflection về evidence/generation | Tiền lệ cho retrieval decision, relevance, support | W14; A |
| 16 | RAFT | Huấn luyện sử dụng tài liệu liên quan và bỏ qua distractors | Baseline SFT theo dữ liệu can thiệp; không coi là thuật toán ảnh y khoa nguyên bản | W15; A |
| 17 | AMANDA | Question decomposition kết hợp biomedical KG retrieval trong framework không cần train | Baseline knowledge augmentation nếu chi phí inference cho phép | W16; A |
| 18 | MMKC-Bench | Factual conflicts dạng context-memory và inter-context, nhiều loại multimodal conflict | Tiền lệ taxonomy và paired conflict evaluation | W17; A |
| 19 | Insight Over Sight | Vision–commonsense conflict, benchmark và Focus-on-Vision prompting | Cần baseline prompt ưu tiên ảnh; parametric prior khác external K | W18; A |
| 20 | Unraveling Cross-Modality Knowledge Conflicts | Phân tích xung đột parametric giữa modality và dynamic contrastive decoding | Baseline decoding nếu có logits; không phải cùng nguồn xung đột với KG | W19; A |
| 21 | Same Semantics, Different Outcome | Xét cùng evidence dưới text/rendered-image và thứ tự đưa vào | Cần kiểm tra vị trí, modality, format của evidence | W20; A |
| 22 | Diagnosing Knowledge Conflict in Multimodal Long-Chain Reasoning | Phân biệt conflict ở input/process và khảo sát representation | Đọc nếu mở rộng mechanistic analysis; không cần cho MVP | W21; A |
| 23 | Knowing or Guessing? — CCL/RoMed | Consistency/contrastive learning cho paraphrase robustness trong MedVQA | Cần holdout template và paraphrase; consistency loss không mới | W22; A |
| 24 | HEAL-MedVQA / Localize-before-Answer | Grounding benchmark và localize-before-answer với segmentation annotations | Nguồn thiết kế kiểm tra visual shortcut, baseline grounding | W23; A |
| 25 | MedVH | Benchmark hallucination y khoa với nhiều loại input và task | Bổ sung irrelevant/misleading input và over-refusal controls | W24; A |
| 26 | ProbMed — Worse than Random? | Câu hỏi đối kháng và probing chẩn đoán nhiều chiều | Không thể suy độ tin cậy từ accuracy gộp | P12; A |
| 27 | Act as You Think / MCR | Consistency giữa perception, reasoning, answer trong RLVR | Lời giải hợp lý và đáp án đúng phải được kiểm tra riêng | P01/W25; A |
| 28 | A3Tune | Weak visual labels từ SAM/BiomedCLIP để căn chỉnh attention | ROI/attention tuning không phải novelty riêng của CKA | P17; A |
| 29 | MedVP | Explicit visual prompting và fine-tuning hướng vùng quan tâm | Baseline hỗ trợ grounding; kiểm tra chi phí/lỗi localizer | P16; A |
| 30 | MMCAP | Medical KG, concept alignment và type-conditional prompt | “Conditional theo loại câu hỏi” đã có tiền lệ trong MedVQA | P15; A |
| 31 | GEMeX-RMCoT, còn gọi GEMeX-ThinkVG | Region-grounded reasoning dataset và verifiable reward | Không đếm hai tên là hai paper; tham khảo đánh giá reasoning-grounding | W26; A |
| 32 | MedThink | Rationale-augmented R-RAD/R-SLAKE/R-Path và rationale-guided training | Rationale generation không tự chứng minh faithfulness | P06; A |
| 33 | STLLaVA-Med | Self-training visual instructions, DPO với giám sát từ model mạnh hơn | Synthetic data + preference learning đã có; phải kiểm tra label noise | P10; A |
| 34 | CAF-RAG | Question-type analysis và clinical filter cho text-only medical RAG | Dù dùng câu hỏi từ MedVQA, nghiên cứu image-free; không gọi đây là visual grounding baseline trực tiếp | W27; L |

### Tầng III — dữ liệu, transfer và nền tảng

| Hạng | Công trình | Vai trò và giới hạn đối với đề tài | Nguồn; mức đọc |
|---:|---|---|---|
| 35 | SLAKE | Có knowledge-based QA, KG và visual annotations; cần audit lại câu hỏi thật sự dùng ảnh | W28; A |
| 36 | GEMeX | CXR QA có visual/textual explanations; không có sẵn nhãn applicability theo thiết kế đề xuất | W29; A |
| 37 | ViX-Ray | CXR và reports tiếng Việt; thích hợp extension ngôn ngữ, chưa phải benchmark conflict | W30; A |
| 38 | Med-MAT | Generalization qua modality × anatomy × task | P02; A |
| 39 | MasonNLP at MEDIQA-WV 2025 | Multimodal exemplar RAG cho wound-care; baseline retrieval thực dụng | P04; A |
| 40 | MIRAGE/MedRAG | Benchmark/toolkit text medical RAG; giúp thiết kế retrieval baselines | W31; A |
| 41 | Fusion of Domain-Adapted Vision and Language Models | Fusion backbone domain-specific và parameter-efficient adaptation | P03; A |
| 42 | PubMedCLIP | Domain adaptation của image–text encoder cho MedVQA | P09; A |
| 43 | OmniMedVQA | Benchmark nhiều modality; hữu ích kiểm tra transfer sau khi có core CXR | P07; A |
| 44 | WorldMedQA-V | Multilingual medical examination VQA; scope rộng hơn CXR findings | P11; A |
| 45 | PathVQA | Benchmark pathology VQA; dùng nếu mở rộng ngoài CXR | P08; A |
| 46 | X-FLoRA | Cross-modal federated learning với modality-expert LoRA; khác failure mode của đề tài | P13; A |

PathVQA ưu tiên trước X-FLoRA cho extension ngoài CXR. X-FLoRA không được chọn làm trụ cột vì federated learning không giải quyết trực tiếp vấn đề sử dụng tri thức có điều kiện.

## 4. Những claim của proposal phải sửa

| Claim có thể bị phản biện | Kết luận survey | Cách viết phù hợp hơn |
|---|---|---|
| “Lần đầu kết hợp KG và MedVQA” | Không đứng vững: SLAKE, MMCAP, MedKInstruct, AMANDA | Nêu KG là nguồn evidence có kiểm soát |
| “Lần đầu chỉ dùng knowledge khi cần” | Không đứng vững: Self-RAG, mR²AG, CKRA | Nêu rõ tiền đề nào phải được ảnh hỗ trợ và kiểm chứng nó |
| “Lần đầu giảm hại mà giữ lợi ích RAG” | Không đứng vững: Yoran và cộng sự, RAFT | Nghiên cứu failure mode hẹp và trade-off trên domain cụ thể |
| “Gate theo image–knowledge conflict là mới” | CAER và selective fusion làm claim này quá rộng | Nếu dùng gate, xem là implementation; chứng minh supervision/unknown đem lại khác biệt |
| “Đã có conflict benchmark nhưng chưa có paired tests” | CMC-Bench và CAER đã có cấu trúc đánh giá liên quan | Tập trung đúng-y-khoa nhưng chưa đủ tiền đề, không chỉ false claim |
| “DEPO là đối thủ gần nhất duy nhất” | Thiếu CKRA, MR-RAG, CAER, mR²AG, AMiF | Phân nhóm đối thủ theo objective, kiến trúc và evaluation |
| “CFR/KUG là metric mới được cộng đồng công nhận” | Không có căn cứ trong tài liệu hiện tại | Gọi là operational metrics của proposal, công khai công thức và liên hệ với transition/gain metrics |
| “CE + utility CE tạo một thuật toán mới” | Hai số hạng có thể chỉ tương đương weighted SFT | Phải có same-data, same-budget SFT ablation để xác định phần đóng góp thực |

**Điểm phân biệt quan trọng:** AMiF dùng answer-vocabulary prior; DEPO perturb query/ảnh; CAER xét claim/presupposition; MMKC xét factual conflicts. Không gom tất cả thành “KG gây nhiễu”. Đó là các loại evidence và can thiệp khác nhau. Ngược lại, khác nguồn evidence không tự động đủ tạo một đóng góp học thuật mạnh.

## 5. Khoảng trống ứng viên và phép thử để bác bỏ

### 5.1. Định nghĩa đối tượng thật sự cần nghiên cứu

Cho ảnh I, câu hỏi q, tri thức tổng quát K và kết luận c. Tách ít nhất bốn thuộc tính:

1. **Truth:** K có đúng theo nguồn y khoa không?
2. **Relevance:** K có liên quan đến chủ đề của q không?
3. **Premise support:** những điều kiện cần để vận dụng K có được I và các input được phép cung cấp hỗ trợ không?
4. **Conclusion sufficiency:** ngay cả khi tiền đề đúng, K có đủ để kết luận c, hay chỉ cho biết một khả năng/liên hệ?

Một quan hệ dạng “A có thể liên quan B” không tương đương quy tắc “A ⇒ B”. Trường hợp có A nhưng suy B chắc chắn là lỗi độ mạnh suy luận, không chỉ lỗi thiếu tiền đề. Dataset cần tách hai kiểu này. Đây là **đề xuất phân rã cho nghiên cứu**, không tuyên bố một taxonomy mới đã được chứng minh.

Với CXR, nhiều điều kiện như triệu chứng, thời gian, xét nghiệm hoặc bệnh sử không quan sát được. Khi không có input đó, nhãn đúng thường là **không xác định được điều kiện áp dụng**, không phải chắc chắn “không áp dụng”. Do vậy nên dùng thuật ngữ **evidence-conditioned support** trong task core thay vì hứa suy ra toàn bộ clinical applicability chỉ từ ảnh.

### 5.2. Ma trận prior art ở mức đã xác minh

| Trục kiểm tra | Tiền lệ rõ | Phần ứng viên cần làm rõ |
|---|---|---|
| Chọn lọc evidence có liên quan | CKRA, AMiF, mR²AG | Relevance cao nhưng tiền đề chưa được hỗ trợ |
| Conflict detection/routing | CAER, MMKC | Phân biệt unsupported với contradicted trong tri thức tổng quát y khoa |
| Phân biệt ảnh và văn bản đúng/sai | CMC-Bench | Giữ K đúng, thay trạng thái hỗ trợ của ca, không dùng misinformation làm toàn bộ benchmark |
| Robustness và utility đồng thời | Yoran et al., RAFT, mR²AG | Đo đồng thời trên các nhóm MedVQA đã kiểm chứng joint dependence |
| Grounded Medical VQA | HEAL-MedVQA, GEMeX, DEPO, A3Tune | Kết nối lỗi grounding với việc vượt quá mức kết luận mà evidence cho phép |
| Medical RAG reliability | MedRGB | Bổ sung tiền đề quan sát từ ảnh và nguồn uncertainty tách biệt |

**Kết luận được phép:** “Trong các phần tài liệu đã truy cập, chưa xác minh được một công trình kết hợp đầy đủ các điều kiện trên theo đúng giao thức đề xuất.” **Không được biến thành:** “Không có công trình nào giải quyết bài toán này.”

### 5.3. Ba khoảng trống, xếp theo độ phù hợp cho nhóm sinh viên

**G1 — Evaluation/data, ưu tiên cao nhất.** Xây các nhóm can thiệp giữ ảnh–câu hỏi cố định và thay loại K; đồng thời dùng cặp ảnh thật có nhãn khác nhau để kiểm tra cùng một K. Một K phải có khả năng hữu ích ở ca này và không đủ căn cứ ở ca khác, để model không chỉ nhận diện văn phong của negative. Cần chuyên gia kiểm định; đây là nút thắt lớn hơn việc viết retriever.

**G2 — Method, triển khai sau G1.** Học tiền đề supported/contradicted/unknown và sử dụng nó để điều kiện hóa trả lời. So với gating relevance, thử thêm giám sát support và objective nhất quán có điều kiện. Nếu same-data SFT đạt tương đương, kết luận đóng góp chính là dữ liệu/protocol; không cố gọi loss là thuật toán mới.

**G3 — Transfer, mở rộng có chọn lọc.** Kiểm tra unseen knowledge source, rule family, template và tiếng Việt. Tiếng Việt chỉ đáng là đóng góp riêng nếu có kiểm định thuật ngữ, phủ định/uncertainty và phân tích language shift; dịch máy bộ tiếng Anh rồi đo accuracy chưa đủ.

### 5.4. Các phát hiện có thể làm đề tài phải đổi hướng

- Một đối thủ đã có cùng task, annotation và evaluation: bỏ claim “task mới”, chuyển sang reproduction/extension có câu hỏi khác.
- K_na không gây lỗi hơn distractor thông thường sau khi cân bằng độ dài/chủ đề: novelty của failure mode yếu.
- Người gán nhãn không thể phân biệt supported/contradicted/unknown ổn định: sửa task hoặc hạ phạm vi.
- Knowledge QA vẫn giải được chỉ từ text/K: chưa chứng minh nghiên cứu MedVQA joint reasoning.
- SFT cùng dữ liệu giải quyết được toàn bộ vấn đề: đóng góp dữ liệu đáng báo cáo; method claim không được hỗ trợ.
- Mô hình luôn từ chối hoặc bỏ K: không thành công dù harmful flips giảm.

## 6. Thiết kế khóa luận được đề xuất

### 6.1. Research questions và giả thuyết

| RQ | Giả thuyết có thể bác bỏ | Thiết kế kiểm tra |
|---|---|---|
| RQ1 | Tri thức đúng, liên quan nhưng chưa đủ tiền đề gây lỗi khác với irrelevant/false evidence | Matched groups, cân bằng wording/length/topic; phân tích paired transitions |
| RQ2 | Giám sát premise support hữu ích hơn relevance gating | Cùng backbone/dữ liệu/budget; so gate relevance với gate support |
| RQ3 | Conditional learning giảm harmful errors mà giữ utility | Risk tại cùng coverage, accuracy/gain trên knowledge groups và confidence intervals |
| RQ4 | Lợi ích tồn tại ngoài template/source train | Holdout rule families, sources, templates và một external dataset |

Giả thuyết không phải kết quả mong muốn phải đạt. Trước khi train, ghi primary comparison, endpoint, non-inferiority margin và quy tắc dừng trong protocol.

### 6.2. Input, output và phạm vi

**Input:** một CXR frontal, một câu hỏi ngắn, tối đa một nhóm tri thức nhỏ có provenance. Cấu hình core không đưa ground-truth report hoặc nhãn finding vào inference. Metadata nào có sử dụng phải được khai báo và cung cấp công bằng cho baselines.

**Output:** answer chuẩn hóa; trạng thái support; quyết định trả lời/từ chối; evidence ID nếu có. Không buộc sinh chain-of-thought dài. Lời giải thích chỉ là output cần kiểm định, không phải bằng chứng trực tiếp của cơ chế nội bộ.

**Task core:** visual finding và câu hỏi liên kết một finding quan sát được với tri thức mô tả/giải thích có kiểm định. Tránh treatment recommendation và chẩn đoán nguyên nhân không thể xác định từ ảnh.

**Ngôn ngữ:** core tiếng Anh để tận dụng baseline/dataset; tiếng Việt là extension nếu có review. Đây là khuyến nghị thiết kế, không phải yêu cầu đã được người dùng xác nhận.

### 6.3. Dữ liệu và chất lượng nhãn

| Nguồn | Vai trò đề xuất | Tình trạng đã kiểm tra | Điều kiện còn thiếu |
|---|---|---|---|
| SLAKE | Pilot knowledge/visual QA | Trang tác giả có KG, annotations và download; bản 1.0 được làm sạch có thể khác paper [W28] | Đọc license của gói tải; audit joint dependence; không mặc định toàn bộ SLAKE là CXR |
| GEMeX | Core CXR QA/grounding nếu access phù hợp | Repo tác giả có data subset, training code; có paper CXR explainability [W29] | Kiểm tra quyền ảnh nguồn, split/patient IDs, nhãn nào do người hay máy tạo |
| VinDr-CXR | Finding/localization validation | PhysioNet mô tả radiologist annotations và credentialed access [D01] | Quyền truy cập thực tế của nhóm chưa xác minh; không coi “open” là tải tự do |
| ViX-Ray | External Vietnamese validation | LREC 2026 xác nhận 5,400 ảnh với findings/impressions do chuyên gia viết [W30] | License, download và bệnh nhân/split chưa audit trong phiên này; report không tự là nhãn absence |
| VQA-RAD | Benchmark phụ | Được sử dụng xuyên suốt bộ paper | Không dùng benchmark nhỏ làm bằng chứng duy nhất; xác minh bản tải và split trước chạy |

Không cần tải hàng loạt MIMIC-CXR hoặc xây UMLS KG lớn cho pilot. Nếu chọn UMLS/SNOMED/RadLex, phải kiểm tra đúng điều khoản phiên bản và kiểu quan hệ; tên ontology không bảo đảm có các quy tắc suy luận cần cho task.

**Schema tối thiểu cho một intervention group:**

```text
group_id, image_id, patient_id_if_available, source_dataset, split
question, task_type, verified_answer, answerability
visual_premise, premise_support, conclusion_strength
knowledge_text, knowledge_source, source_version
knowledge_truth, topic_relevance, context_role
label_provenance, reviewer_ids, disagreement, adjudication
```

Tách `unknown` về bằng chứng khỏi `abstain` là hành động của model. Một đáp án “không thể xác định từ ảnh” có thể là câu trả lời đúng của task, không tự động là model từ chối trả lời.

**Các điều kiện K:** không K; K hữu ích; K đúng nhưng không liên quan; K đúng và liên quan nhưng tiền đề bị phủ định; K đúng nhưng tiền đề chưa xác định; false claim control. Không gọi tất cả các trường hợp này là “conflict”.

**Các control bắt buộc:** cùng K trên nhiều ảnh; cùng ảnh với nhiều K; K-only; q-only; I+q không K; ảnh không phù hợp có kiểm soát; paraphrase nguồn khác; cân bằng độ dài và vị trí; negative không chỉ chứa từ “không”.

Đổi sang ảnh khác không mặc định giữ đáp án. Occlusion không mặc định biến positive thành negative; ảnh bị che có thể trở thành unanswerable. Không thấy finding trong report không đồng nghĩa finding âm tính.

### 6.4. Quy mô: chỉ là mục tiêu thiết kế

**Pilot:** 100–200 group phát triển và một tập audit độc lập nhỏ; mục tiêu phát hiện lỗi nhãn, shortcut, compute và tính tồn tại của failure mode. Quy mô này không đủ bảo đảm phát hiện cải thiện vài điểm phần trăm.

**Thesis:** xây khoảng 1,000–3,000 group train nếu pipeline nhãn đáng tin; validation riêng; test chuyên gia kiểm định ưu tiên 300–600 group độc lập hoặc nhiều hơn theo power analysis. Một group có nhiều context variants nhưng không được tính chúng như nhiều bệnh nhân độc lập.

Các số trên là mốc lập kế hoạch đề xuất, không phải dữ liệu đã có hoặc yêu cầu tối ưu đã được chứng minh. Sau pilot, dùng variance và tỷ lệ discordant pairs để quyết định cỡ test thực tế. Nếu ngân sách reviewer không đủ, giảm phạm vi claim thay vì tăng synthetic test không kiểm định.

### 6.5. Mô hình và cơ chế tối giản

**Backbone ứng viên:** MedGemma 4B instruction-tuned vì model card chính thức mô tả xử lý medical image/text; cần đồng ý HAI-DEF terms [D02]. Đây không phải khẳng định model tốt nhất cho task. Freeze vision encoder giai đoạn đầu; LoRA/QLoRA cho các module được runtime hỗ trợ. GPU cần thiết phải đo bằng smoke test, không suy ra chỉ từ số tham số.

**Baseline thứ hai:** chọn một VLM thuộc family khác sau pilot và xác minh checkpoint/license. Có thể dùng checkpoint LLaVA-Med để đối chiếu paper khi phù hợp. Không khóa version hoặc lời hứa bộ nhớ khi chưa chạy.

**Bước đầu tiên:** chuẩn hóa answer space có giới hạn, không dùng sinh tự do dài cho primary metric. Retrieval corpus nhỏ, có provenance; cố định context budget và input context giữa methods.

**Phương pháp ứng viên:** một support head g(I,q,K) dự đoán tiền đề ở ba trạng thái; decoder điều kiện hóa vào support representation dự đoán. Nhãn role không đi vào inference như oracle. Một candidate objective:

\[
L=L_{ans}+\lambda_s L_{support}+\lambda_c L_{conditional}.
\]

- L_ans: supervised answer loss trên từng điều kiện có target đã xác minh.
- L_support: CE cho nhãn premise support; chỉ train trên nhãn đủ tin cậy.
- L_conditional: regularizer giữa answer distributions của các context variants mà đáp án được xác minh là bất biến; không áp cho mọi câu knowledge-dependent.

Nếu dùng teacher no-K, tránh distill các dự đoán sai; nếu lọc teacher-correct, ghi rõ selection bias. So với CE trên gold trước khi quyết định KL có cần thiết. Không bổ sung utility CE trùng L_ans rồi tuyên bố đó là objective mới. Utility là ràng buộc đánh giá và cân bằng dữ liệu; chỉ thêm loss khi có giả thuyết độc lập.

**Không dạy abstain vì thiếu external K nếu mô hình có thể biết đáp án từ parametric knowledge.** “Cần tri thức ngoài ảnh” và “bắt buộc cần retrieved context với model này” là hai khái niệm khác. Nếu cần closed-book evidence task, phải công bố quy tắc đánh giá đó và không mô tả nó như QA lâm sàng tự nhiên.

## 7. Baselines, ablations và đánh giá công bằng

### 7.1. Thứ tự baseline

| Mã | Baseline | Mục đích |
|---|---|---|
| B0 | q-only và K+q không ảnh | Kiểm tra shortcut/ngôn ngữ và joint dependence |
| B1 | I+q, không external K | Năng lực quan sát và mốc no-retrieval |
| B2 | I+q+raw K | Định lượng lợi/hại khi thêm knowledge |
| B3 | Raw K + prompt ưu tiên evidence/không suy quá mức | Baseline rẻ, phải vượt trước khi train phức tạp |
| B4 | Relevance filter/gate | Phân biệt lợi ích support supervision với lọc theo topic |
| B5 | Same-data SFT, gồm toàn bộ K variants | Baseline quan trọng nhất để tách contribution của data và loss |
| B6 | Unconditional consistency | Kiểm tra nghiệm bỏ K và vai trò điều kiện hóa |
| B7 | Một method gần: CKRA/MR-RAG hoặc adaptation minh bạch; thêm CAER/DEPO nếu phù hợp | Kiểm tra khoảng trống trước prior art mạnh |
| B8 | Oracle support gate | Chỉ diagnostic upper bound, không so như hệ thống deployable |

Không gọi một bản “inspired by CKRA” là tái lập CKRA. Nếu kiến trúc/dữ liệu khác nhau, báo rõ adapted baseline. So sánh paper numbers trực tiếp giữa các split và answer-normalization khác nhau không hợp lệ.

### 7.2. Ablation theo giả thuyết

1. Full vs cùng dữ liệu chỉ L_ans: đóng góp objective ngoài data.
2. Support supervision vs relevance supervision: semantic applicability có cần không.
3. Ba trạng thái vs hai trạng thái: unknown có thực sự cải thiện hay chỉ thêm class.
4. Conditional vs unconditional regularization: có tránh bỏ qua K không.
5. Có/không K đúng nhưng thiếu tiền đề, giữ tổng số examples/token budget: contribution của loại negative.
6. Predicted vs oracle support: lỗi từ perception/support estimator hay generator.
7. Held-out sources/templates/rules và cùng K trên ảnh mới: có học shortcut không.
8. Hai backbone, seed lặp cho cấu hình chính: mức tổng quát và độ ổn định.

Không cần chạy full factorial của mọi lựa chọn. Khóa backbone/retriever/answer format trước rồi thay từng yếu tố trọng tâm. Baseline phải nhận cùng context, dữ liệu train, số bước hoặc training-token budget; ghi compute bổ sung cho router/localizer.

## 8. Metrics và cách tránh kết luận sai

Với một group i, đặt z_i^0=1 nếu mô hình trả lời đúng trong điều kiện không K, và z_i^k=1 nếu đúng dưới condition k. Abstention trên câu hỏi answerable không được tính là đúng; ghi riêng tác hại do wrong answer và do abstention.

### 8.1. Accuracy và transition table

Báo cáo accuracy/macro-F1 theo từng question type và từng context role. Báo cả bảng đúng→đúng, đúng→sai, sai→đúng, sai→sai, và transitions sang abstain. Accuracy gộp có thể che việc knowledge QA tăng nhưng visual QA giảm.

### 8.2. CFR và conditional-set bias

Nếu giữ cách đặt tên trong proposal:

\[
CFR_k=\frac{\sum_i z_i^0(1-z_i^k)}{\sum_i z_i^0}.
\]

Đây là tỷ lệ mất đáp án đúng trên các case model đã đúng khi không K; không mặc định mọi mất đúng đều là hallucination. Tách harmful wrong-answer rate và unnecessary-abstention rate. Khi mẫu số bằng 0, trả NA, không ghi 0.

Mỗi model có thể có tập no-K-correct khác nhau. Vì thế **không dùng CFR có mẫu số riêng làm bằng chứng so sánh duy nhất**. Báo thêm unconditional harmful-flip rate trên toàn test, accuracy under intervention, paired success và một common-support subset được mô tả rõ. Không chọn subset có lợi sau khi nhìn kết quả.

### 8.3. Knowledge utility

Trên tập joint knowledge QA cố định J:

\[
KUG=\frac{1}{|J|}\sum_{i\in J}(z_i^{help}-z_i^0).
\]

KUG là net accuracy gain, không xác nhận model thật sự dùng ảnh/K theo cơ chế mong muốn. Báo riêng số case được cứu, số case bị làm hỏng và kiểm tra no-image/no-K. Chọn J bằng task annotation, không chọn vì chính baseline đang trả lời sai.

CFR=0 và KUG=0 có thể xuất hiện ở model luôn bỏ K; đó không phải thành công. KUG thấp cũng có thể do no-K ceiling hoặc model biết kiến thức từ pretraining, không tự chứng minh gate thất bại.

### 8.4. Reliability và grounding

- Risk–coverage curve: so tỷ lệ sai trong phần đã trả lời tại cùng coverage.
- Over-refusal trên các case answerable; hiệu quả nhận diện unanswerable riêng.
- Macro-F1/confusion matrix cho support states; đặc biệt nhầm unknown→contradicted.
- Grounding IoU/pointing nếu có nhãn phù hợp; không coi attention heatmap là causal faithfulness.
- Thí nghiệm thay ảnh có nhãn lại và xóa/đổi K: chứng cứ hành vi về dependency, không chứng minh quan hệ nhân quả y học.

### 8.5. Thống kê

Bootstrap theo patient/group, không theo variant độc lập; paired comparison trên cùng case; báo 95% CI. Với nhiều seed, báo variability giữa seed và uncertainty theo test groups riêng. Có thể dùng McNemar cho outcome nhị phân paired khi đơn vị độc lập phù hợp; không áp mù quáng trên nhiều QA cùng bệnh nhân.

Tham khảo độ bất định: với tỷ lệ khoảng 0.5 và n=400 quan sát độc lập, xấp xỉ chuẩn cho CI 95% có half-width khoảng 4.9 điểm phần trăm. Đây là tính toán minh họa 1.96√(0.25/400), không phải power analysis của nghiên cứu; cluster dependence làm cỡ mẫu hiệu dụng nhỏ hơn. Pilot 100–200 groups không chứng minh chắc chắn một cải thiện 2–3 điểm.

## 9. Quy trình nghiên cứu tiếp theo

### 9.1. Lộ trình PoC 8 tuần — đề xuất

| Tuần | Công việc | Deliverable kiểm tra được | Quyết định |
|---|---|---|---|
| 1 | Hoàn thiện literature audit đối thủ tầng I, nhất là các mục L; kiểm tra access | Bảng exact overlap kèm section/page; danh sách code/data/license | Nếu trùng task+protocol thì chỉnh claim trước train |
| 2 | Chốt task definition, rubric và annotation guide cùng người chuyên môn | 30–50 mẫu thử có disagreement/adjudication | Nếu nhãn applicability không đáng tin thì thu hẹp premise support |
| 3 | Tạo pilot groups, dedup/split; tải một checkpoint hợp lệ | Dataset card, schema, checksum, split manifest | Không có data/reviewer thì chưa chuyển sang training |
| 4 | Chạy B0–B4 và đo compute | Error taxonomy, paired transitions, GPU-time/VRAM | Xác nhận failure mode không do template/ảnh quá khó |
| 5 | Same-data SFT B5 và candidate support model | Training/eval reproducible, cùng ngân sách | Đánh giá liệu method có giá trị ngoài data |
| 6 | Ablation trọng tâm | Conditional vs unconditional; 2 vs 3 trạng thái | Nếu utility mất thì sửa hoặc bỏ method claim |
| 7 | Audit độc lập, held-out template/source | Báo cáo leakage/shortcut và CI | Nếu chỉ synthetic gain thì thiết kế lại data |
| 8 | Viết PoC và quyết định phạm vi khóa luận | Quyết định go/pivot/no-go có số liệu | Chỉ mở rộng khi có bằng chứng và nguồn lực |

### 9.2. Sau PoC

**Tháng 3–4:** mở rộng dữ liệu/annotation; khóa test; tái lập đối thủ phù hợp. **Tháng 5–6:** thí nghiệm chính, ablation và seed. **Tháng 7–8:** external transfer, kiểm định lỗi, viết luận văn và release artifacts. **Tháng 9–12 nếu có:** thêm backbone/language, paper submission hoặc benchmark extension.

Đây là lịch đề xuất, không phải thời gian hoàn thành đã xác minh. Chưa có deadline, ngân sách tối đa và lịch chuyên gia hiện tại của nhóm; **không đủ dữ liệu để xác minh khả năng hoàn thành trong một học kỳ cụ thể**.

### 9.3. Go/no-go gates

- **G0 Novelty:** đọc được đối thủ quan trọng và chỉ ra khác biệt kiểm nghiệm được; không còn dùng claim đã bị prior art phủ định.
- **G1 Label validity:** rubric được kiểm tra; disagreement có xử lý; không đánh đồng missing report với negative.
- **G2 Task validity:** evidence cho thấy test có cả joint dependence và failure mode; không chỉ model không đọc được ảnh.
- **G3 Method validity:** vượt strong baseline hoặc đạt trade-off tốt hơn với non-inferiority về utility/clean performance; CI phù hợp. Margin đặt trước từ pilot và mục tiêu ứng dụng, không bịa ngưỡng “chuẩn”.
- **G4 Generalization:** lợi ích không mất hoàn toàn trên nguồn/template/rule chưa thấy.

Nếu G3 không đạt nhưng G1/G2/G4 tốt, luận văn có thể chuyển thành benchmark và phân tích giới hạn có giá trị. Nếu G1 hoặc G2 không đạt, tăng model size không giải quyết được vấn đề thiết kế.

## 10. Khả thi, nhân lực và chi phí

**Đánh giá:** khả thi có điều kiện cho nhóm 2–3 người nếu chỉ làm CXR, một backbone chính, một benchmark có kiểm định và PEFT. Không đủ dữ liệu để khẳng định ngân sách cụ thể hoặc chắc chắn thuê GPU là đủ. Reviewer chuyên môn và tính hợp lệ của labels là điều kiện thực tế quan trọng.

| Vai trò | Trách nhiệm |
|---|---|
| Thành viên 1 | Data protocol, evidence provenance, QA generation, leakage/split audit |
| Thành viên 2 | Baselines, training, experiment tracking, statistics |
| Thành viên 3 nếu có | Reproduction, external validation, annotation tooling, tài liệu/release |
| Người chuyên môn | Kiểm định tri thức và tiền đề ca ảnh; adjudication các case khó |

Nếu chỉ có hai người, ưu tiên giảm số backbone và extension ngôn ngữ; không giảm chất lượng core test để giữ thêm module.

**Không sao chép giá GPU giả định từ proposal thành báo giá hiện tại.** Phiên này chưa khảo giá nhà cung cấp; chi phí tiền thực tế: **Không đủ dữ liệu để xác minh**. Lập ngân sách theo đo lường:

\[
T_{train}=\sum_j\frac{N_j E_j}{r_j\,3600},\qquad
C_{GPU}=\sum_j T^{billed}_j p_j.
\]

N là số training examples thực sự sau nhân variants; E là epochs; r là examples/second đo trên đúng model, resolution, packing và GPU; T_billed gồm setup, evaluation và idle nếu bị tính tiền. Tổng cash cost cộng storage, transfer, API, reviewer và dự phòng.

Ví dụ **chỉ để kiểm tra số học**, không phải dự báo: 10,000 examples × 2 epochs ÷ 0.5 examples/second = 40,000 giây ≈11.1 giờ training/run. Sáu cấu hình × ba seed ≈200 GPU-giờ, chưa gồm evaluation/tuning. Tốc độ 0.5 ở đây là giá trị minh họa, chưa đo.

Reviewer budget: số groups × phút/group × số người review, cộng adjudication. Ví dụ 400 groups × 5 phút × 2 reviewers ≈66.7 reviewer-giờ; 5 phút là giả định minh họa và có thể quá thấp khi đọc ảnh/tri thức khó. Phải đo 30–50 case đầu trước khi đặt lịch.

**Giảm chi phí có cơ sở:** single-image; answer space hữu hạn; frozen encoder; cache embeddings khi kiến trúc cho phép; pilot inference trước train; không RL rollouts từ đầu; không full-finetune model lớn; không xây KG tổng quát. Lưu cả thất bại/seed để tránh chạy lại không cần thiết.

## 11. Giá trị học thuật sẽ đến từ đâu?

### Kịch bản đóng góp mạnh

1. Task có định nghĩa không nhập nhằng, phân biệt truth/relevance/premise support/conclusion strength.
2. Test có review, provenance, paired interventions, held-out nguồn/quy tắc và leakage controls.
3. Chỉ ra failure mode không được benchmark chuẩn đo đúng và không bị baseline rẻ giải quyết hoàn toàn.
4. Method hoặc training recipe cải thiện trade-off so với same-data SFT và đối thủ gần, tái lập trên điều kiện ngoài train.
5. Công bố protocol, code, annotation guide, splits, model configs, failure analysis và dữ liệu trong phạm vi license.

### Kịch bản giá trị vừa nhưng hợp lệ

Benchmark tốt và nghiên cứu negative result: support-gating phức tạp không hơn SFT; lỗi chủ yếu do perception; hoặc model mới đã khá robust nhưng unknown bị xử lý sai. Giá trị phụ thuộc chất lượng phép thử và insight, không bắt buộc một bảng SOTA.

### Kịch bản yếu cần tránh

Một pipeline ghép VLM + KG + prompt, test do cùng LLM tạo/chấm, chỉ so raw RAG, tăng accuracy nhỏ trên random QA split, gọi attention map là giải thích trung thực, hoặc chọn tên method mới làm bằng chứng novelty.

**Khuyến nghị cuối:** chốt nghiên cứu ở mức **“benchmark có kiểm định + nghiên cứu sử dụng tri thức theo bằng chứng của ca”**; giữ method CKA là giả thuyết cần kiểm nghiệm. Đây là cách giữ khả thi và vẫn có cơ hội đóng góp học thuật khi phương pháp đơn giản hóa hoặc không vượt đối thủ.

## 12. Các sửa đổi cụ thể nên áp dụng vào proposal gốc

1. Bổ sung CKRA, MR-RAG, CAER, mR²AG, AMiF, CMC-Bench, Yoran et al., RAFT, MedRGB vào related work bắt buộc.
2. Chuyển novelty từ “conditional KG/gating” sang task/protocol chưa được kiểm chứng đầy đủ.
3. Tách not-applicable, unsupported premise, insufficient conclusion strength và factually false.
4. Thay “image-only” bằng “I+q, no external K” ở công thức và bảng baseline để tránh nhầm input.
5. Tách clinical unknown khỏi model abstention; không ép abstain chỉ vì thiếu retrieved K.
6. Xem utility CE có trùng answer CE không; bắt buộc same-data SFT và matched compute.
7. Sửa cách so CFR giữa model vì tập denominator có thể khác.
8. Đưa access dữ liệu và expert review thành gate trước mở rộng compute.
9. Giữ tiếng Việt/ViX-Ray là extension, trừ khi task và annotation tiếng Việt là trọng tâm được nhóm chốt.
10. Không giữ “first”, “chưa ai làm”, hoặc số tiền/hiệu năng giả định như kết luận đã xác minh.

## 13. Tài liệu tham khảo và tình trạng nguồn

### 13.1. Tài liệu do người dùng cung cấp

Các mã P dưới đây trỏ tới PDF cùng tên trong bộ tài liệu dự án. Nội dung được đọc từ bản người dùng cung cấp; không suy năm/venue khi chưa đối chiếu được.

- **P01:** Act as you think: Reinforcing Consistent Reasoning in Medical Visual Question Answering. `01-Act_as_you_think.pdf`.
- **P02:** Exploring Compositional Generalization of Multimodal LLMs for Medical Imaging. `02-Exploring_Compositional_Generalization_of_Multimodal_LLMs.pdf`.
- **P03:** Fusion of Domain-Adapted Vision and Language Models for Medical Visual Question Answering. `03-Fusion_of_Domain-Adapted_Vision_and_Language_Models_for_Med.pdf`.
- **P04:** MasonNLP at MEDIQA-WV 2025: Multimodal Retrieval-Augmented Generation with Large Language Models for Medical VQA. `04-MasonNLP_at_MEDIQA-WV_2025.pdf`.
- **P05:** MedKInstruct. `05-MedKInstruct.pdf`.
- **P06:** MedThink: A Rationale-Guided Framework for Explaining Medical Visual Question Answering. `06-MedThink-s_R-RAD.pdf`.
- **P07:** OmniMedVQA: A New Large-Scale Comprehensive Evaluation Benchmark for Medical LVLM. `07-OmniMedVQA.pdf`.
- **P08:** Towards Visual Question Answering on Pathology Images. `08-PathVQA.pdf`.
- **P09:** PubMedCLIP: How Much Does CLIP Benefit Visual Question Answering in the Medical Domain? `09-PubMedCLIP.pdf`.
- **P10:** Self-Training Large Language and Vision Assistant for Medicine. `10-Self-Training_Large_Language_and_Vision_Assistant_for_Medic.pdf`.
- **P11:** WorldMedQA-V: a multilingual, multimodal medical examination dataset for multimodal language models evaluation. `11-WorldMedQA-V.pdf`.
- **P12:** Worse than Random? An Embarrassingly Simple Probing Evaluation of Large Multimodal Models in Medical VQA. `12-Worse_than_Random.pdf`.
- **P13:** X-FLoRA: Cross-modal Federated Learning with Modality-expert LoRA for Medical VQA. `13-X-FLoRA.pdf`.
- **P14:** Beyond Surface Features: Advancing Medical Vision-Language Alignment via Dynamic Evidence-Guided Preference Optimization. `14-Beyond-Surface-Features.pdf`, đặc biệt §2.
- **P15:** Multi-modal Concept Alignment Pre-training for Generative Medical Visual Question Answering. `15-MMCAP.pdf`.
- **P16:** Guiding Medical Vision-Language Models with Explicit Visual Prompts: Framework Design and Comprehensive Exploration of Prompt Variations. `16-GuidingMedicalVLMwEVP.pdf`.
- **P17:** Focus on What Matters: Enhancing Medical Vision-Language Models with Automatic Attention Alignment Tuning. `17-Focus-on-What-Matters.pdf`.

### 13.2. Nguồn web chính được đối chiếu

- **W01:** [Beyond Static Knowledge: Dynamic Context-Aware Cross-Modal Contrastive Learning for Medical Visual Question Answering — CKRA](https://pubmed.ncbi.nlm.nih.gov/41052164/). IEEE TMI, issue 2026; DOI 10.1109/TMI.2025.3617289. Đọc abstract tác giả qua PubMed; chưa audit full paper/code.
- **W02:** [MR-RAG: Multimodal Relevance-Aware Retrieval-Augmented Generation for Medical Visual Question Answering](https://openaccess.thecvf.com/content/CVPR2026/html/Li_MR-RAG_Multimodal_Relevance-Aware_Retrieval-Augmented_Generation_for_Medical_Visual_Question_Answering_CVPR_2026_paper.html). CVPR 2026. Metadata nguồn CVF được tìm thấy; mở HTML/PDF bị 403. Không dùng bản tóm tắt bên thứ ba để chốt chi tiết kiến trúc.
- **W03:** [DEPO / Beyond Surface Features](https://aclanthology.org/2026.acl-long.1200/). ACL 2026; đối chiếu P14.
- **W04:** [CAER](https://arxiv.org/html/2607.28991v1). Preprint 31/07/2026. Đọc motivation/evaluation và mô tả router; chưa kiểm tra reproduction.
- **W05:** [mR²AG](https://arxiv.org/abs/2411.15041). V1 2024, v2 17/08/2026; trang arXiv ghi accepted TMM. Đọc abstract hiện hành, không đếm versions riêng.
- **W06:** [AMiF](https://arxiv.org/html/2510.08791v1). CVPR 2025, arXiv upload 09/10/2025. Đọc §3.3–3.4; phân biệt ngày upload với năm hội nghị.
- **W07:** [When Image and Text Disagree: Cross-Modal Evidence Conflict in Multimodal RAG](https://aclanthology.org/2026.magmar-main.3/). MAGMaR 2026, CMC-Bench. Đọc abstract chính thức.
- **W08:** [Making Retrieval-Augmented Language Models Robust to Irrelevant Context](https://arxiv.org/abs/2310.01558). Yoran và cộng sự; v1 2023, revised 2024. Đọc abstract chính thức.
- **W09:** [Comprehensive and Practical Evaluation of RAG Systems for Medical Question Answering](https://arxiv.org/abs/2411.09213). MedRGB, 2024. Đọc abstract chính thức.
- **W10:** [Fine-grained knowledge fusion for retrieval-augmented medical VQA](https://www.sciencedirect.com/science/article/pii/S1566253525001320). Metadata/abstract excerpt; full page bị 403.
- **W11:** [MedEvidence-RAG for Joint Evidence Retrieval and Generator Alignment in Medical VQA and Report Generation](https://ieeexplore.ieee.org/abstract/document/11682264/). Metadata nguồn IEEE; mở trang bị lỗi. Không kết luận về chất lượng thực nghiệm hoặc đầy đủ novelty.
- **W12:** [Enhancing Retrieval-Augmented LVLMs via Knowledge Conflict Mitigation](https://ojs.aaai.org/index.php/AAAI/article/view/37216). AAAI 2026, KCM. Đọc abstract.
- **W13:** [Utility-Oriented Visual Evidence Selection for Multimodal RAG](https://aclanthology.org/2026.acl-long.1620/). ACL 2026. Đọc abstract.
- **W14:** [Self-RAG](https://arxiv.org/abs/2310.11511). 2023 preprint. Đọc abstract.
- **W15:** [RAFT](https://arxiv.org/abs/2403.10131). 2024. Đọc abstract.
- **W16:** [AMANDA](https://aclanthology.org/2025.findings-emnlp.1350/). Findings EMNLP 2025. Đọc abstract và trích đoạn paper nguồn ACL.
- **W17:** [Benchmarking Multimodal Knowledge Conflict for Large Multimodal Models](https://arxiv.org/html/2505.19509v1). MMKC-Bench, preprint 2025; [AAAI 2026 record](https://ojs.aaai.org/index.php/AAAI/article/view/39385).
- **W18:** [Insight Over Sight: Exploring the Vision-Knowledge Conflicts in Multimodal LLMs](https://aclanthology.org/2025.acl-long.872/). ACL 2025. Đọc abstract.
- **W19:** [Unraveling Cross-Modality Knowledge Conflicts in LVLMs](https://arxiv.org/abs/2410.03659). 2024. Đọc abstract.
- **W20:** [Same Semantics, Different Outcome: On the Modality Robustness of Multimodal LLMs under Knowledge Conflict](https://arxiv.org/abs/2609.00550). 2026 preprint; abstract qua search nguồn arXiv, mở trực tiếp lỗi.
- **W21:** [Diagnosing Knowledge Conflict in Multimodal Long-Chain Reasoning](https://arxiv.org/abs/2602.14518). 2026 preprint; abstract qua search nguồn arXiv.
- **W22:** [Knowing or Guessing? Robust Medical VQA via Joint Consistency and Contrastive Learning](https://papers.miccai.org/miccai-2025/0471-Paper3893.html). MICCAI 2025; RoMed/CCL. Trang đọc được ghi code/dataset N/A, không đồng nghĩa chắc chắn chưa có release ở nơi khác.
- **W23:** [Localizing Before Answering: A Benchmark for Grounded Medical VQA](https://ijcai.org/proceedings/2025/853). IJCAI 2025; HEAL-MedVQA/LobA. Đọc abstract.
- **W24:** [MedVH: Towards Systematic Evaluation of Hallucination for LVLMs in the Medical Context](https://arxiv.org/abs/2407.02730). 2024 preprint; abstract tác giả qua search.
- **W25:** [Act as you think](https://aclanthology.org/2026.acl-long.81/). ACL 2026; đối chiếu P01.
- **W26:** [GEMeX-RMCoT](https://arxiv.org/abs/2506.17939). V2 28/10/2025, ACM MM 2025; tên cũ GEMeX-ThinkVG. Không phải hai nghiên cứu độc lập.
- **W27:** [When does retrieval help medical question answering? … CAF-RAG](https://www.sciencedirect.com/science/article/pii/S2352914826000808). Nguồn xuất bản ghi text-based, image-free; trích đoạn nguồn chính, mở full page bị 403.
- **W28:** [SLAKE — trang tác giả](https://www.med-vqa.com/slake/). ISBI 2021; mô tả dataset và bản download.
- **W29:** [GEMeX](https://arxiv.org/abs/2411.16778); [repo chính thức](https://github.com/Awenbocc/GEMeX-Project). ICCV 2025 theo repo; đọc abstract và README.
- **W30:** [ViX-Ray](https://aclanthology.org/2026.lrec-1.239/). LREC 2026; đọc abstract/metadata.
- **W31:** [Benchmarking Retrieval-Augmented Generation for Medicine](https://arxiv.org/abs/2402.13178); [repo MIRAGE](https://github.com/gzxiong/MIRAGE). 2024; metadata/abstract qua nguồn tác giả.

### 13.3. Tài nguyên triển khai, không tính là paper survey riêng

- **D01:** [VinDr-CXR trên PhysioNet](https://physionet.org/content/vindr-cxr/1.0.0/). Nội dung nguồn chính qua search; mở trang timeout. Chưa kiểm tra quyền truy cập tài khoản của nhóm.
- **D02:** [MedGemma 4B IT — model card chính thức](https://huggingface.co/google/medgemma-4b-it). Đọc mô tả và điều kiện truy cập; chưa tải/chạy model.

### 13.4. Candidate chưa đưa vào bằng chứng chốt novelty

Các kết quả như *PathFinder: Graph-structured Reasoning for Medical VQA* (OpenReview bị challenge), *S-Chain*, *MVE-QGCAF*, *Does More Retrieved Evidence Help Visual RAG with Diffusion Language Models?* và các survey MedVQA mới là danh sách mở rộng cần sàng lọc tiếp. Chưa dùng để khẳng định một phương pháp đã/không giải quyết gap. Việc còn các candidate này là một lý do không tuyên bố survey bao phủ tuyệt đối.

## 14. Năm câu hỏi cần nhóm và giảng viên chốt

1. Ai đủ chuyên môn để phân biệt “tiền đề không được ảnh hỗ trợ” với “ảnh không đủ để biết”, và nguồn lực review thực tế là bao nhiêu?
2. Nếu same-data SFT ngang CKA, nhóm có chấp nhận khóa luận benchmark/analysis thay vì nhất thiết một thuật toán mới không?
3. Làm sao chọn knowledge QA mà cả ảnh và K đều có vai trò, nhưng không buộc model giả vờ không biết tri thức đã học từ pretraining?
4. Khi cải thiện robustness đổi lấy giảm coverage, mức trade-off nào được coi là thành công và sẽ được đăng ký trước ra sao?
5. Deadline và trần ngân sách hiện tại có đủ cho nhãn độc lập, strong baselines và external test, thay vì chỉ đủ cho một lần fine-tuning không?

**Phán quyết:** tiếp tục PoC có điều kiện; chưa xác nhận novelty tuyệt đối hoặc method superiority. Giá trị học thuật cao phải được tạo bởi một câu hỏi rõ, test đáng tin, đối chiếu prior art mạnh và kết quả tái lập — không thể được bảo đảm chỉ từ ý tưởng ban đầu.
