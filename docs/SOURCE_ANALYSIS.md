# Phân tích tài liệu và quyết định thiết kế PoC

Ngày: 16/11/2026, múi giờ Asia/Bangkok. Phạm vi khảo sát: các file trong thư mục CKA-Med và một số nguồn chính thức liên quan trực tiếp đến khả năng thực hiện trước 23/11/2026.

## 1. Hiện trạng và tài liệu đã kiểm tra

Thư mục ban đầu có 7 file, chưa có source code, environment, dataset ảnh, checkpoint hoặc kết quả CKA; cũng chưa là Git repository. Các con số trong proposal không phải kết quả đã chạy.

| File | Nội dung và vai trò | Cách sử dụng |
|---|---|---|
| `Conditional_Knowledge_Alignment_MedVQA_Research_Report.md` | Proposal nền ngày 20/10; task, objective, schema, baseline, chi phí giả định, roadmap | Đọc các phần khoa học và thực nghiệm; giữ ý tưởng robustness đồng thời utility |
| `CKA_MedVQA_Research_Report.pdf` | 35 trang; cùng cấu trúc chủ đề với proposal Markdown | Đã trích text và đối chiếu cấu trúc; dùng Markdown để định vị mục. Không khẳng định hai bản giống từng byte |
| `CKA_MedVQA_Literature_Survey_2026-10-24.md` | Survey ngày 24/10; sửa novelty, task semantics, metric và thiết kế so sánh | Ưu tiên các sửa đổi khoa học trong tài liệu này khi mâu thuẫn proposal cũ |
| `CKA_MedVQA_Literature_Survey_2026-10-24.pdf` | 19 trang; bản trình bày survey | Đã trích text, kiểm tra cấu trúc; không xem là một nguồn bằng chứng độc lập thứ hai |
| `MedKInstruct.pdf` | Paper gốc 13 trang; method §3–4, experiments §5, limitations và appendix | Đọc trực tiếp để phân biệt KG ở training với KG ở inference |
| `old/CKA_MedVQA.pptx` | Bản thuyết trình cũ; trích được 9 slide XML, dù nhãn trang dùng `/10` | Chỉ là tóm tắt ý tưởng; không dùng làm implementation specification |
| `old/CKA_MedVQA_GVHD_ThuyetTrinh_10slides.md` | Script trình bày; có placeholder và tên file deck khác file thực tế | Dùng hiểu động lực; không dùng chứng minh kết quả, novelty hoặc chọn backbone |

Text trích xuất lưu tại `artifacts/source_audit/`. Bộ 17 PDF và các công trình khác được nhắc trong survey không hiện diện đầy đủ ở thư mục này. Không được nói đã đọc lại toàn bộ các paper đó trong phiên này.

## 2. Hiểu đề tài ở mức có thể kiểm nghiệm

Đầu vào dự kiến là ảnh I, câu hỏi q và tri thức ngoài K. Câu hỏi chính: **có thể hạn chế việc K làm hỏng đáp án vốn được ảnh hỗ trợ, đồng thời giữ lợi ích của K khi câu hỏi cần tri thức không?**

Một phát biểu có thể đúng tổng quát nhưng chưa đủ căn cứ cho ca ảnh. Quan hệ `A may cause B` không chứng minh bệnh nhân có A, cũng không chứng minh B chắc chắn có mặt. Phải tách:

1. Độ đúng của K theo nguồn.
2. Liên quan chủ đề giữa K và q.
3. Bằng chứng cho tiền đề trên ca cụ thể.
4. Độ mạnh kết luận mà quan hệ cho phép.

Ví dụ suy tim có thể liên quan tràn dịch là một minh họa logic, **không phải nhãn cho bất kỳ ảnh nào trong repo**. Không thấy phát hiện trong report không đủ để gán `no`; không có clinical metadata không đủ để gán tiền đề `contradicted`.

Tăng accuracy tổng không trả lời được câu hỏi nghiên cứu. Cần xem transitions dưới các điều kiện K, utility trên tập câu hỏi cố định, phụ thuộc ảnh và khả năng bỏ K/từ chối hàng loạt.

## 3. Các điểm cần sửa khi chuyển proposal thành code

| Vấn đề | Quyết định cho PoC |
|---|---|
| Proposal dùng `K_na` cho nhiều trường hợp | Tách `source_supported_distractor`, `premise_contradicted`, `premise_unknown`, `false_assertion_control`; chỉ dùng nhãn premise nếu có nguồn xác nhận |
| `unknown` dùng chung cho thiếu bằng chứng và từ chối | Tách semantic answer, trạng thái bằng chứng và hành động `abstain`; không tự tạo gold unknown |
| Đề xuất thêm utility CE ngoài answer CE | Không cộng hai loss tương đương rồi gọi là thuật toán mới; utility trước hết là yêu cầu đánh giá |
| Pseudocode consistency có thể giữ teacher sai | Candidate dùng gold CE và regularization giữa nhánh; không lấy no-K prediction làm gold |
| CFR có mẫu số riêng từng model | Báo cả counts, accuracy và flip trên toàn tập; CFR là chỉ số phụ trợ |
| KG có thể truy xuất từ gold finding/report | Cấm gold trong inference. Tách controlled context replay khỏi retrieval tự động |
| Gate nhận role/type oracle | Role chỉ cho sampler/loss; model chỉ nhận thông tin công bố trong input contract |
| Proposal có Qwen ~3B, survey có MedGemma 4B | Dùng Qwen3B làm đường chạy không phụ thuộc cấp quyền; MedGemma là lựa chọn thay nếu có access sớm và competence tốt hơn trên development |
| Roadmap 8 tuần và budget 87,25 USD | Đã bị thay bằng hạn 23/11 và chỉ dùng quota/credits miễn phí theo xác nhận của người dùng |
| Novelty chủ yếu là conditional gate/KG | Không được giữ claim rộng. PoC kiểm tra usefulness và gap; không hứa thuật toán mới hoặc SOTA |

## 4. MedKInstruct thực sự cung cấp bằng chứng gì?

Theo PDF địa phương:

- §3.1 mô hình hóa MedVQA dưới dạng `A = M(Q, P)`.
- §4.1–4.2 dùng ảnh/caption và MMKG để tạo instruction data, có exact/fuzzy entity matching, multi-hop QA và hard negatives.
- §4.3 SFT từ ảnh và câu hỏi; GRPO dùng key answer, path entities và format trong reward.
- §5.1 dùng hai backbone 7B, LoRA; đánh giá close-ended accuracy và open-ended recall. Vì vậy average score trong paper không đồng nhất với accuracy chuẩn hóa của PoC.
- Mức tăng 4,16% và 4,50% là kết quả paper báo cáo, không là dự đoán CKA. Không dùng chúng để suy ra rằng thêm retrieved K vào inference chắc chắn hữu ích.
- Hard negative được tạo bằng disease tương tự rồi đặt đáp án No (§4.2). Không sao chép cách này thành gold mới nếu không có nguồn nhãn xác nhận âm tính.
- Appendix có prompt ép chọn bệnh ngay cả khi không chắc chắn. Không đưa quy tắc đó vào annotation pipeline của PoC.
- Path reward dựa trên entity xuất hiện trong output không tự chứng minh visual grounding hoặc reasoning faithfulness.

Repository tác giả tồn tại tại [MedKinstruct](https://github.com/Sonder-hang/MedKinstruct); chưa clone, chạy hoặc audit toàn bộ code. Không cam kết tái lập full RL pipeline trong một tuần.

## 5. Những xác nhận mới của người dùng

| Ràng buộc | Đã chốt |
|---|---|
| Deadline | 23/11/2026 |
| Compute | Kaggle GPU và Modal miễn phí; không có ngân sách trả phí được cấp |
| Chuyên môn | Không có bác sĩ hoặc người chuyên môn review |
| Access | Chưa xác nhận có quyền dataset/model hạn chế; chỉ xin nếu thực sự cần |
| Kết quả âm | Chấp nhận và có thể cân nhắc pivot |
| Phạm vi ảnh | Chấp nhận pilot radiology nếu CXR không đủ; hướng CXR giữ cho khóa luận |
| Người thực hiện/nghiệm thu | Gemini thực hiện; Codex nghiệm thu từng phase |

Hệ quả: PoC là **nghiên cứu hành vi trên benchmark có nhãn tham chiếu**, không phải xác minh clinical applicability mới. Nhánh supported/contradicted/unknown cần nhãn chuyên môn sẽ nằm ở backlog.

## 6. Kiểm tra SLAKE bằng metadata tải thực tế

Nguồn tải là mirror được [trang tác giả SLAKE](https://www.med-vqa.com/slake/) dẫn tới. Dataset card khai báo CC BY 4.0; bản tải có các file QA và KG, và đổi tên `validate.json` thành `validation.json`. [Dataset card](https://huggingface.co/datasets/BoKelvin/SLAKE/raw/main/README.md).

Đã tải khoảng 4,3 MB metadata + KG, **chưa tải ảnh**. Script `scripts/audit_slake_metadata.py` đếm lại, lưu SHA-256 và giới hạn trong `artifacts/source_audit/slake_metadata/audit_summary.json`.

| Đại lượng đo từ file | Train | Validation | Test |
|---|---:|---:|---:|
| QA tiếng Anh | 4.919 | 1.053 | 1.061 |
| Ảnh theo `img_name` | 450 | 96 | 96 |
| QA visual CLOSED | 1.772 | 377 | 377 |
| QA loại `kvqa` | 582 | 155 | 148 |
| Ảnh ứng viên CXR theo metadata | 122 | 27 | 30 |
| Visual CLOSED trên ứng viên CXR | 454 | 95 | 114 |
| `kvqa` trên ứng viên CXR | 189 | 54 | 54 |

Ứng viên CXR ở đây dùng `modality == X-Ray` và `location` có Chest hoặc bằng Lung. Đây là filter metadata, chưa xác nhận view/frontal hoặc chất lượng ảnh. Một ảnh có nhiều QA nên 114 QA không là 114 bệnh nhân.

Không thấy `img_name` giao nhau giữa ba split. Chưa kiểm tra pixel hash/perceptual duplicate vì chưa tải ảnh. QA không có patient ID; không có cơ sở xác nhận patient-disjoint hoặc absence of pretraining contamination.

Một số câu `kvqa` training là câu về điều trị, phòng bệnh hoặc triệu chứng cần giả định chẩn đoán. **Loại khỏi core một tuần.** Ưu tiên anatomy/function/body-system có tham chiếu rõ, nhưng phải kiểm tra xem câu hỏi có tự tiết lộ organ và K có trả lời được khi không ảnh hay không.

`kvqa` không tự chứng minh joint dependence. KG gốc có cả các mô tả đơn giản hóa; gọi là `source_asserted`, không gọi toàn bộ là medically verified truth. Chỉ giữ entry có nguồn rõ; entry nghi vấn bị loại, không nhờ Gemini tự sửa y khoa.

Đã xem metadata counts và vài record tham chiếu trước khi có model. Ghi nhận việc này trong protocol; từ khi khóa protocol, không dùng test để chọn prompt/model/hyperparameter.

## 7. Đối chiếu nguồn ngoài và khoảng trống còn lại

- [DEPO, nguồn ACL](https://aclanthology.org/2026.acl-long.1200/) xác nhận mục tiêu evidence alignment và dynamic preference optimization. Không thể gọi việc chống text-over-image là mới.
- [CAER, toàn văn tác giả](https://arxiv.org/html/2607.28991v1) có evidence router và dual prefix experts cho supported/contradicted. Do đó gate theo conflict không đủ làm novelty.
- CKRA đã được survey xếp mức quan trọng; lần mở [PubMed record](https://pubmed.ncbi.nlm.nih.gov/41052164/) trong phiên này không trả abstract có thể đọc. Không coi là đã kiểm chứng lại phương pháp/code.
- MR-RAG, mR²AG, CMC-Bench, AMiF và các prior art còn lại cần được Gemini kiểm tra theo thứ tự trong Phase 0; ghi rõ paper nào mới đọc metadata/abstract. Không biến survey cũ thành chứng minh rằng không có phương pháp tương đương.
- Quyền dùng và đường tải chi tiết cho tài nguyên trong `RESOURCE_ACCESS.md`. Đã kiểm tra quota Kaggle read-only: còn khoảng 27,28 GPU-hours ở snapshot 16/11; số dư Modal chưa xác minh.

## 8. Đánh giá tính khả thi

Khả thi trong tuần: pipeline tái lập, benchmark reference audit, baselines có/không K, báo cáo failure mode, một cặp SFT–candidate nhỏ nếu task và GPU cho phép, kết luận go/pivot/inconclusive.

Không đủ điều kiện trong tuần: clinical support gold mới, benchmark applicability được chuyên gia kiểm định, chứng minh hơn prior art toàn diện, external validation rộng, nhiều backbone/seeds hoặc full RL reproduction.

Phép nghiệm thu quan trọng nhất là **không đánh tráo hai kết luận**: làm đúng một PoC và chứng minh CKA tốt hơn đều cần bằng chứng riêng. Nếu dữ liệu không tạo được joint QA đáng tin hoặc utility không đo được, PoC phải báo điều đó và giảm claim.
