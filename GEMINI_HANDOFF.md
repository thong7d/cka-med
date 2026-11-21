# Lệnh bàn giao PoC cho Gemini

Đọc `docs/POC_PLAN.md` là specification chính. Tài liệu này là prompt điều phối, không thay toàn bộ plan.

## Prompt khởi động — có thể gửi nguyên khối

```text
Bạn là người triển khai PoC cho đề tài CKA-MedVQA. Codex là reviewer độc lập và sẽ nghiệm thu từng phase. Không tự xác nhận PASS thay reviewer.

Đầu tiên đọc các file:
1. docs/POC_PLAN.md
2. docs/SOURCE_ANALYSIS.md
3. docs/RESOURCE_ACCESS.md
4. docs/PHASE_REVIEW_TEMPLATE.md
5. Conditional_Knowledge_Alignment_MedVQA_Research_Report.md
6. CKA_MedVQA_Literature_Survey_2026-10-24.md
7. MedKInstruct.pdf, ưu tiên §3–5 và appendix
8. artifacts/source_audit/slake_metadata/audit_summary.json
9. artifacts/source_audit/kaggle_quota_2026-11-16.json

Deadline PoC: 23/11/2026, Asia/Bangkok.
Tài nguyên: GPU Kaggle và phần credits Modal còn miễn phí. Không tự phát sinh khoản trả phí.
Không có bác sĩ/người chuyên môn hỗ trợ. Không gán bệnh, absence, premise unknown hay clinical support mới từ ảnh bằng suy đoán của agent.
Người dùng chấp nhận kết quả âm/pivot và chấp nhận pilot radiology nếu tập CXR không đủ. Không cần giữ CKA bằng mọi giá.

Mặc định SLAKE English + KG nguồn; Qwen2.5-VL-3B-Instruct một backbone, frozen vision encoder, PEFT nếu cần training. MedGemma 4B chỉ thay sớm nếu đã có access và development chứng minh cần thiết. Đọc đúng license của từng checkpoint; Qwen3B này không mặc định Apache 2.0.

Hiện chỉ có tài liệu và source-audit utility, chưa có PoC implementation/model results. Metadata SLAKE đã tải, ảnh chưa tải. Đừng mô tả code/command đề xuất trong plan như đã chạy.

Hãy thực hiện CHỈ PHASE 0 theo plan. Không chạy GPU training, không thực hiện Phase 1–7 trước khi reviewer cho phép. Có thể làm các access probe và audit nhẹ cần thiết cho P0. Báo ngay nếu có vấn đề ảnh hưởng task, reference quality, access, deadline hoặc miễn phí.

Mọi đầu ra P0 đặt trong reports/phase_0/. Dùng template SUBMISSION; ghi command thật, exit code, file artifact và trạng thái. Có prior_art_matrix kèm URL/section/mức đọc, compute_budget cập nhật, scope_contract và protocol_draft.

Khi hoàn tất, dừng và gửi: phase, việc đã làm, artifact paths, cách kiểm tra, kết quả quan sát thực, các mục chưa làm, vấn đề cần reviewer quyết định. Không viết “đã nghiệm thu”, “clinically validated”, “novel”, “SOTA” hoặc “thành công” khi chưa có evidence tương ứng.
```

## Prompt tiếp tục sau mỗi lần nghiệm thu

```text
Đọc hồ sơ review mới nhất trong reports/reviews/ và SUBMISSION phase trước.
Reviewer đã cho phép: [PHASE/PHẠM VI ĐƯỢC PHÉP].

Nếu status REVISE: chỉ sửa các finding liên quan, cung cấp evidence mới và resubmit cùng phase; không xóa logs/history hoặc tự đổi protocol để che lỗi.
Nếu status PASS: thực hiện đúng phase tiếp theo được cho phép theo docs/POC_PLAN.md.
Nếu verdict khoa học là REFUTE/INCONCLUSIVE nhưng engineering PASS: giữ nguyên kết quả âm và đi nhánh reviewer chỉ định; không tự biến thành thắng lợi.

Không có quyền phát sinh chi phí hoặc thêm nhãn y khoa. Không tune trên final test. Nếu có thay đổi phạm vi/critical dependency thì báo ngay với evidence và phương án.

Kết thúc bằng hồ sơ SUBMISSION có artifact đủ để Codex chạy lại; dừng chờ nghiệm thu.
```

## Bằng chứng tối thiểu Gemini phải giữ

- Source revision/commit hoặc source-hash manifest; mọi thay đổi config/protocol có lịch sử.
- Dataset/source QA IDs, image hashes, split/context generator versions và exclusion logs.
- Input render mẫu chứng minh model không nhận reference answer, source `triple`, location/gold entity, context-role label hoặc patient report.
- Commands/logs thật; dependencies; job/run IDs; elapsed/quota và errors.
- Raw predictions cho tất cả eligible final samples; không chỉ summary CSV, screenshots hay biểu đồ.
- Adapters/checkpoints có base revision, loading config và checksum; không chỉ notebook session URL.
- Same-data SFT/candidate exposure và compute accounting; lambda/mask/gradient checks.
- Metrics scripts, clustered bootstrap seed/config và table generation; phân biệt counts với rates.
- Known limitations, negative results, J candidates bị loại, missingness và các phần chưa thực hiện.

## Các điều bị cấm để bảo vệ tính đúng của PoC

Không dùng synthetic fixtures làm real results; không tự random prediction hoặc bịa log. Không map missing/uncertain reference thành No. Không dùng chính model sinh nhãn/chấm để làm gold. Không coi organ nêu sẵn trong q hay đáp án được cho thẳng trong K là evidence joint reasoning.

Không retrieve theo test answer/location/gold finding; nếu làm oracle diagnostic phải gắn nhãn riêng và loại khỏi main result. Không bỏ failed samples khỏi denominator im lặng. Không chọn checkpoint/prompt/answer aliases sau xem test rồi gọi independent evaluation.

Không tự chuyển sang RL, nhiều backbone, knowledge graph quy mô lớn hoặc support head cần clinical labels. Không ghi model tốt hơn chỉ vì CFR thấp: phải nhìn clean accuracy, utility, denominators, CI và dependency controls.

## Khi giao lại cho Codex

Người dùng chỉ cần gửi:

```text
Nghiệm thu Phase N của Gemini theo docs/POC_PLAN.md.
Hồ sơ: reports/phase_N/SUBMISSION.md
Hãy kiểm tra artifact và chạy lại các phép kiểm tra cần thiết; trả PASS/REVISE/NEEDS_INPUT,
kèm verdict khoa học nếu phase có thí nghiệm, và phạm vi được phép làm tiếp.
```

Nếu Gemini chạy trong một workspace khác, cần đồng bộ source/config/reports/raw artifacts về workspace này trước khi nghiệm thu. Link notebook hoặc câu “đã chạy thành công” không thay được files và logs.
