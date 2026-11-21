# Hồ sơ bàn giao và nghiệm thu từng phase

Gemini sao chép phần A vào `reports/phase_N/SUBMISSION.md`. Codex lưu phần B vào `reports/reviews/phase_N_review_vX.md`. Không điền PASS trước khi reviewer kiểm tra.

## A. SUBMISSION của Gemini

```markdown
# Phase N — [Tên phase]

- Ngày/giờ Asia/Bangkok:
- Plan version:
- Source commit hoặc source manifest hash:
- Protocol/data/model/config versions:
- Phase trước và review decision:
- Scope được reviewer cho phép:
- Job/run IDs:

## 1. Mục tiêu và việc thực hiện

[Viết chính xác, phân biệt completed / not run / blocked / optional deferred.]

## 2. Artifact index

| Artifact path | Mục đích | Hash/version | Cách kiểm tra |
|---|---|---|---|
| ... | ... | ... | ... |

## 3. Commands thực sự đã chạy

| Command/config | Environment/runner | Exit/status | Log path | Runtime |
|---|---|---|---|---|
| ... | ... | ... | ... | ... |

[Không đưa secrets; log retry và failure, không chỉ run thành công.]

## 4. Evidence theo acceptance criteria

| Criterion từ POC_PLAN | Evidence path/run/table | Kết quả quan sát | Giới hạn |
|---|---|---|---|
| ... | ... | ... | ... |

## 5. Kết quả thí nghiệm, nếu có

- Dataset/split eligibility, N groups, N unique images, counts per condition/task.
- Raw predictions path và missing/invalid/abstain counts.
- Baselines/comparator, primary endpoint, delta, CI và denominators.
- Same-data exposure và compute differences.
- Dependency/shortcut controls, ceiling và các kết quả âm.
- Kết luận nào được evidence hỗ trợ; kết luận nào chưa được hỗ trợ.

## 6. Tài nguyên

- Quota/credit trước và sau, thời điểm kiểm tra:
- GPU/CPU/storage/API dùng thực tế:
- Chi trả phát sinh: [phải bằng 0 theo scope hiện tại]
- Dự báo phase tới và evaluation reserve:

## 7. Sai lệch so với plan/protocol

[Không có thì ghi rõ; có thì lý do, thời điểm, ảnh hưởng đến independence/fairness.]

## 8. Vấn đề cần quyết định ngay

[Evidence, tác động, phương án đề xuất; không yêu cầu reviewer phê duyệt mơ hồ.]

## 9. Đề nghị review

[Phase đã sẵn sàng nghiệm thu hoặc chưa hoàn tất vì lý do cụ thể.]
```

## B. Review của Codex

```markdown
# Review Phase N — vX

- Reviewed source/protocol/artifact hashes:
- Ngày/giờ Asia/Bangkok:
- Engineering status: PASS | REVISE | NEEDS_INPUT
- Scientific verdict: NOT_APPLICABLE | SUPPORT | REFUTE | INCONCLUSIVE

## 1. Kết luận

[Một đoạn nêu việc có đúng scope và evidence đủ không.]

## 2. Phạm vi kiểm chứng thực tế

[Files đã đọc; command/test đã chạy lại; checks không làm được và lý do.
Không nói đã xác minh GPU training nếu chỉ đọc log.]

## 3. Findings

| ID | Severity | File/sample/run | Vấn đề và evidence | Yêu cầu sửa/kiểm tra |
|---|---|---|---|---|
| F-N-01 | BLOCKER / MAJOR / MINOR | ... | ... | ... |

## 4. Acceptance criteria

| Criterion | PASS / FAIL / NOT_RUN / N/A | Evidence |
|---|---|---|
| ... | ... | ... |

## 5. Tính hợp lệ khoa học

[Source labels, input leakage, pairing, dependency, compute fairness,
denominators, uncertainty và mức claim cho phép.]

## 6. Hành động tiếp theo được phép

[Phase cụ thể hoặc chỉ sửa findings. Nêu hạng mục độc lập được phép làm nếu có.]
```

## C. Quy tắc ra quyết định

**PASS**: đáp ứng các tiêu chí bắt buộc trong scope đã chốt, evidence trace được, reviewer tái kiểm tra được phần trọng yếu; không còn blocker/major chưa xử lý. Minor có owner và không ảnh hưởng validity có thể theo backlog.

**REVISE**: lỗi implementation/metric, leakage, không có bằng chứng run, không đủ artifacts, so sánh thiếu công bằng hoặc claim vượt evidence. Gemini sửa đúng finding rồi nộp lại; không được lướt qua phase.

**NEEDS_INPUT**: cần một thông tin/quyền/scope decision thật sự từ người dùng và không có fallback đã được phép cho phần phụ thuộc. Reviewer chỉ rõ câu hỏi và nguồn yêu cầu; các việc độc lập có thể tiếp tục theo phạm vi được ghi.

Scientific **REFUTE/INCONCLUSIVE không tự khiến engineering REVISE**. Một PoC chạy đúng có thể chứng minh giả thuyết không được hỗ trợ. Ngược lại metric đẹp không bù được leakage hoặc gold không hợp lệ.

Codex review khoa học/phần mềm không thay thế xác minh y khoa bởi bác sĩ. Bất kỳ giới hạn chưa kiểm chứng nào phải giữ nguyên trong hồ sơ cuối.
