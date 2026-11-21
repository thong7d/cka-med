# CKA-MedVQA — PoC trước 23/11/2026

Trạng thái: đã phân tích tài liệu, kiểm tra metadata/access và soạn kế hoạch; **chưa triển khai/chạy thực nghiệm CKA**. Gemini sẽ thực hiện; Codex nghiệm thu từng phase.

## Tài liệu thực hiện

- [Kế hoạch PoC đầy đủ](docs/POC_PLAN.md): scope, lịch 16–23/11, 8 phase, task/data/loss/metrics, artifacts và acceptance criteria.
- [Prompt bàn giao Gemini](GEMINI_HANDOFF.md): bắt đầu **chỉ Phase 0**, rồi bàn giao review.
- [Phân tích tài liệu gốc](docs/SOURCE_ANALYSIS.md): hiểu đề tài, điểm cần chỉnh và giới hạn của PoC.
- [Data/model access và compute](docs/RESOURCE_ACCESS.md): nguồn chính thức, quyền cần thao tác, Kaggle/Modal.
- [Template bàn giao và nghiệm thu](docs/PHASE_REVIEW_TEMPLATE.md): engineering status và scientific verdict riêng.

## Dữ liệu đã kiểm tra

Metadata SLAKE và KG gốc ở `artifacts/source_audit/slake_metadata/`; chưa tải ảnh. Chạy lại audit bằng Python 3.10+:

```text
python scripts/audit_slake_metadata.py
```

Script chỉ dùng thư viện chuẩn, không tải mạng hoặc chạy model. Output là `audit_summary.json`, kèm hashes và các giới hạn. Snapshot Kaggle trong `artifacts/source_audit/kaggle_quota_2026-11-16.json` phải cập nhật trước khi lập lịch jobs.

Text PDF và slide trích xuất ở `artifacts/source_audit/`; các file đề tài gốc được giữ nguyên. Code entry points và report directories trong plan là đầu ra cần Gemini tạo, không phải implementation đã có.
