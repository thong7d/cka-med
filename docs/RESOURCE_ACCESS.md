# Tài nguyên và quyền truy cập cho PoC trước 23/11

Kiểm tra nguồn chính thức ngày 16/11/2026. Đây là ghi nhận điều kiện được công bố tại nguồn, không thay cho kiểm tra điều khoản của đúng bản tải hoặc quyền của tài khoản nhóm.

## 1. Lựa chọn đề xuất

**Đường chính:** SLAKE tiếng Anh + KG đi kèm, một backbone nhỏ, Kaggle GPU. **Backbone mặc định:** `Qwen/Qwen2.5-VL-3B-Instruct`; dùng đúng Qwen Research License. MedGemma 4B là ứng viên thay thế khi có access kịp và cần cải thiện visual competence; không chạy cả hai full experiment.

Chưa cần xin VinDr/MIMIC cho PoC một tuần. Không đợi approval dài hoặc thêm dataset lớn để bắt đầu công việc độc lập.

## 2. Bảng quyết định access

| Tài nguyên | Nguồn chính thức và điều kiện thấy được | Có cần người dùng thao tác? | Quyết định PoC |
|---|---|---|---|
| SLAKE | [Trang tác giả](https://www.med-vqa.com/slake/) dẫn đến [BoKelvin/SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE); [card](https://huggingface.co/datasets/BoKelvin/SLAKE/raw/main/README.md) ghi CC BY 4.0 | Không thấy gate xét duyệt trên các file metadata đã tải; phải attribution và giữ provenance | Dùng chính; Gemini tải `imgs.zip` và kiểm tra file kèm/license trước xử lý ảnh |
| Qwen2.5-VL-3B-Instruct | [Model](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct), [LICENSE](https://huggingface.co/Qwen/Qwen2.5-VL-3B-Instruct/raw/main/LICENSE): research/non-commercial; commercial cần license riêng | Không thấy gate approval trên card; sử dụng vẫn phải theo license | Mặc định để tránh chờ; **không ghi nhầm Apache 2.0** |
| MedGemma 4B IT | [Model card](https://huggingface.co/google/medgemma-4b-it) yêu cầu đăng nhập HF và đồng ý HAI-DEF terms để access files | Người dùng cần tự review/accept tại trang model; card nói request được xử lý ngay. Chưa xác nhận access tài khoản | Tùy chọn, chỉ thay backbone trước khi khóa thí nghiệm |
| VinDr-CXR 1.0 | [PhysioNet](https://physionet.org/content/vindr-cxr/1.0.0/) yêu cầu credentialed user, CITI Data or Specimens Only Research và ký DUA của project | Cần credentialing/training/DUA; không dự đoán thời gian được duyệt | Backlog khóa luận; đáng cân nhắc nếu chuyển sang CXR với radiologist annotations |
| MIMIC-CXR-JPG 2.1 | [PhysioNet](https://physionet.org/content/mimic-cxr-jpg/2.1.0/) có điều kiện credentialed access, training và DUA | Cần quyền tương ứng; nhãn tự động từ report không tự thành gold visual | Chưa cần trong tuần; không là fallback qua mirror không rõ quyền |
| GEMeX | [Repo tác giả](https://github.com/Awenbocc/GEMeX-Project) có subset và code | Repo công khai không tự chứng minh quyền tải/chia sẻ mọi ảnh nguồn; cần audit cụ thể | Không là dependency bắt buộc của PoC |
| Kaggle GPU | [Docs Notebooks](https://www.kaggle.com/docs/notebooks); trang docs không cung cấp text đọc được ở lần mở này | Kiểm tra Accelerator, Internet, xác minh tài khoản nếu nền tảng yêu cầu, quota còn lại trong UI | Runner chính; không mặc định có số giờ hay loại GPU cố định |
| Modal | [Pricing](https://modal.com/pricing): Starter công bố credit miễn phí theo tháng, compute vẫn được tính phí tài nguyên | Cần kiểm tra credits **còn lại**, billing, giới hạn và credentials của tài khoản | Runner dự phòng; chỉ dùng trong phần miễn phí thực tế còn lại |

## 3. Nếu muốn chuẩn bị MedGemma ngay

1. Đăng nhập HF bằng tài khoản sẽ dùng trên runner.
2. Mở [google/medgemma-4b-it](https://huggingface.co/google/medgemma-4b-it), đọc và chấp nhận HAI-DEF terms nếu đồng ý.
3. Tạo token đọc phù hợp, lưu qua Kaggle Secrets hoặc Modal Secrets. Không gửi token vào chat, commit, notebook output hoặc report.
4. Gemini chạy access probe không in token, lưu `accessible=true/false`, revision và lỗi đã che thông tin nhạy cảm.

Đây là ứng viên được đề xuất trong survey, **chưa phải điều kiện bắt buộc**. Nếu chưa có access trong ngày 17/11, tiếp tục đường Qwen; chỉ đổi trước Phase 3 khi development chứng minh cần đổi. Thay backbone sau khi mở test đòi evaluation version mới và không còn là test xác nhận ban đầu.

## 4. Quản lý compute miễn phí

[Modal Pricing](https://modal.com/pricing) công bố Starter có 30 USD credit/tháng, nhưng không cho biết số dư tài khoản nhóm. T4 được niêm yết 0,000164 USD/giây, L4 0,000222 USD/giây và A10 0,000306 USD/giây ở lần kiểm tra; CPU/RAM/storage có khoản riêng. Không coi credit là GPU miễn phí không giới hạn, và không dùng giá giả định cũ của proposal.

Gemini phải lưu `reports/phase_0/compute_budget.json` gồm:

```json
{
  "as_of_local": "TO_FILL",
  "kaggle_gpu_type": "TO_VERIFY",
  "kaggle_remaining_gpu_hours": null,
  "modal_remaining_credit_usd": null,
  "paid_spend_authorized_usd": 0,
  "session_timeout_minutes": 60,
  "eval_reserve_fraction": 0.30,
  "training_allowed": false
}
```

Đã kiểm tra read-only qua Kaggle MCP trong phiên lập plan: quota GPU 30 giờ, đã dùng khoảng 2,72 giờ, reserved 0, còn khoảng **27,28 giờ**. Mốc refresh trả về là **21/11/2026 lúc 07:00 Asia/Bangkok**. Snapshot nằm ở `artifacts/source_audit/kaggle_quota_2026-11-16.json`. Đây không là cam kết GPU luôn sẵn hoặc quota tương lai; Gemini phải cập nhật trước chạy. Modal chưa có số dư được xác minh.

`null` là chưa biết, không là vô hạn. Chỉ bật training khi đã đo smoke test, dự báo cả training/evaluation và chứng minh đủ quota. Tự giới hạn concurrency, timeout, max steps, số lần retry và checkpoint; kiểm tra lại số dư trước mỗi batch job Modal. Nếu không bảo đảm giữ trong phần miễn phí, dùng Kaggle hoặc bỏ lượt chạy; không tự nạp tiền hay đổi paid plan.

Ưu tiên tối thiểu 30% compute khả dụng đã xác nhận cho evaluation, export và chạy lại khi có lỗi. Không hứa số GPU-hours cần thiết trước profiling. Chạy cùng một GPU class và cấu hình cho SFT/candidate khi khả thi; ghi khác biệt nếu phải đổi.

## 5. Những việc không cần đợi access

Gemini có thể làm schema, metric fixtures, audit metadata, split checks, context rendering, experiment configs và tài liệu trên CPU. Không khởi tạo GPU chỉ để cài package, tải metadata, đọc survey hoặc viết báo cáo.

Chỉ xin quyền tài nguyên hạn chế khi Phase 0/2 chứng minh nó là nút thắt thật. Nêu rõ task nào không làm được bằng SLAKE/Qwen và chi phí thời gian nếu bổ sung. Người dùng tự ký điều khoản; không để agent ký thay.
