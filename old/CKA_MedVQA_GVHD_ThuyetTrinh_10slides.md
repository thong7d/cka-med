# Kịch bản thuyết trình CKA-MedVQA trước giảng viên hướng dẫn

**Đề tài:** *Conditional Knowledge Alignment for Evidence-Grounded Medical Visual Question Answering*  
**Bộ trình chiếu:** `CKA_MedVQA_GVHD_Proposal_10slides.pptx`  
**Số slide:** 10 · **Thời lượng mục tiêu:** khoảng **7–9 phút** (không tính hỏi đáp)  
**Thành viên:** `[Họ tên thành viên 1]` và `[Họ tên thành viên 2]` — thay vào slide 1 và slide 10 trước khi trình bày.

> **Nguyên tắc diễn đạt:** Đây là **đề xuất nghiên cứu**, chưa phải phương pháp đã có kết quả. Không dùng các cụm “đã cải thiện”, “đã chứng minh” để nói về CKA. Hai số 4,16% / 4,50% là **kết quả MedKInstruct công bố**, còn 65 GPU-hours / 87,25 USD là **kịch bản dự toán** trong báo cáo nền.

---

## Slide 1 — Giới thiệu đề tài (35–45 giây)

**Lời thuyết trình**

“Em chào thầy. Sau đây nhóm xin trình bày hướng đi của khóa luận *Conditional Knowledge Alignment for Evidence-Grounded Medical Visual Question Answering*, viết tắt là **CKA-MedVQA**.

Ý tưởng của nhóm xuất phát từ một vấn đề khá đơn giản: một mô hình VLM có thể có nhiều kiến thức y khoa (của chính mô hình và tri thức bên ngoài), nhưng kiến thức đúng nói chung chưa chắc đúng với hình ảnh của một bệnh nhân cụ thể. Nhóm muốn nghiên cứu cách để mô hình sử dụng kiến thức **khi cần**, nhưng **không để kiến thức lấn át bằng chứng (evidence) từ hình ảnh**.”

**Chuyển slide:** “Trước tiên, em xin nói ngắn gọn vì sao đây là vấn đề đáng nghiên cứu.”

## Slide 2 — Bối cảnh và động lực (40–50 giây)

**Lời thuyết trình**

Hướng đi này nhóm em lấy ý tưởng chính từ paper **MedKInstruct**, trong paper này có báo cáo mức cải thiện **4,16% trên SLAKE** và **4,50% trên VQA-RAD** . Điều đó cho thấy hướng tăng cường tri thức bên ngoài có giá trị. Tuy nhiên, các nghiên cứu từ các papers như **DEPO** hay **Act as You Think** cũng chỉ ra rằng VLM có thể sinh câu trả lời thiếu căn cứ hình ảnh. Vì vậy, mục tiêu của nhóm không phải là thêm càng nhiều tri thức càng tốt, mà là **khi nào nên sử dụng và tin tri thức đó**.”

## Slide 3 — Ví dụ lỗi cần xử lý (40–50 giây)

**Lời thuyết trình**

“Em lấy một ví dụ giả định để dễ hình dung. Giả sử ta có ảnh X-quang phổi hiện tại **chưa đủ bằng chứng để xác nhận** tràn dịch màng phổi. Nhưng sau khi trích xuất KG, tri thức nói rằng suy tim **có thể gây ra** tràn dịch màng phổi.

Nếu mô hình lấy quan hệ ‘có thể gây tràng dịch màng phổi’ rồi kết luận luôn ‘bệnh nhân này có tràn dịch màng phổi’, thì đó là lỗi: thông tin y khoa có thể đúng, nhưng kết luận trên **ca ảnh cụ thể** lại chưa đủ căn cứ. Nhóm mong muốn là mô hình phân biệt được hai nguồn thông tin (ảnh và KG), và nói rõ cho người dùng khi **chưa đủ bằng chứng để kết luận**.”

## Slide 4 — Research gap và câu hỏi nghiên cứu (45–55 giây)

**Lời thuyết trình**

“Qua tìm hiểu sâu hơn về đề tài, nhóm nhận ra bài toán có **hai vấn đề chính**. Vấn đề thứ nhất là mô hình luôn tin tri thức bên ngoài; điều này dễ khiến nó suy diễn quá mức từ ảnh. Vấn đề thứ hai là mô hình bỏ qua tri thức hoàn toàn; lúc đó có thể chống nhiễu tốt hơn nhưng lại mất khả năng trả lời câu hỏi cần kiến thức y khoa ngoài tầm hiểu biết của VLM.

Do đó, nhóm tập trung vào khái niệm **sử dụng tri thức có điều kiện**. Vấn đề nghiên cứu là: *khi nào tri thức bên ngoài được phép ảnh hưởng đến câu trả lời từ hình ảnh?*”

**Chuyển slide:** “Từ đó, nhóm đề xuất ý tưởng CKA.”

## Slide 5 — Ý tưởng cốt lõi của CKA (45–55 giây)

**Lời thuyết trình**

“CKA không đơn thuần là thêm một bộ lọc tri thức. Nhóm muốn mô hình học **hai cách xử lý khác nhau tùy loại câu hỏi**.

Với câu hỏi thuần thị giác, ví dụ ‘có dấu hiệu này/sự bất thường trên ảnh không?’, câu trả lời phải bám vào ảnh và không bị một đoạn tri thức bên ngoài gây nhiễu làm sai lệch. Ngược lại, với câu hỏi cần kiến thức, chẳng hạn liên hệ giữa một dấu hiệu đã quan sát với cơ chế bệnh lý, mô hình phải **tận dụng được tri thức phù hợp**.

Trong trường hợp chưa đủ bằng chứng, mô hình cần biết giới hạn kết luận. Như vậy, thành công không chỉ là chống nhiễu, mà còn phải **giữ được lợi ích khi trích xuất tri thức từ KG**.”

## Slide 6 — Kiến trúc tổng thể hệ thống (50–60 giây)

**Lời thuyết trình**

“Về kiến trúc, hệ thống có hai luồng chính. Luồng thứ nhất là nhận **ảnh và câu hỏi**, sau đó biểu diễn những thông tin liên quan đến bằng chứng hình ảnh. Luồng thứ hai là truy xuất một số quan hệ y khoa từ **Knowledge Graph** phù hợp với câu hỏi.

Hai luồng đi vào thành phần trọng tâm là **Conditional Knowledge Alignment**, nơi mô hình học cách cân nhắc vai trò của tri thức được bổ sung. Sau đó một Medical VLM sinh câu trả lời cùng trạng thái bằng chứng: có đủ cơ sở hay chưa hay chưa đủ cơ sở để kết luận.

Nhóm dự kiến bắt đầu bằng VLM cỡ **3B tham số**, fine-tune bằng **QLoRA**, chứ không huấn luyện một mô hình lớn từ đầu.”

## Slide 7 — Các trường hợp xung đột (45–55 giây)

**Lời thuyết trình**

Với cùng một ảnh và một câu hỏi, nhóm tạo nhiều phiên bản thông tin: tri thức **hữu ích**, tri thức **đúng nhưng không phù hợp với ca ảnh**, và tình huống **phản thực/đối kháng** so với ảnh đã được gắn nhãn 1 cách rõ ràng. Mục tiêu là kiểm tra liệu chỉ cần thay tri thức bên ngoài có làm mô hình đổi đáp án hay không.”

## Slide 8 — ĐÁNH GIÁ & PHẢN CHỨNG (45–55 giây)

**Lời thuyết trình**

“Nhóm sử dụng 1 số metrics đánh giá. Ví dụ như Chỉ số thứ nhất là **CFR (Conflict Flip Rate)**, đo tỷ lệ câu mà mô hình từ trả lời đúng khi không có tri thức (KG) nhưng lại chuyển thành sai khi nhận tri thức (KG) đầu vào xung đột nhau và gây nhiễu. Với chỉ số này, **càng thấp càng tốt**.

Chỉ số thứ hai là **KUG (Knowledge utility gain)**, đo mức cải thiện khi đưa vào tri thức thực sự hữu ích trên những câu hỏi cần tri thức bên ngoài. Với chỉ số này, **càng cao càng tốt**.

Nhóm sẽ so sánh CKA với các baseline đơn giản mà quan trọng như image-only, thêm KG qua prompt, lọc KG và SFT với hard-negative. Nếu CKA chỉ giảm CFR bằng cách bỏ KG, hoặc chỉ hoạt động trên dữ liệu tự sinh, nhóm sẽ **không xem đó là hướng đi có giá trị**.”

## Slide 9 — Kết thúc và xin góp ý (25–35 giây)

**Lời thuyết trình**

“Đó là hướng nghiên cứu nhóm em muốn đề xuất. Tóm lại, mục tiêu không phải làm một hệ thống Medical VQA có thêm nhiều tri thức, mà là một hệ thống **biết khi nào dùng kiến thức, khi nào phải bám vào ảnh, khi nào cho người dùng biết chưa đủ cơ sở để trả lời**.

Dạ phần trình bày nhóm em đến đây là hết, cảm ơn thầy đã lắng nghe”

---

## Ba câu hỏi GVHD có thể hỏi (chuẩn bị trước, không cần đưa vào slide)

**1. “Đề tài mới ở đâu, DEPO đã chống hallucination do text override rồi?”**  
“Dạ đúng, vì vậy nhóm **không tuyên bố là hướng đầu tiên chống việc văn bản lấn át ảnh**. Phần nhóm muốn kiểm chứng là **giới hạn tác động của tri thức theo điều kiện**: giảm sai lệch từ KG không phù hợp, đồng thời vẫn sử dụng tốt KG hữu ích. Nếu baseline filtering hoặc DEPO tương ứng giải quyết được như vậy, nhóm phải chỉnh lại novelty claim.”

**2. “Nếu mô hình không cần Knowledge Graph thì sao?”**  
“Dạ đó là một khả năng nhóm đưa vào thí nghiệm. Nếu image-only và các baseline đơn giản đáp ứng tốt cả hai nhóm câu hỏi, việc dùng KG có thể không cần thiết trong scope này. Nhóm sẽ thu hẹp hoặc chuyển hướng thay vì giữ KG bằng mọi giá.”

**3. “Không có bác sĩ xác minh nhãn thì kết quả tin được không?”**  
“Nhóm sẽ ưu tiên các nhãn đã được xác minh từ nguồn benchmark, gắn trạng thái `unknown` cho ca không chắc và lập tập đánh giá nhỏ độc lập. Nếu không thể có nhãn đủ tin cậy, nhóm **không đưa ra tuyên bố về độ an toàn lâm sàng** và sẽ thu hẹp mục tiêu nghiên cứu.”

---

## Checklist 2 phút trước khi thuyết trình

- [ ] Thay tên thật của **hai thành viên** trên slide 1 và 10.
- [ ] Chuẩn bị cách đọc rõ: **CFR** = Conflict-induced Flip Rate; **KUG** = Knowledge Utility Gain; **CKA** = Conditional Knowledge Alignment.
- [ ] Đọc đúng nguồn kết quả: MedKInstruct **4,16% / 4,50%**; **chưa có kết quả CKA**.
- [ ] Đọc đúng nguồn chi phí: **65 GPUh / 87,25 USD là ước lượng**, không phải hóa đơn hay báo giá.
- [ ] Thống nhất nhóm dùng tiếng Việt cho phần giải thích; giữ các thuật ngữ VLM, KG, QLoRA, CFR và KUG khi cần.

*Nguồn nền: `Conditional_Knowledge_Alignment_MedVQA_Research_Report.md` và các paper đã cung cấp trong Project. Không tra cứu Internet để dựng tài liệu này.*
