---
name: stop-and-report-trace-errors
description: "Lỗi trace/lift/FIR: mặc định tự sửa tới khi chạy đúng, không dừng hỏi; chỉ dừng báo khi user dặn rõ là báo lỗi"
metadata:
  node_type: memory
  type: feedback
  originSessionId: dda4b68f-9037-40a7-8307-40dea514d13a
  modified: 2026-11-16T14:50:22.280Z
---

Mặc định: FIR/lift/trace có bug thì tôi có nghĩa vụ tự sửa tới khi chạy đúng (khớp native), không dừng lại hỏi "có sửa tiếp không". Chỉ dừng và báo nguyên văn lỗi khi user đã dặn rõ trong phiên đó là "gặp lỗi thì báo".

**Why:** Ngày 11/11/2026 user giận vì tôi tự sửa 5 lỗi liên tiếp sau khi họ dặn báo lỗi. Ngày 16/11/2026 thì ngược lại: pha prefill Codestral lift ra token sai, tôi dừng hỏi và user nổi giận ("FIR bị bug thì mày có nghĩa vụ phải fix", "ko automatic đc à… lãng phí thời gian").

**How to apply:** Khi nhiệm vụ là "trace/lift cho thành công" hay "áp tối ưu", lỗi đúng/sai của graph là việc trong scope: giao sub-agent sửa ngay và chỉ báo kết quả. Xem [[finish-everything-no-partial]].
