---
name: finish-everything-no-partial
description: "Làm hết toàn bộ scope một lượt, không dừng hỏi giữa chừng, không báo xong khi còn thiếu"
metadata:
  node_type: memory
  type: feedback
  originSessionId: dda4b68f-9037-40a7-8307-40dea514d13a
  modified: 2026-11-11T13:33:28.880Z
---

Khi user giao một scope lớn thì làm trọn vẹn cả scope: không dừng lại hỏi, không báo "xong" khi còn phần được yêu cầu chưa làm, không chia nhỏ rồi bỏ lại phần sau.

**Why:** User nổi giận nhiều lần khi tôi làm dở (Codestral, Mamba-3, GEMM đủ kiểu) rồi vẫn commit và báo xong.

**How to apply:** Lập danh sách đầy đủ theo lời user, làm lần lượt đến hết, chỉ báo xong khi từng mục đã làm và kiểm chứng; mục nào chưa kiểm chứng được thì nói rõ.
