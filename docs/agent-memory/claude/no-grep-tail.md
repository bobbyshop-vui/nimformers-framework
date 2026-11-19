---
name: no-grep-tail
description: "Cấm dùng grep và tail (cả công cụ Grep); đọc file bằng Read, xem output lệnh trực tiếp"
metadata:
  node_type: memory
  type: feedback
  originSessionId: dda4b68f-9037-40a7-8307-40dea514d13a
  modified: 2026-11-11T15:13:00.762Z
---

Cấm grep và tail, trong shell lẫn công cụ Grep. Muốn tìm trong file thì dùng Read với offset/limit; muốn xem output lệnh thì xem nguyên output.

**Why:** User cấm thẳng sau khi tôi dùng grep/tail để lọc output và tìm code. Để an toàn cũng tránh luôn các bộ lọc đường ống kiểu head/awk.

**How to apply:** Cần phần cuối của một file thì Read với offset. Cần tìm symbol thì đọc theo từng đoạn hoặc đọc đúng module liên quan. Lệnh có output dài thì chỉnh lệnh cho in đúng thứ cần thay vì lọc sau. Liên quan [[no-log-files]] và [[no-rerun-finished-runs]].
