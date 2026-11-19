---
name: no-log-files
description: Không tạo file log; xem output build và test trực tiếp
metadata:
  node_type: memory
  type: feedback
  originSessionId: dda4b68f-9037-40a7-8307-40dea514d13a
  modified: 2026-11-11T15:13:04.236Z
---

Cấm redirect output build/test vào file log; xem trực tiếp.

**Why:** User yêu cầu.

**How to apply:** Chạy lệnh và đọc output ngay. Không dùng tail hay grep để cắt output (xem [[no-grep-tail]]); chỉnh lệnh cho in đúng phần cần.
