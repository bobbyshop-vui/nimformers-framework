---
name: trace-must-be-core-nim
description: "Trace phải làm ở lõi bằng Nim, cấm trace bằng Python"
metadata:
  node_type: memory
  type: feedback
  originSessionId: dda4b68f-9037-40a7-8307-40dea514d13a
  modified: 2026-11-11T13:33:57.021Z
---

Trace thực hiện ở lõi bằng Nim. Cấm Python trace. Đụng runtime đủ để tạo được model.fir. Không spam Opus.

**Why:** User dặn.

**How to apply:** Thêm recorder vào lõi Nim, không viết trace phía Python.
