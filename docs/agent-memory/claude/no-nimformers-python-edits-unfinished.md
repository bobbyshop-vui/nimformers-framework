---
name: no-nimformers-python-edits-unfinished
description: "Không sửa nimformers (binding Python, lib_c_api) khi việc đang làm chưa xong"
metadata:
  node_type: memory
  type: feedback
  originSessionId: dda4b68f-9037-40a7-8307-40dea514d13a
  modified: 2026-11-11T13:33:30.823Z
---

Chưa xong Mamba-3, Codestral và lifter thì không đụng nimformers/vm.py, lib_c_api.nim hay thư viện libnimformers.

**Why:** User cấm gay gắt khi tôi tự chuyển sang thêm op simd vào binding Python giữa chừng.

**How to apply:** Làm xong phần đang dở trước; chỉ sửa binding Python khi user giao việc đó.
