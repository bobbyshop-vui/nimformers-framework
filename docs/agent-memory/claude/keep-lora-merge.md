---
name: keep-lora-merge
description: Giữ merge LoRA; cấm --no-merge-lora và BYBY_MERGE_LORA=0
metadata:
  node_type: memory
  type: feedback
  originSessionId: dda4b68f-9037-40a7-8307-40dea514d13a
  modified: 2026-11-11T13:33:46.494Z
---

Không dùng --no-merge-lora hay BYBY_MERGE_LORA=0. Khi user hỏi lệnh thì chỉ đưa lệnh, không sửa notebook.

**Why:** User yêu cầu.

**How to apply:** Lệnh chạy mặc định để merge LoRA.
