---
name: no-heavy-model-runs
description: Không tự chạy train hoặc forward model 7B để kiểm chứng; máy 16GB bị ì
metadata:
  node_type: memory
  type: feedback
  originSessionId: dda4b68f-9037-40a7-8307-40dea514d13a
  modified: 2026-11-11T13:33:32.435Z
---

Không tự chạy finetune_lora hay nạp Codestral 7B để kiểm chứng. Mỗi lần mất 20-40 phút và làm máy ì nặng.

**Why:** User ngắt giữa chừng vì chạy lâu và hành máy.

**How to apply:** Kiểm chứng bằng đọc code, file có sẵn hoặc harness nhỏ không nạp model. Cần chạy thật thì đưa lệnh cho user.
