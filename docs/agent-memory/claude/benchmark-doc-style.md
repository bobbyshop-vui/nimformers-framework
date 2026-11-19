---
name: benchmark-doc-style
description: Tài liệu benchmark chỉ ghi số hiện tại mỗi máy/backend và tốc độ không tính JIT lạnh
metadata:
  node_type: memory
  type: feedback
  originSessionId: dda4b68f-9037-40a7-8307-40dea514d13a
  modified: 2026-11-11T15:54:46.484Z
---

Không ghi run cũ, revision hay mốc thời gian trong benchmark.md; mỗi máy/backend chỉ một số hiện tại. Tốc độ prefill, decode và chat không tính setup/JIT kernel lạnh; phần setup in riêng.

**Why:** User muốn tài liệu gọn và số so được với TinyLlama; user nói thẳng "đừng tính cả JIT lạnh vào".

**How to apply:** Khi cập nhật benchmark, thay số cũ bằng số mới đã trừ setup/JIT. Chat của formers-cli phải warmup trước dòng `ready` để lượt đầu không dính JIT. Liên quan [[benchmark-gpu-only]] và [[no-rerun-finished-runs]].
