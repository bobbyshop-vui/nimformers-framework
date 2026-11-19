---
name: show-full-log-no-filter
description: "Cấm lọc output lệnh (cut, head, awk, sed...), phải hiện đủ log"
metadata:
  node_type: memory
  type: feedback
  originSessionId: dda4b68f-9037-40a7-8307-40dea514d13a
  modified: 2026-11-12T09:55:50.207Z
---

Khi chạy lệnh (build, trace, benchmark), hiện toàn bộ output. Không pipe qua `cut`, `head`, `awk`, `sed`, `grep`, `tail` để cắt/lọc dòng.

**Why:** User ngắt lệnh 12/11 và nói "cấm lọc text mà phải hiện đủ log" khi tôi chạy benchmark TinyLlama với `cut -c1-260 | head -12`.

**How to apply:** Chạy lệnh trần, để harness hiện đủ output. Không redirect vào file log ([[no-log-files]]). Muốn bớt ồn thì chỉnh cờ của chính chương trình, không cắt bằng pipe. Liên quan: [[no-grep-tail]].
