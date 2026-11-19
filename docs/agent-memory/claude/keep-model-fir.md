---
name: keep-model-fir
description: Khi dọn rác không xoá model.fir và trace.data đi kèm
metadata:
  node_type: memory
  type: feedback
  originSessionId: dda4b68f-9037-40a7-8307-40dea514d13a
  modified: 2026-11-11T13:34:00.000Z
---

Giữ mọi models/*/model.fir đã lift cùng trace.data đi kèm khi dọn rác.

**Why:** User dặn; model.fir cần trace.data để chạy.

**How to apply:** Trước khi rm, liệt kê find models -name model.fir và loại khỏi danh sách xoá.
