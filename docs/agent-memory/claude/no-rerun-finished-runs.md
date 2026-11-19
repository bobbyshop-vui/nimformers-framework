---
name: no-rerun-finished-runs
description: Không chạy lại lượt model/benchmark đã chạy xong chỉ để lọc hay xem lại output; dùng kết quả đã có
metadata:
  node_type: memory
  type: feedback
  originSessionId: dda4b68f-9037-40a7-8307-40dea514d13a
  modified: 2026-11-11T15:13:03.065Z
---

Một lượt chạy model xong thì dùng output của chính lượt đó. Không chạy lại chỉ để lọc, xem thêm hay kiểm tra lại.

**Why:** User không cho phép chạy lại một lượt inference đã chạy vì mất công và tốn thời gian, hành máy.

**How to apply:** Trước khi chạy model, xác định cần ghi những số hay đoạn nào rồi đọc ngay từ output lượt đó. Chỉ chạy lại sau khi đã đổi code và cần số mới, và chạy đúng một lần. Liên quan [[no-heavy-model-runs]] và [[no-grep-tail]].
