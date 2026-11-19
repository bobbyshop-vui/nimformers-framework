---
name: lifted-fir-into-project
description: "model.fir đã lift phải đặt trong thư mục project (models/<model>/lifted/) để user tự chạy, không để ở scratchpad"
metadata:
  node_type: memory
  type: feedback
  originSessionId: dda4b68f-9037-40a7-8307-40dea514d13a
  modified: 2026-11-13T08:02:57.858Z
---

Khi lift xong một model, copy `model.fir` (và `config.json`) vào thư mục project, ví dụ `models/tinyllama/lifted/model.fir` (GPU Apple) và `model_intel.fir` (GPU Intel), rồi đưa lệnh chạy cho cả Mac M1 (`formers-cli_arm64`) và Mac Intel (`formers-cli`).

**Why:** user muốn tự chạy inference với model.fir trên cả hai máy; file nằm ở scratchpad thì user không dùng được và bị xoá khi dọn.

**How to apply:** scratchpad chỉ dùng cho lượt lift thử; bản cuối luôn nằm trong project. Build lại cả hai binary CLI khi runner thay đổi. Liên quan: [[keep-model-fir]], [[cleanup-trace-artifacts]].
