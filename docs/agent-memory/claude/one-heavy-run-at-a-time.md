---
name: one-heavy-run-at-a-time
description: Không bao giờ để hai sub-agent cùng chạy firlift hay inference; agent sau phải đợi agent trước xong
metadata:
  node_type: memory
  type: feedback
  originSessionId: 5fd80fc8-731a-4407-a69b-4d85d45cc4e7
  modified: 2026-11-17T10:20:21.989Z
---

Chỉ một tiến trình nặng (firlift --run, formers-cli inference/--fir, model 7B) chạy tại một thời điểm trên máy, tính chung cho mọi sub-agent (Claude lẫn Codex). Agent nào muốn chạy mà thấy `ps` đang có formers-cli/firlift của agent khác thì phải đợi nó xong rồi mới chạy.

**Why:** Ngày 17/11/2026 user dặn: chạy hai cái cùng lúc làm lag máy (16 GB RAM, model 7B đã swap nặng).

**How to apply:** Ghi luật này vào prompt của mọi sub-agent có thể chạy model; chỉ giao việc chạy model cho một agent, các agent khác làm code/test/docs. Xem [[no-heavy-model-runs]].
