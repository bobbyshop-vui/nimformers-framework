---
name: formers-n-framework-design
description: "formers.n nằm ở other/formers.n, code bằng .bybylang, chỉ import block và tokenizer"
metadata:
  node_type: memory
  type: project
  originSessionId: dda4b68f-9037-40a7-8307-40dea514d13a
  modified: 2026-11-11T13:33:41.208Z
---

Dự án formers.n có thư mục riêng, code bằng .bybylang. Chỉ import block và tokenizer có sẵn; phần điều phối tự viết lại.

**Why:** Quyết định thiết kế của user.

**How to apply:** Không viết điều phối bằng Nim thuần; đặt code mới trong other/formers.n.
