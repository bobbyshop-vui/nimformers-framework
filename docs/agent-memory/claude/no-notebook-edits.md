---
name: no-notebook-edits
description: Không tự sửa notebook (.ipynb); đưa code cell trong chat để user tự dán
metadata:
  node_type: memory
  type: feedback
  originSessionId: 9ed0419b-f915-48d3-9415-218a58f696b3
  modified: 2026-11-19T05:41:12.349Z
---

Không sửa file notebook (ví dụ notebooks/deepseek_colab_vulkan.ipynb), kể cả sửa nhỏ như `_ = run_streaming(...)`.

**Why:** Ngày 15/11/2026 user ngắt giữa chừng và nói "ko đc sửa notebook" khi tôi sửa cell lift/run. Notebook thường đang có output Colab chưa commit của user.

**How to apply:** Muốn đổi notebook thì đưa code cell trong chat cho user tự dán vào Colab. Ngoại lệ: khi user bảo rõ "sửa vào file notebook" (19/11/2026) thì sửa thẳng .ipynb, giữ id/metadata cell và output của cell không đổi code. Chỉ sửa code nguồn (Nim/bybylang). Nếu lỡ sửa thì hoàn nguyên đúng nội dung của user, đừng `git checkout`. Liên quan [[commit-include-notebooks]].
