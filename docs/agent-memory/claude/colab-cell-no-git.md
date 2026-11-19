---
name: colab-cell-no-git
description: "Cell Colab đưa cho user chỉ làm đúng một việc; cấm nhét git fetch/checkout/submodule vào, giữ trật tự notebook"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 9ed0419b-f915-48d3-9415-218a58f696b3
  modified: 2026-11-15T14:13:52.169Z
---

Cell Colab đưa trong chat chỉ làm đúng việc của nó (chạy graph, lift...). Không nhét git fetch/checkout/submodule hay bước cập nhật repo vào. Việc cập nhật và build đã có cell riêng trong notebook (cell 3 clone/update, cell build).

**Why:** Ngày 15/11/2026 tôi đưa một cell chạy FIR có kèm git fetch/checkout/submodule và make. User nổi giận: "đừng bao gồm git vào… phá trật tự notebook".

**How to apply:** Chỉ đưa cell chạy. Nếu cần code mới thì bảo user chạy lại cell cập nhật và cell build có sẵn của notebook. Xem thêm [[no-notebook-edits]].
