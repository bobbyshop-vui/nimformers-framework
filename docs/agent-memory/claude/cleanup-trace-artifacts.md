---
name: cleanup-trace-artifacts
description: "Máy hết GB: dọn chủ động và triệt để file tạm, probe/trace data, build test, worktree cheapag và thùng rác, không đợi tới cuối"
metadata:
  node_type: memory
  type: feedback
  originSessionId: dda4b68f-9037-40a7-8307-40dea514d13a
  modified: 2026-11-12T09:26:49.534Z
---

Dọn thường xuyên và triệt để, ngay khi xong từng bước: file tạm, probe/trace data (`.fprobe`, `.fprobe.data`), thư mục lift thử, binary test chưa theo dõi, nimcache của test, worktree cheapag, và phần của mình còn nằm trong `~/.Trash`. Không đợi tới cuối việc.

**Why:** User phàn nàn nhiều lần (11–12/11) rằng máy hết GB ("tốn hết gb máy tôi", "máy tốn gb lắm rồi, nên dọn nữa"). `rm` trong shell của harness chỉ chuyển vào `~/.Trash` nên xoá bằng `rm` không giải phóng gì; riêng scratchpad cũ và thư mục phiên cũ đã nằm đó ≈750 MB.

**How to apply:**
- Xoá bằng `/bin/rm` với đường dẫn tường minh, rồi xem `~/.Trash` và `/bin/rm` phần của mình (giữ mục của project khác như ielts-mobile).
- Ghi trace/probe: lift và so khớp xong thì xoá probe ngay; đo xong thì xoá thư mục lift thử, không để tới cuối phiên.
- Báo `df -h` trước và sau, liệt kê đúng những gì đã xoá.
- Chỗ giấu lớn nhất: `/private/var/folders/zn/kzx82bhs51qg9l_gk_wvm8fc0000gn/C/com.apple.gpuarchiver` (cache shader Metal, từng 73,7 GB, xoá an toàn bằng `/bin/rm -rf`, chạy nền vì lâu) cùng `com.apple.python3`, `org.python.python`. Kiểm bằng `du -sk` thư mục `C` khi máy đầy.
- Không xoá `models/*/model.fir` và `trace.data` đi kèm (xem [[keep-model-fir]]), không đụng model/checkpoint của user, cache của app khác (Codex, VSCode, Google, pip) hay `~/Library/Caches/nimformers` (JIT archive, tạo lại mất vài phút trên Intel); chỉ nêu chúng cho user nếu cần thêm chỗ.
