---
name: commit-user-pending-changes
description: "Commit kèm file user sửa dở, NGOẠI TRỪ test/deepseek-inference/benchmark.md: cấm commit file đó"
metadata:
  node_type: memory
  type: feedback
  originSessionId: 9ed0419b-f915-48d3-9415-218a58f696b3
  modified: 2026-11-15T15:26:50.270Z
---

Khi user bảo commit, đưa vào commit cả các file tracked mà user đang sửa, không chỉ file tôi sửa. Chỉ để ngoài binary/artefact build chưa track.

NGOẠI LỆ: cấm commit test/deepseek-inference/benchmark.md (bản sửa benchmark DeepSeek) và cấm commit nào tên kiểu "Update DeepSeek benchmark". Giữ nó là thay đổi chưa commit trong working tree.

**Why:** Ngày 15/11/2026 user lúc đầu giận vì benchmark.md bị bỏ khỏi commit. Tôi commit nó thành a2a6b86 "Update DeepSeek benchmark". Sau đó user nổi giận hơn ("cấm commit deepseek benchmark edit… cấm dìm hàng tao"), bắt xoá commit đó khỏi lịch sử main và force-push lại. Tôi đã drop nó bằng rebase, main mới là a946283.

**How to apply:** Trước khi commit, xem git status; add mọi file tracked đã modified (kể cả notebook, xem [[commit-include-notebooks]]), trừ test/deepseek-inference/benchmark.md. Không bao giờ tự commit benchmark DeepSeek trừ khi user nói rõ ngay lúc đó.

"Commit và push tất cả" nghĩa là tất cả việc tôi đã làm, KHÔNG gồm benchmark.md và cũng không gồm file thực thi (binary build như firlift, test_*). Cũng cấm đặt tên commit kiểu "Update benchmark"; tên commit phải nói về việc tôi đã làm.

Ngày 15/11/2026 tôi hiểu sai "tất cả" và commit cả benchmark.md lẫn binary. User nói "cấm push execute", rồi giận vì commit tên "Update benchmark" ("đặt update benchmark là ý gì… xoá ngay"). Tôi viết lại main: d6ceb0c "Ignore firlift binaries" chỉ chứa .gitignore, còn benchmark.md để lại chưa commit.
