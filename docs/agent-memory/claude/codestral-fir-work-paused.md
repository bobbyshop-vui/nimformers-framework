---
name: codestral-fir-work-paused
description: "Trạng thái việc Codestral FIR (Metal+Vulkan) khi hết usage 17/11/2026, chưa commit"
metadata:
  node_type: memory
  type: project
  originSessionId: 5fd80fc8-731a-4407-a69b-4d85d45cc4e7
  modified: 2026-11-17T11:48:49.432Z
---

Dừng ngày 17/11/2026 vì hết usage limit. Mục tiêu: Codestral FIR chạy đúng và nhanh bằng/hơn native trên Metal và Vulkan, Tensor Core tự chọn (Vulkan coopmatrix, Metal simdgroup matrix).

Đã xong, CHƯA commit (main = 232d071):
- Codestral lift khớp token native trên Vulkan và Metal. Graph trong models/codestral/lifted/ (Vulkan, chạy được cả Metal) và lifted-metal/, có các pha prefill 128/64/32/16/8/5 hàng.
- Bug prefill đã sửa (slice giữ wrapper micro:, SSD không lift được nên lúc ghi trace dùng scan micro-VM).
- Metal: sửa reset graph, replay sessionFp4GemmMetal; kernel Tensor Core hạ xuống Metal simdgroup (biên dịch qua framework OK, chưa chạy trên Apple Silicon).
- Tăng tốc: session node bind buffer thật (BYBY_FIR_SESSION_DIRECT), MatmulNT replay native, BYBY_PREFILL_MIN_ROWS=5 (native + FIR).
- 5 lỗi Codex review đã sửa; test_library, test_q4_selection, test_steps qua. Docs runtime-flags (EN+VI) đã cập nhật.

Còn dở:
- Đo lại cả bảng khi máy rảnh (số cuối bị chạy chồng): 13/22/139 token, native vs FIR, Vulkan+Metal.
- Prefill Metal vẫn chậm hơn native (3.49 vs 4.62 trên 139 token, số chưa sạch) → profile BYBY_FIR_PROFILE=2.
- Record/replay decode Metal: một agent Opus đang làm (ICB hoặc command-list replay, khởi từ metal_graph_* trong metal_shim.m) thì bị dừng; trạng thái file chưa rõ, phải build lại xem.
- Commit/push, cell Colab T4.

**How to apply:** trước tiên make formers-cli + build test để xem phần dở có biên dịch không; luật một lượt chạy nặng [[one-heavy-run-at-a-time]].
