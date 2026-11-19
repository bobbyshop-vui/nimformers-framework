---
name: trace-without-disk-files
description: "Trace bằng `firlift --run` (pipe, trong RAM), không tạo .fprobe/.data ra ổ cứng"
metadata:
  node_type: memory
  type: feedback
  originSessionId: dda4b68f-9037-40a7-8307-40dea514d13a
  modified: 2026-11-12T09:45:37.297Z
---

Khi trace/lift, dùng `firlift --config C --weights W --out DIR --run CMD...` (đặt `--run` cuối cùng). Recorder ghi qua `BYBY_FIR_PROBE=fd:T,D` vào pipe, lifter giữ trace và bytes upload trong RAM; chỉ `model.fir` và `trace.data` (đã nén) nằm trong `--out`.

**Why:** User phàn nàn máy hết GB vì probe data (`.fprobe` + `.fprobe.data` hàng trăm MB tới GB); yêu cầu 12/11 "trace mà ko tạo probe data ra ổ cứng".

**How to apply:** Không dùng `BYBY_FIR_PROBE=<đường dẫn file>` nữa. Binary phải build bằng recorder mới (probe.nim có `fd:`), nếu không nó ghi file tên `fd:T,D` ở thư mục hiện tại (kiểm tra và xoá bằng /bin/rm). Đặt `BYBY_FIR_PROBE_DATA_MAX` nếu RAM không đủ. Xem [[cleanup-trace-artifacts]], [[trace-must-be-core-nim]], [[keep-model-fir]].
