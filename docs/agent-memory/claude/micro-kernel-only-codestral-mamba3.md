---
name: micro-kernel-only-codestral-mamba3
description: Codestral và Mamba 3 mặc định chỉ dùng micro kernel; cấm kernel .metal/.glsl viết tay riêng backend (fp4km)
metadata:
  node_type: memory
  type: feedback
  originSessionId: 5fd80fc8-731a-4407-a69b-4d85d45cc4e7
  modified: 2026-11-18T06:24:22.699Z
---

Mọi kernel của Codestral và Mamba 3 (FP4-KM GEMV/GEMM, Tensor Core, FP16 Linear++) phải là micro kernel: chương trình micro-VM dựng trong Nim, được jit_vm hạ xuống MSL/GLSL/CPU. Nhánh Tensor Core (cooperative matrix trên Vulkan, simdgroup matrix trên Apple GPU) cũng phải do lowering sinh ra, không đọc template. Cấm tạo file kiểu gemm_fp4km.metal, matmul_fp4km_*.glsl, fir_fp4_*_micro.metal/.glsl.

**Why:** Ngày 18/11/2026 user phát hiện các file fp4km viết tay riêng Metal/Vulkan và rất giận: đã dặn phải micro kernel. Ý user là viết micro kernel mới dựa trên các kernel đó rồi thay thế, xoá mọi fp4km trong thư mục kernels.

**How to apply:** Việc chuyển đổi này user sẽ bảo làm sau; chỉ bắt đầu khi được nhắc. Khi làm: không chạy model 7B nếu user dặn máy nóng; kiểm bằng build + unit test nhỏ. Liên quan [[one-heavy-run-at-a-time]].
