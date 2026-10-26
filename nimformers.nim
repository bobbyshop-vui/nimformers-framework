## nimformers.nim - ĐIỂM VÀO DUY NHẤT của framework.
##
## TRƯỚC ĐÂY: mỗi file dùng framework phải tự liệt kê đủ 5 import riêng lẻ
## (thứ tự PHẢI đúng vì có phụ thuộc chéo: backend <- bybylang,
## nimformer <- backend + quant + customfloat, tensor_ops <- backend + nimformer):
##
##   import nimformer
##   import backend
##   import customfloat
##   import quant
##   import tensor_ops
##
## Quên 1 trong 5 (hay quên `export` đúng) là lỗi "undefined identifier"
## mù mờ, khó dò nếu không quen code. GIỜ: chỉ cần
##
##   import nimformers
##
## là có đủ Tensor/Backend/QuantTensor/BybyLang exports (sessionMatmulQ4,
## GpuBackend, v.v. - toàn bộ danh sách export đã có sẵn trong backend.nim),
## toàn bộ hàm dựng model của nimformer.nim/customfloat.nim/quant.nim, VÀ
## các phép tensor bổ sung (view/permute/reduction/concat/split/stack/
## dropout/comparison/cumsum/cumprod/scan - tensor_ops.nim, dùng nhiều cho
## Mamba/SSM).
##
## KHÔNG đổi/xoá 5 file gốc (nimformer.nim/backend.nim/customfloat.nim/
## quant.nim/tensor_ops.nim) - các file cũ đang import riêng lẻ
## (nim_inference.nim, finetune_lora.nim, graft_attention.nim, v.v.) VẪN
## chạy đúng y hệt, không bắt buộc phải đổi sang "import nimformers" ngay.
## Đây là facade CỘNG THÊM, không phải thay thế bắt buộc.
##
## "if else chọn backend" nằm sẵn trong backend.nim (resolveBackendKind,
## detectBackend qua BybyLang) - nimformers.nim KHÔNG lặp lại logic đó,
## chỉ export lại để người dùng gọi newBackend() ngay sau khi
## `import nimformers` mà không cần biết backend.nim tồn tại riêng.
##
## LƯU Ý VỀ CÁC HÀM KERNEL (beMatmul/beMatmulQ4/beAdd/beExp/beSoftmax/
## beAttentionFused/beResidentUploadQ4/... - 43 proc public tính đến hiện
## tại): KHÔNG cần liệt kê thủ công trong file này. `export backend`
## (dòng dưới) đã tự động re-export TOÀN BỘ API public của backend.nim
## theo đúng ngữ nghĩa `export <module đã import>` chuẩn của Nim - bao
## gồm cả khối `export byby.X, byby.Y, ...` (re-export API BybyLang) LẪN
## mọi proc `beXxx*` định nghĩa ngay trong backend.nim, không phải chỉ
## danh sách con nào. Thêm hàm kernel mới vào backend.nim (đánh dấu `*`)
## là tự động lộ ra qua `import nimformers`, không cần sửa file này.

import nimformer
import backend
import customfloat
import quant
import tensor_ops

export nimformer
export backend
export customfloat
export quant
export tensor_ops

# backend.nim re-export một phần API từ vendor/bybylang/gpubackend.
# Nim không tự truyền tiếp các symbol imported-and-reexported đó qua
# `export backend`, vì vậy facade phải expose chúng tường minh.
export backend.GpuBackend, backend.SessionHandle, backend.ResidentWeightHandle
export backend.sessionBegin, backend.sessionUpload, backend.sessionUploadIndices,
       backend.sessionRope, backend.sessionAllocScratch, backend.sessionMatmul,
       backend.sessionMatmulNT, backend.sessionSoftmax, backend.sessionVecOp,
       backend.sessionActivation, backend.sessionEnd, backend.sessionAllocHeap,
       backend.sessionAllocFromHeap, backend.sessionFreeHeap, backend.sessionRead,
       backend.sessionFree, backend.sessionCopyInto, backend.sessionScale, backend.residentUploadQ4,
       backend.sessionMatmulQ4, backend.residentFreeQ4, backend.sessionLayernormEnc,
       backend.sessionRmsnormEnc, backend.sessionAttentionFusedEnc,
       backend.sessionAttentionDecodeGqaEnc, backend.cpuAttentionDecodeGqa,
       backend.gpuGather, backend.gpuArgmaxIndex, backend.gpuCompare,
       backend.gpuDropoutMask, backend.gpuPermute, backend.gpuSplitSlice,
       backend.gpuConcat

# ── micro-VM / JIT (vendor/bybylang/micro_kernel) ──────────────────────────
# session_vm + session_vm_dispatch + jit_vm đã chuyển khỏi root vào
# vendor/bybylang/micro_kernel/. App-level chỉ cần `import nimformers` là
# có VmProgram / encodeVm / JIT cache; không import trực tiếp 3 file cũ.
# (Root session_vm.nim / jit_vm.nim / session_vm_dispatch.nim vẫn là shim
# tương thích ngược nếu code cũ còn `import session_vm`.)
import vendor/bybylang/micro_kernel
export micro_kernel
