# Backend CPU

`vendor/bybylang/backends/cpu/` cài đặt backend không cần GPU cho BybyLang và
Nimformers. Nó có cùng các phép tức thời, session API, resident weights và chạy
bytecode micro-VM như backend Metal/Vulkan.

## File

| File | Vai trò |
| --- | --- |
| `cpu_kernels.h` | C ABI mà wrapper Nim gọi |
| `cpu_kernels.c` | dò tính năng CPU, bảng dispatch theo ISA, chia block theo cache |
| `cpu_kernels_impl.inc` | thân kernel, include một lần cho mỗi ISA |
| `cpu_vm.c` | trình thông dịch micro-VM chạy lockstep (thanh ghi R/U, barrier threadgroup/grid) |
| `cpu_backend.nim` | wrapper Nim, chuyển dữ liệu seq, `cpuInfo()` / `cpuFeatures()` |

## Dispatch ISA lúc chạy

Constructor C dò CPU một lần và dựng bảng hàm cho từng ISA:

| ISA | Cách build |
| --- | --- |
| scalar | vòng lặp C cơ bản |
| SSE4.1 | `__attribute__((target("sse4.1")))` |
| AVX | `target("avx")` |
| AVX2 + FMA | `target("avx2,fma")` |
| AVX-512F | `target("avx512f")` |
| NEON | NEON AArch64/ARM (target mặc định) |

Mã nguồn dùng vòng lặp portable; Clang/GCC tự vector hóa từng hàm có đánh dấu
target. Disassembly của object sinh ra có `vaddps`/`vfmadd` với thanh ghi
`ymm`/`zmm`, nên việc chọn ISA là thật, không chỉ trên danh nghĩa.

### Chính sách cho phép nhỏ (AVX frequency offset)

Trên Intel, khi vào code AVX/AVX-512 rộng, xung nhân có thể bị hạ trong một
khoảng ngắn ("AVX offset"). Vì vậy việc elementwise nhỏ dùng SSE4/NEON:

```
n <= NIMFORMERS_CPU_SMALL (mặc định 4096)          -> SSE4 / NEON
elementwise lớn hơn                                -> AVX2+FMA (hoặc AVX)
work >= NIMFORMERS_CPU_AVX512_MIN (mặc định 131072) -> AVX-512F nếu có
```

Matmul dùng ước lượng công việc riêng (`M*K*N`): tích nhỏ ở lại SSE, tích lớn
dùng AVX2/AVX-512. Ngưỡng được chọn để chi phí khởi tạo vector và chuyển xung
không lấn át các kernel tí hon.

### Chia block matmul theo L3

Dispatcher đọc kích thước L1/L2/L3 và chọn tile `NC/KC/MC`:

- panel B đã đóng gói (`KC x NC`) có kích thước để được dùng lại từ L3 qua các
  block hàng,
- tile C (`MC x NC`) nằm gọn trong L2,
- vòng trong là axpy liên tục `C[i,:] += A[i,k] * B[k,:]`, tự vector hóa với
  FMA.

Ghi đè bằng `NIMFORMERS_CPU_BLOCK_NC`, `NIMFORMERS_CPU_BLOCK_KC` và
`NIMFORMERS_CPU_BLOCK_MC`.

## Biến môi trường

| Biến | Ý nghĩa |
| --- | --- |
| `NIMFORMERS_CPU_ISA` | `scalar`, `sse`, `avx`, `avx2`, `avx512`, hoặc để trống/auto |
| `NIMFORMERS_CPU_SMALL` | ngưỡng số phần tử để ưu tiên SSE |
| `NIMFORMERS_CPU_AVX512_MIN` | ngưỡng công việc trước khi xét AVX-512 |
| `NIMFORMERS_CPU_BLOCK_NC` / `_KC` / `_MC` | tile cache của matmul |

Xem giá trị dò được từ Nim hoặc Python:

```nim
import vendor/bybylang/backends/cpu/cpu_backend as cpu
echo cpu.cpuInfo()
# (isa: 3, isaName: "avx2", hasSse4: true, ..., l3Bytes: ...)
```

```python
from nimformers import cpu_info
print(cpu_info())
# {'isa': 3, 'isa_name': 'avx2', ..., 'block_mc': 32, 'block_nc': 256, 'block_kc': 128}
```

## Phần được hỗ trợ

- elementwise add/sub/mul/div, phép với scalar, relu/sqrt/exp/log/sigmoid/tanh,
  pow, APFLU + backward, add+activation fuse
- matmul chia block theo cache, matmul-NT, matmul int4 (Q4)
- softmax, layernorm, RMSNorm, layernorm backward
- attention causal fuse (forward), attention backward
- embedding, gather, compare, dropout mask, argmax, permute, split, concat,
  RoPE
- session upload/read/free, các phép session, resident weights FP32/Q4
- micro-VM: số học cơ bản, load/store, LUT, vòng lặp/if, phép địa chỉ thanh ghi
  U, bộ nhớ threadgroup, barrier threadgroup và grid

## Giới hạn

- Kernel CPU chạy đơn luồng. Matmul có chia block theo cache và SIMD nhưng chưa
  dùng OpenMP hay nhiều nhân.
- AVX-512 chỉ được dùng khi CPU có `avx512f`; CPU Intel trên macOS thường không
  có.
- Micro-VM cho các thread tiến lockstep. Cách này giữ đúng ngữ nghĩa barrier
  nhưng chậm hơn dispatch GPU rất nhiều; nó dành cho tính đúng đắn trên máy
  không có GPU, không phải cho throughput.
- Opcode virtual-threadgroup (`VTG_*`) chạy như một threadgroup bình thường;
  `LOAD_VACT` luôn báo active.

## Test

```bash
make bybylang-test        # có cả test/bybylang_test/cpu_test.bybylang
pytest -q test/python_test/test_cpu_backend.py
```

Bộ test CPU Python bao gồm toán SIMD, softmax/layernorm/attention, Q4,
resident weights, micro-VM CPU và một lần sinh Llama tí hon.

## Benchmark

Đo GFLOPS matmul FP32 trên các backend CPU, Metal và Vulkan:

```bash
python3 benchmarks/bench_gflops.py
# hoặc: make bench-gflops
python3 benchmarks/bench_gflops.py --backends cpu,metal,vulkan \
    --sizes 256,512,1024 --iters 5

# Một dispatch 8096^3 có thể vượt watchdog command buffer của Metal/MoltenVK
# trên GPU tích hợp (iGPU ở đây timeout khi lớn hơn 6144^3).
# Chế độ tile tính cùng 2*N^3 FLOP bằng các dispatch tile ngắn:
python3 benchmarks/bench_gflops.py --backends metal,vulkan \
    --sizes 8096 --iters 1 --warmup 0 --tile 1024

# cùng preset, lệnh ngắn nhất:
python3 benchmarks/bench_gflops.py --gpu-8096
```

Backend không có sẽ báo `SKIP`; có thể xuất kết quả bằng `--csv results.csv`.
Script báo cả mẫu tốt nhất và trung vị, cùng ISA/tile cache CPU đang dùng.
Fallback CPU của GPU bị tắt mặc định để một kernel lỗi không bị đo nhầm thành
CPU; chỉ dùng `--allow-cpu-fallback` khi thật sự muốn vậy.
