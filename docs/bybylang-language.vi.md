# BybyLang

[English](bybylang-language.md) | [Tiếng Việt](bybylang-language.vi.md)

Đây là bản song ngữ được theo dõi trong framework của [README BybyLang vendored](../vendor/bybylang/readme.md). Ví dụ micro-kernel nhiều group tính chỉ số toàn cục từ group/local ID.

DSL viết bằng Nim, compile AOT sang source Nim rồi thành binary native.
Lệnh `gpu ...` tạo phép toán tensor; `session:` gom các phép được hỗ trợ;
`micro kernel` định nghĩa kernel VM BybyLang native để lower sang code GPU
Metal/Vulkan. Ngôn ngữ còn có điều khiển luồng và lệnh mô phỏng phần cứng.


## Build và chạy

```bash
make -C vendor/bybylang build
make -C vendor/bybylang test
# Or, from vendor/bybylang itself:
make build
make test
# equivalent to the build and AOT run:
nim c -d:release -o:bybylang bybylang.nim
./bybylang demo/demo_gpu.bybylang --aot=demo/demo_gpu_out
./demo/demo_gpu_out
```

`--aot=<path>` tạo `<path>.nim`, compile bằng Nim và xuất binary `<path>`.

---


## 1. Cú pháp chung

- Mỗi dòng `.bybylang` là một command. Thụt dòng bằng space/tab xác định
  block `if/elif/else/while/for`.
- Dòng bắt đầu bằng `#` là comment.
- `import name` hoặc `import "path/name.bybylang"`: import đệ quy, tự thêm
  đuôi `.bybylang`, chống vòng lặp bằng đường dẫn tuyệt đối.
- `print <expr>` sinh `echo <expr>`.
- `function NAME` mở định nghĩa, thân là các dòng thụt sâu hơn; kết thúc
  bằng dedent như `if/while/for`, không cần marker đóng. Gọi bằng `call NAME`
  hoặc một dòng chỉ chứa NAME.
- `if cond:` / `elif cond:` / `else:` / `while cond:` /
  `for x in range(a, b):` chuyển thẳng sang Nim, có thể lồng nhau.
- `mode is N`, N từ 1 đến 4, in "Mode 1: Low-level", "Mode 2: Mid-level",
  "Mode 3: High-level", "Mode 4: Web-level"; N khác in "Unknown mode".


## 2. Lệnh `gpu ...`


### Chọn backend

```
gpu backend is "auto"      # auto | cpu | cuda (Vulkan alias) | metal | vulkan
```

`auto` dò Metal trên macOS, tiếp đến Vulkan rồi CPU. `cuda` là tên cũ
chuyển sang Vulkan; bản này chưa có CUDA/cuBLAS. `opencl` không phải lựa
chọn hiện tại của `GpuBackend`.

`gForbidCpuFallback` mặc định `false`. Gọi `setForbidCpuFallback(true)` để
GPU-only báo lỗi thay vì cho phép từng phép fallback CPU. Backend và độ
phủ phép toán khác nhau; kiểm tra `bool` của encoder session và xử lý lỗi
`sessionEnd`.


### Khai báo array

```
gpu array A = [1, 2, 3, 4]
```

Sinh `var A: seq[float32] = @[1, 2, 3, 4].mapIt(it.float32)`.
**Phải khai báo trước khi dùng**; codegen không hoist mà dịch theo thứ tự file.


### Phép hai ngôi cơ bản

```
gpu add A, B -> C
gpu sub A, B -> C
gpu mul A, B -> C
gpu div A, B -> C
```

`size N` cuối dòng chỉ để mô tả, không ảnh hưởng codegen ngoài session.


### Nhân ma trận

```
gpu matmul A, B -> C m M k K n N
```

A có shape `[M×K]`, B `[K×N]`, kết quả C `[M×N]`.

`gpuMatmul2` chạy hai phép độc lập trong một lần gọi backend từ Nim, nhưng
chưa có cú pháp `.bybylang`; gọi trực tiếp `gpuMatmul2(...)` từ Nim.


### Activation

```
gpu relu X -> Y
gpu sigmoid X -> Y
gpu tanh X -> Y
gpu apflu X -> Y alpha A beta B      # alpha/beta optional; defaults to alpha=0.1, beta=1.0 if omitted
```

`apflu_backward` và `fused_add_act` đều có cú pháp DSL và API Nim
`gpuApfluBackward`/`gpuFusedAddAct`.


### Softmax và LayerNorm

```
gpu softmax X -> Y rows R cols C
gpu layernorm X, GAMMA, BETA -> Y rows R cols C eps E
gpu layernorm_backward DY, X, GAMMA, BETA -> DX, DGAMMA, DBETA rows R cols C eps E
```

### Embedding

```
gpu embedding TABLE, INDICES -> Y vocab V dim D
```

`INDICES` được tự đổi sang `int32`.


### Attention gộp

```
gpu attention Q, K, V -> O, S_MATRIX B H S D scale SCALE
gpu attention_backward Q, K, V, S_MATRIX, DY -> DQ, DK, DV B H S D scale SCALE
```

B là batch, H là số head, S là sequence length, D là head dimension.

### RoPE

```
gpu rope X, COS, SIN -> Y seqlen T heads H dim D
```

Xoay X tại chỗ bằng bảng COS/SIN. Dùng trong `session:`; chưa có
implementation immediate `gpu rope`. Nếu không mở được session, code sinh
ra cảnh báo rồi trả input nguyên vẹn, nên không được coi nhánh đó đã tính RoPE.


### Phép DSL khác

Parser còn hỗ trợ `sqrt`, `exp`, `log`, `pow`, `gather`, `compare`,
`fused_add_act`, `apflu_backward`, `permute`, `split_slice`, `concat`,
`argmax_index`, `dropout_mask`. Cú pháp toán hạng/shape nằm trong
[`demo/demo_gpu.bybylang`](../vendor/bybylang/demo/demo_gpu.bybylang).
`compare` trả mask boolean; chỉ số gather/embedding được đổi sang int32.


### Phép tổng quát và fallback

```
gpu <other_op> A, B -> C
```

Sinh `gpuOp("<other_op>", backend, A, B)`; phải có nhánh tương ứng trong
`gpuOp` để thực hiện phép đó.

---


## 3. Block `session:` — gom nhiều lệnh GPU

```
session:
  gpu add A, B -> C size 4
  gpu mul C, A -> D size 4
  gpu relu D -> E
  gpu rope Q, COS, SIN -> Q2 seqlen 4 heads 1 dim 4
```

Các dòng GPU được bọc bằng Session API (`sessionBegin`, `sessionUpload`,
`session<Op>`, `sessionEnd`, `sessionRead`, `sessionFree` trong
`gpubackend.nim`) để kết quả trung gian ở lại GPU, tránh đọc về
`seq[float32]` sau mỗi lần gọi.

- Các phép có đường encode session gồm `add`/`sub`/`mul`/`div`, activation,
  `softmax`, `matmul`, `layernorm`, `rope`, `fused_add_act`,
  `apflu`/`apflu_backward`, `gather`, `compare`, `dropout_mask`,
  `split_slice`. Các phép `embedding`, `attention`, `attention_backward`,
  `layernorm_backward`, `permute`, `concat`, `argmax_index` trong block vẫn
  chạy immediate API, không trở thành phép resident session.
- Khác bên ngoài session, `size N` ở đây **có** ảnh hưởng codegen: đó là
  số phần tử truyền cho phép session.
- Nếu `sessionBegin()` thất bại, block dùng immediate cho phép có fallback.
  `gpu rope` không có immediate; nhánh này cảnh báo và giữ output bằng
  input, tức bỏ qua RoPE.
- Dòng không phải `gpu` trong block như assignment/print vẫn chạy thông
  thường, ngoài phần batching GPU.

---


## 4. API GPU và session từ Nim

Import backend trực tiếp từ ứng dụng Nim, chạy từ thư mục gốc project:

```nim
import vendor/bybylang/gpubackend

let backend = parseBackend("vulkan") # also "metal", "auto", "cpu"
let a = @[1'f32, 2, 3, 4]
let b = @[10'f32, 20, 30, 40]
let c = gpuOp("add", backend, a, b)
echo c
```

Import `vendor/bybylang/bybylang` để dùng facade compiler DSL và helper VM;
module export lại `gpubackend`, `micro_kernel`, Metal/Vulkan. Chỉ dùng
backend thì import `vendor/bybylang/gpubackend` để giảm bề mặt import.

Immediate API `gpu*` nhận/trả `seq` CPU, nên mỗi lần gọi có thể truyền dữ
liệu. Nhóm API gồm `gpuMatmul`, `gpuMatmul2`, `gpuMatmulQ4`, vector ops
(`gpuRelu`, `gpuSigmoid`, `gpuTanh`, `gpuFusedAddAct`), softmax, LayerNorm
và backward, fused attention và backward, embedding, session RoPE cùng
các tiện ích gather/compare/permute/split/concat/argmax/dropout. Xem các
khai báo export trong `gpubackend.nim` để biết kích thước/thứ tự đối số.

Dùng `SessionHandle` cho chuỗi phép GPU để giữ tensor trung gian trên
thiết bị. Luôn kiểm tra giá trị encoder và kết thúc session trước khi đọc:

```nim
import vendor/bybylang/gpubackend

let backend = parseBackend("vulkan")
if not sessionBegin(backend):
  raise newException(IOError, "GPU sessions are unavailable")
var a = sessionUpload(backend, @[1'f32, 2, 3, 4])
var b = sessionUpload(backend, @[10'f32, 20, 30, 40])
var c: SessionHandle
if not sessionVecOp(backend, "add", a, b, c, 4):
  raise newException(IOError, "add encode failed")
if not sessionEnd(backend):
  raise newException(IOError, sessionGetLastError(backend))
echo sessionRead(backend, c, 4)
sessionFree(backend, a)
sessionFree(backend, b)
sessionFree(backend, c)
```

Vòng đời chính gồm `sessionBegin`,
`sessionUpload`/`sessionUploadBorrowed`/`sessionUploadPrivate`,
`sessionAllocScratch`, `sessionEnd`, `sessionRead`, `sessionFree`. Encoder
bao gồm vector/activation, matmul/Q4, softmax, LayerNorm/RMSNorm, RoPE,
attention, copy/scale, gather/compare, dropout, slice, embedding khi
backend có implementation. `sessionCommitContinue` commit công việc rồi
tiếp tục encode mà không cần đọc CPU; `sessionRecoverQueue` phục hồi sau
session thất bại. `sessionAllocHeap` hiện chỉ tối ưu heap trên Metal và
trả nil ở backend khác; kiểm tra trước khi suballocate.

Với Q4, upload một lần rồi dùng lại resident handle qua các session:

```nim
var q4 = residentUploadQ4(backend, packedWeights, scales, zeros,
                          k = 4096, n = 4096, groupSize = 32,
                          nGroupsPerRow = 128)
discard residentPrepareQ4(q4, M = 1) # optional autotune; false means no tuned plan
if not sessionBegin(backend): raise newException(IOError, "session unavailable")
var x = sessionUpload(backend, activations)
var y: SessionHandle
if not sessionMatmulQ4(backend, x, q4, y, M = 1):
  raise newException(IOError, "Q4 matmul encode failed")
if not sessionEnd(backend): raise newException(IOError, sessionGetLastError(backend))
let result = sessionRead(backend, y, 4096)
sessionFree(backend, x)
sessionFree(backend, y)
residentFreeQ4(q4) # call only when unloading the weights
```

Tên resident tensor `cu*` cũ chỉ là stub tương thích ở phiên bản này,
không cung cấp CUDA/cuBLAS. Dùng Session API chung. Tên backend `cuda`
chuyển sang Vulkan.


## 5. Micro-kernel BybyLang native

Viết block `micro kernel` trực tiếp trong `.bybylang`, không cần module
Nim helper hay import micro-kernel ngoài. Compiler tạo `VmProgram`, lower
sang Metal/Vulkan, dispatch trong session GPU rồi đọc output:

```text
gpu backend is "vulkan"
gpu array INPUT = [1, 2, 3, 4, 5, 6, 7, 8]

micro kernel scale INPUT -> OUTPUT threads 8 tg 4 out 8:
  emitUTgid 0
  emitUMulK 0 0 4
  emitULid 1
  emitUAdd 0 0 1
  emitULoadIn 1 0 0
  emitLoadConstF32 2 2.0'f32
  emitMul 1 1 2
  emitUStoreF32 1 0 0
  emitHalt

print OUTPUT
```

API instruction VM được export bởi `vendor/bybylang/micro_kernel` và
export lại bởi `vendor/bybylang/bybylang`: tạo `VmProgram`, thêm lệnh qua
`emit*`. Lệnh gồm load/store float và integer, số học, FMA, loop/condition,
ID thread/group, bộ nhớ group và barrier, atomic, helper virtual group.
Thanh ghi integer 32-bit; thanh ghi float là float32. Mỗi lệnh nhận chỉ số
thanh ghi/offset theo khai báo trong `micro_kernel/session_vm.nim`.

Header DSL là `micro kernel NAME INPUT -> OUTPUT threads N tg N out N:`;
thân thụt dòng nhận các lệnh `emit...`. `threads` là tổng thread dispatch,
`tg` là số thread/group, `out` là số phần tử output. Chọn kích thước và
chặn truy cập để mọi thread ở trong giới hạn. Nhánh này cần Metal/Vulkan,
báo lỗi với CPU/backend không hỗ trợ. Kernel phức tạp có thể tạo
`VmProgram` từ Nim sau khi import `vendor/bybylang/bybylang`, rồi gọi
`executeMicroKernel(backend, kernelName, program, input, outputElems, threads, tgSize)`.

---


## 6. Lệnh mô phỏng phần cứng cấp thấp

Các lệnh sau **có cú pháp DSL thực**, được parse trong `genBlock` của
`bybylang.nim` (vùng dòng 668–714 của bản nguồn được mô tả) rồi lower thành
lệnh thao tác RAM mô phỏng 1024 số nguyên, BUS `seq[string]` và 32 pin:

```
apu tran "chip1" with 101010          # -> apuTran("chip1", 101010)
apu mem write RAM0 with 42            # -> apuMem("write", "RAM0", "42")
apu mem read RAM0 with 0              # -> apuMem("read", "RAM0", "0")
apu core run                          # -> apuCore(1, "run")  (mode is always hardcoded to 1)
apu pin 3 is high                     # -> apuPin(3, "high")
bit send 1010                         # -> bitSend("1010")
bit recv                              # -> bitRecv()
mem map "device0"                     # -> memMap("device0")
mem push RAM0 with 99                 # -> memPush("RAM0", "99")
tran pulse pin 3 width 10ns           # -> tranPulse(3, "10ns")
```

Các điểm cần biết về parser:

- `apu mem <action> <target> with <value>`: target là từ thứ tư
  (`left[3]`), bỏ dấu nháy; action nên là `write` hoặc `read`.
- `apu core ...` bỏ qua phần sau `apu core`, luôn sinh `apuCore(1,"run")`.
- `apu pin <n> is <state>` lấy n tại từ index 2, state tại index 4; thừa
  hoặc thiếu từ có thể bị parse sai mà không báo lỗi.
- `tran pulse pin <n> width <w>` lấy n ở từ index 3 và w ở **từ cuối** dòng.

Các cảnh báo "declared but not used" của `apuTran`, `apuMem`, `apuCore`,
`apuPin`, `bitSend`, `bitRecv`, `memMap`, `memPush`, `tranPulse` khi compile
compiler `bybylang.nim` không có nghĩa DSL không chạy. Output AOT khai
báo bản riêng của các proc này (vùng dòng 775 trở đi trong bản nguồn
được mô tả) để chương trình compile gọi được chúng lúc runtime.

---


## 7. Backend

| Backend | File | Ghi chú |
| --- | --- | --- |
| Metal | `backends/metal/metal_backend.nim`, `backends/metal/kernels/` | GPU macOS, buffer/pipeline/session native |
| Vulkan | `backends/vulkan/vulkan_driver.nim`, `backends/vulkan/kernels/` | NVIDIA/AMD/Intel khi driver và tính năng hỗ trợ phép toán |
| CPU | phép tham chiếu trong `gpubackend.nim` | fallback hoặc implementation CPU tường minh khi có; có thể cấm bằng `setForbidCpuFallback(true)` |

`cuda` là tên tương thích được chuẩn hóa sang Vulkan, không load CUDA
Runtime/cuBLAS. OpenCL không phải backend chọn được trong phiên bản hiện tại.


## 8. Ví dụ tối thiểu

```
gpu backend is "vulkan"
gpu array A = [1, 2, 3, 4]
gpu array B = [5, 6, 7, 8]
gpu add A, B -> C
print C
```
