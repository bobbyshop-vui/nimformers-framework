# Nimformer Framework

Nimformer là framework transformer thuần **Nim**, hỗ trợ đa backend (CPU, Metal, CUDA), lượng tử hóa thích ứng (int8/int4/fp8/APF) và checkpoint nén `.nimq`.

## 1. Cấu trúc dự án

```
LICENSE                         export_to_nimq.py               nim_inference.nim               test_nimformer.nim
Makefile                        finetune.nimq.ckpt              nimformer.nim                   tokenizer.json
backend.nim                     harness_attn.nim                quant.nim                       vendor
customfloat.nim                 harness_attn2.nim               readme.md
databricks-dolly-15k.jsonl      metal_shim.m                    test_nimformer.exe

./vendor:
bybylang

./vendor/bybylang:
LICENSE         backends        bybylang.nim    demo            gpubackend.nim  makefile        readme.md       tsic_ir.nim

./vendor/bybylang/backends:
cuda    metal   opencl

./vendor/bybylang/backends/cuda:
cuda_driver.nim         cuda_runtime.nim        kernels

./vendor/bybylang/backends/cuda/kernels:
vecop.ptx

./vendor/bybylang/backends/metal:
kernels                 metal_backend.nim       metal_shim.h            metal_shim.m

./vendor/bybylang/backends/metal/kernels:
vecop_matmul.metal

./vendor/bybylang/backends/opencl:
kernels         opencl_api.nim

./vendor/bybylang/backends/opencl/kernels:
vecop_matmul.cl

./vendor/bybylang/demo:
demo_gpu.bybylang       demo_gpu_out.nim
```

---

## 2. Backend API – các phép toán có sẵn

`backend.nim` cung cấp interface thống nhất cho CPU, Metal và CUDA. Tất cả các hàm dưới đây đều có sẵn qua `Backend` object.

### 2.1. `beMatmul` – phép nhân ma trận

```nim
proc beMatmul(ctx: Backend, a: seq[float32], M, K: int,
              b: seq[float32], K2, N: int): seq[float32]
```

Nhân ma trận `A (M x K)` với `B (K x N)`, trả về `C (M x N)` dạng phẳng.

### 2.2. `beMatmul2` – hai phép matmul độc lập trong cùng một lần gọi

```nim
proc beMatmul2(ctx: Backend,
               a1: seq[float32], M1, K1: int, b1: seq[float32], K1b, N1: int,
               a2: seq[float32], M2, K2: int, b2: seq[float32], K2b, N2: int):
               tuple[y1, y2: seq[float32]]
```

Thực hiện hai phép matmul riêng biệt trong cùng một command buffer (Metal/CUDA) để giảm overhead.

### 2.3. `beAdd` – cộng hai vector

```nim
proc beAdd(ctx: Backend, a, b: seq[float32]): seq[float32]
```

### 2.4. `beRelu` – ReLU activation

```nim
proc beRelu(ctx: Backend, x: seq[float32]): seq[float32]
```

### 2.5. `beSigmoid` – Sigmoid activation

```nim
proc beSigmoid(ctx: Backend, x: seq[float32]): seq[float32]
```

### 2.6. `beTanh` – Tanh activation

```nim
proc beTanh(ctx: Backend, x: seq[float32]): seq[float32]
```

### 2.7. `beSoftmax` – Softmax theo hàng

```nim
proc beSoftmax(ctx: Backend, x: seq[float32], rows, cols: int): seq[float32]
```

Softmax theo từng hàng của ma trận `rows x cols`.

### 2.8. `beLayernorm` – Layer Normalization

```nim
proc beLayernorm(ctx: Backend, x: seq[float32], gamma, beta: seq[float32],
                 rows, cols: int, eps: float32 = 1e-5): seq[float32]
```

### 2.9. `beEmbeddingLookup` – tra cứu embedding

```nim
proc beEmbeddingLookup(ctx: Backend, table: seq[float32], vocab, dim: int,
                        indices: seq[int32]): seq[float32]
```

Tra cứu các hàng trong `table (vocab x dim)` theo `indices`, trả về `(len(indices) x dim)`.

---

## 3. Cách chọn backend

Trong `main.nim` hoặc code của bạn:

```nim
import backend

# Tự động chọn: Metal nếu có, ngược lại CPU
newBackend("auto")

# Chọn cụ thể
newBackend("cpu")
newBackend("metal")
newBackend("cuda")
newBackend("tsic")
```

Hoặc qua build flag (Makefile):

```bash
make run-metal   # dùng Metal
make run-cpu     # dùng CPU
make run-cuda    # dùng CUDA (cần build cuda_kernels.cu trước)
```

---

## 4. Lượng tử hóa (`quant.nim`)

### 4.1. Các kiểu lượng tử

```nim
type QuantKind* = enum
  qkFp32Raw   # không nén
  qkInt8      # int8 symmetric
  qkInt4      # int4 symmetric (2 giá trị/byte)
  qkFp8E4M3   # fp8 (4 exponent, 3 mantissa)
  qkFp8E5M2   # fp8 (5 exponent, 2 mantissa)
  qkCustom    # CustomFloat bất kỳ
  qkAuto      # APF – tự động chọn số bit
```

### 4.2. Lượng tử hóa và giải mã

```nim
import quant

let qt = quantizeTensor(data, shape, qkInt8)
let raw = dequantizeTensor(qt)

# Với APF (tự động)
let qtAuto = quantizeTensor(data, shape, qkAuto, gradArr = grad)
```

### 4.3. Đọc/ghi checkpoint `.nimq`

```nim
saveQuantStateDict("model.nimq", [vocab, embedDim, heads, layers, ffMult], sd)
let (arch, sd) = loadQuantStateDict("model.nimq")
```

---

## 5. Model (`nimformer.nim`)

### 5.1. Tạo model

```nim
import nimformer, backend

var model = newNimformerModel(
  vocab = 64,
  embedDim = 32,
  nHeads = 4,
  nLayers = 2,
  ffMult = 4
)
```

### 5.2. Forward – batch

```nim
let idsBatch = @[@[1, 2, 3], @[4, 5, 6]]   # [B=2, T=3]
let logits = model.forwardBatch(idsBatch, ctx)   # shape [2, 3, vocab]
```

### 5.3. Backward – batch

```nim
let dLoss = randnTensor(@[2, 3, vocab], 0.01)   # gradient giả
let grads = model.backwardBatch(idsBatch, dLoss, ctx)
# grads là seq[Tensor] theo thứ tự: outProj.W, outProj.B,
# các block từ cuối lên đầu (mỗi block gồm 12 gradient),
# và cuối cùng là embed.weight.
```

### 5.4. ApfAdam optimizer

```nim
var state = newApfAdamState(param.data.len)
let cf = apfAdamStep(param, grad, state, lr = 3e-3, requantizeEvery = 50)
```

---

## 6. Checkpoint (lưu và resume)

### 6.1. Lưu checkpoint đầy đủ (weight + optimizer state + step)

```nim
saveCheckpoint(model, states, stepNo, "ckpt.nimq.ckpt",
               weightKind = qkAuto,
               embedDim, nHeads, nLayers, ffMult)
```

### 6.2. Load checkpoint full

```nim
let (model, states, stepNo) = loadCheckpointFull("ckpt.nimq.ckpt")
# Tiếp tục training từ stepNo
```

---

## 7. Export từ Hugging Face → `.nimq`

```bash
python export_to_nimq.py --model TheBloke/deepseek-coder-6.7B-instruct-GPTQ --output model.nimq --int8
```

Hỗ trợ model thường và GPTQ.

---

## 8. Inference với `nim_inference.nim` (cho LLaMA)

```nim
import nim_inference, metal_ai

let ctx = newMetalContext()
let config = newLlamaConfig("path/to/hf/model")
let tokenizer = newHFTokenizer("path/to/hf/model")
var model = loadLlamaModel("model.nimq", config)

let output = generate(model, tokenizer, ctx, "def fib(n):", max_new_tokens = 64)
echo output
```

---

## 9. Định dạng `.nimq`

- Header: `NIMQ1`
- Arch: 5 số nguyên (vocab, embedDim, nHeads, nLayers, ffMult)
- Danh sách các `(tên, QuantTensor)`
- Quy ước: weight dùng kiểu lượng tử đã chọn; bias và LayerNorm (gamma/beta) luôn `qkFp32Raw`

---

## 10. Lưu ý quan trọng

- **Bias và LayerNorm** luôn giữ fp32 để tránh sai số.
- **Buffer pool** trong Metal: dùng `poolGet`/`poolPut` để tránh rò rỉ RAM.
- **Pipeline cache** – compile kernel một lần duy nhất.
- **`@autoreleasepool`** trong `metal_bridge.m` – bắt buộc để dọn autorelease objects.
