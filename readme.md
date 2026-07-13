# Nimformer Framework

Nimformer là framework transformer thuần **Nim**, hỗ trợ đa backend (CPU, Metal, CUDA), lượng tử hóa thích ứng (int8/int4/fp8/APF) và checkpoint nén `.nimq`.

## 1. Cấu trúc dự án

```
customfloat.nim        – CustomFloat + APF (CPU)
quant.nim              – Lượng tử hóa tensor & đọc/ghi .nimq
backend.nim            – Abstract backend (CPU/Metal/CUDA) + CÁC PHÉP TOÁN
metal_ai.nim           – Metal backend (GPU) – wrapper cho metal_bridge
metal_bridge.h/.m      – Objective-C bridge cho Metal
metal_kernels.metal    – Metal kernels
cuda_bridge.h          – C header cho CUDA
cuda_kernels.cu        – CUDA kernels + cuBLAS
nimformer.nim          – Transformer model (Linear, Attention, Block, Embedding, ApfAdam)
main.nim               – CLI training driver
test_nimformer.nim     – Test nhanh
export_to_nimq.nim     – Export PyTorch model → .nimq (Python)
nim_inference.nim      – Inference LLaMA (DeepSeek-Coder, Llama…)
Makefile               – Build & chạy
```

---

## 2. Backend API – các phép toán có sẵn

`backend.nim` cung cấp interface thống nhất cho CPU, Metal và CUDA. Tất cả các hàm dưới đây đều có sẵn qua `Backend` object.

### 2.1. `beMatmul` – phép nhân ma trận

```
proc beMatmul(ctx: Backend, a: seq[float32], M, K: int,
              b: seq[float32], K2, N: int): seq[float32]
```

Nhân ma trận `A (M x K)` với `B (K x N)`, trả về `C (M x N)` dạng phẳng.

### 2.2. `beMatmul2` – hai phép matmul độc lập trong cùng một lần gọi

```
proc beMatmul2(ctx: Backend,
               a1: seq[float32], M1, K1: int, b1: seq[float32], K1b, N1: int,
               a2: seq[float32], M2, K2: int, b2: seq[float32], K2b, N2: int):
               tuple[y1, y2: seq[float32]]
```

Thực hiện hai phép matmul riêng biệt trong cùng một command buffer (Metal/CUDA) để giảm overhead.

### 2.3. `beAdd` – cộng hai vector

```
proc beAdd(ctx: Backend, a, b: seq[float32]): seq[float32]
```

### 2.4. `beSub` – trừ hai vector

```
proc beSub(ctx: Backend, a, b: seq[float32]): seq[float32]
```

Trừ từng phần tử `a - b`.

### 2.5. `beMul` – nhân từng phần tử (element-wise)

```
proc beMul(ctx: Backend, a, b: seq[float32]): seq[float32]
```

### 2.6. `beDiv` – chia từng phần tử (element-wise)

```
proc beDiv(ctx: Backend, a, b: seq[float32]): seq[float32]
```

### 2.7. `beRelu` – ReLU activation

```
proc beRelu(ctx: Backend, x: seq[float32]): seq[float32]
```

### 2.8. `beSigmoid` – Sigmoid activation

```
proc beSigmoid(ctx: Backend, x: seq[float32]): seq[float32]
```

### 2.9. `beTanh` – Tanh activation

```
proc beTanh(ctx: Backend, x: seq[float32]): seq[float32]
```

### 2.10. `beApflu` – APF Linear Unit activation

```
proc beApflu(ctx: Backend, x: seq[float32],
             alpha: float32 = 0.1, beta: float32 = 0.1): seq[float32]
```

Activation tuỳ biến theo tham số `alpha`/`beta` (mặc định 0.1).

### 2.11. `beApfluBackward` – backward pass cho APFLU

```
proc beApfluBackward(ctx: Backend, x, dy: seq[float32],
                      alpha: float32 = 0.1, beta: float32 = 0.1): seq[float32]
```

Tính gradient ngược cho `beApflu`.

### 2.12. `beSoftmax` – Softmax theo hàng

```
proc beSoftmax(ctx: Backend, x: seq[float32], rows, cols: int): seq[float32]
```

Softmax theo từng hàng của ma trận `rows x cols`.

### 2.13. `beLayernorm` – Layer Normalization

```
proc beLayernorm(ctx: Backend, x: seq[float32], gamma, beta: seq[float32],
                 rows, cols: int, eps: float32 = 1e-5): seq[float32]
```

### 2.14. `beLayernormBackward` – backward pass cho Layer Normalization

```
proc beLayernormBackward(ctx: Backend, dy, x, gamma, beta: seq[float32],
                          rows, cols: int, eps: float32):
                          tuple[dx, dgamma, dbeta: seq[float32]]
```

Tính gradient của input, `gamma` và `beta` từ `dy` (gradient đầu ra).

### 2.15. `beEmbeddingLookup` – tra cứu embedding

```
proc beEmbeddingLookup(ctx: Backend, table: seq[float32], vocab, dim: int,
                        indices: seq[int32]): seq[float32]
```

Tra cứu các hàng trong `table (vocab x dim)` theo `indices`, trả về `(len(indices) x dim)`.

### 2.16. `beAttentionFused` – Attention fused (forward)

```
proc beAttentionFused(ctx: Backend, q, k, v, mask: seq[float32],
                       B, H, S, D: int, scale: float32):
                       tuple[o, s_matrix: seq[float32]]
```

Tính attention (Q, K, V, mask) đã fuse thành một lần gọi kernel duy nhất — giảm overhead so với tách rời matmul + softmax. `B`=batch, `H`=số head, `S`=độ dài chuỗi, `D`=kích thước head. Trả về output `o` và ma trận attention score `s_matrix` (dùng lại cho backward).

### 2.17. `beAttentionFusedBackward` – Attention fused (backward)

```
proc beAttentionFusedBackward(ctx: Backend, q, k, v, s_matrix, dy: seq[float32],
                               B, H, S, D: int, scale: float32):
                               tuple[dq, dk, dv: seq[float32]]
```

Tính gradient `dq, dk, dv` cho attention fused, dùng `s_matrix` đã lưu từ bước forward.

---

## 2b. CustomFloat / APF helpers (`backend.nim`)

Các hàm hỗ trợ mã hoá/giải mã và tự động chọn định dạng CustomFloat, dùng khi cần nén dữ liệu ở độ chính xác thấp hơn float32 chuẩn.

### 2b.1. `beCustomfloatEncode` – encode mảng float32 sang bytes

```
proc beCustomfloatEncode(ctx: Backend, arr: seq[float32], cf: CustomFloat): seq[uint8]
```

### 2b.2. `beCustomfloatDecode` – decode bytes ngược lại float32

```
proc beCustomfloatDecode(ctx: Backend, buf: seq[uint8], cf: CustomFloat): seq[float32]
```

### 2b.3. `beApfCastForTraining` – tự động chọn CustomFloat phù hợp cho training

```
proc beApfCastForTraining(ctx: Backend, arr: seq[float32], gradArr: seq[float32] = [],
                           relErrorTol = APF_DEFAULT_REL_ERROR_TOL,
                           expMargin = APF_EXP_MARGIN_BITS):
                           tuple[data: seq[uint8], cf: CustomFloat]
```

Dựa vào `arr` (và `gradArr` nếu có) để tự động chọn số bit / định dạng CustomFloat (APF) tối ưu, cân bằng giữa sai số cho phép (`relErrorTol`) và biên độ exponent (`expMargin`). Trả về dữ liệu đã encode cùng `CustomFloat` đã chọn.

---

## 3. Cách chọn backend

Trong `main.nim` hoặc code của bạn:

```
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

```
make run-metal   # dùng Metal
make run-cpu     # dùng CPU
make run-cuda    # dùng CUDA (cần build cuda_kernels.cu trước)
```

---

## 4. Lượng tử hóa (`quant.nim`)

### 4.1. Các kiểu lượng tử

```
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

```
import quant

let qt = quantizeTensor(data, shape, qkInt8)
let raw = dequantizeTensor(qt)

# Với APF (tự động)
let qtAuto = quantizeTensor(data, shape, qkAuto, gradArr = grad)
```

### 4.3. Đọc/ghi checkpoint `.nimq`

```
saveQuantStateDict("model.nimq", [vocab, embedDim, heads, layers, ffMult], sd)
let (arch, sd) = loadQuantStateDict("model.nimq")
```

---

## 5. Model (`nimformer.nim`)

### 5.1. Tạo model

```
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

```
let idsBatch = @[@[1, 2, 3], @[4, 5, 6]]   # [B=2, T=3]
let logits = model.forwardBatch(idsBatch, ctx)   # shape [2, 3, vocab]
```

### 5.3. Backward – batch

```
let dLoss = randnTensor(@[2, 3, vocab], 0.01)   # gradient giả
let grads = model.backwardBatch(idsBatch, dLoss, ctx)
# grads là seq[Tensor] theo thứ tự: outProj.W, outProj.B,
# các block từ cuối lên đầu (mỗi block gồm 12 gradient),
# và cuối cùng là embed.weight.
```

### 5.4. ApfAdam optimizer

```
var state = newApfAdamState(param.data.len)
let cf = apfAdamStep(param, grad, state, lr = 3e-3, requantizeEvery = 50)
```

---

## 6. Checkpoint (lưu và resume)

### 6.1. Lưu checkpoint đầy đủ (weight + optimizer state + step)

```
saveCheckpoint(model, states, stepNo, "ckpt.nimq.ckpt",
               weightKind = qkAuto,
               embedDim, nHeads, nLayers, ffMult)
```

### 6.2. Load checkpoint full

```
let (model, states, stepNo) = loadCheckpointFull("ckpt.nimq.ckpt")
# Tiếp tục training từ stepNo
```

---

## 7. Export từ Hugging Face → `.nimq`

```
python export_to_nimq.py --model TheBloke/deepseek-coder-6.7B-instruct-GPTQ --output model.nimq --int8
```

Hỗ trợ model thường và GPTQ.

---

## 8. Inference với `nim_inference.nim` (cho LLaMA)

```
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
- **Backend "auto"**: nếu không dò được GPU nào (CUDA/Metal/OpenCL/TSIC), mặc định sẽ `raise` lỗi thay vì âm thầm rơi về CPU (`gForbidCpuFallback = true`). Muốn chạy CPU khi không có GPU: gọi `setForbidCpuFallback(false)` trước `newBackend()`, hoặc dùng `newBackend("cpu")` tường minh.