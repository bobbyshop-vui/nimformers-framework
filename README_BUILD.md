# Build — Nim port của metal_ai.py

## Yêu cầu
- **macOS** với GPU hỗ trợ Metal (bắt buộc cho phần GPU; phần `customfloat.nim` chạy CPU-only, chạy được mọi OS).
- Xcode Command Line Tools (để có `clang` biên dịch được Objective-C và link framework `Metal`/`Foundation`):
  ```bash
  xcode-select --install
  ```
- Nim >= 2.0: `brew install nim` hoặc qua [choosenim](https://github.com/dom96/choosenim).

## Cấu trúc file
```
customfloat.nim    -- CustomFloat + APF, thuần Nim, không cần GPU
quant.nim          -- Lượng tử hoá int8/int4/fp8_e4m3/fp8_e5m2/custom/auto(APF) + lưu/tải state dict
nimformer.nim      -- Transformer custom + ApfAdam optimizer + quantize/save/load model (đều là THƯ VIỆN, không có main)
metal_bridge.h     -- API C generic để dispatch Metal
metal_bridge.m     -- cài đặt Objective-C của bridge (compile cùng lúc với Nim)
metal_kernels.metal -- kernel Metal thật (add, matmul, relu/sigmoid/tanh, softmax, layernorm, embedding_lookup)
metal_ai.nim       -- MetalContext + wrapper cho TOÀN BỘ kernel trên (nhúng metal_kernels.metal lúc compile qua staticRead)
test_nimformer.nim -- FILE TEST độc lập (có main), build model + forward + optimizer + quantize/save/load
```
Các file thư viện (`customfloat.nim`, `quant.nim`, `nimformer.nim`, `metal_ai.nim`) **không có `when isMainModule`** —
chỉ để `import`, không tự chạy. Muốn chạy thử, dùng `test_nimformer.nim`.

## Chạy file test (CPU-only, không cần Metal/macOS)
```bash
nim c -r test_nimformer.nim
```
File này build 1 model nhỏ, forward, chạy 1 bước `ApfAdam`, rồi lượng tử hoá +
lưu (`.nimq`) + tải lại model với `int8`, `int4`, `fp8_e4m3`, `fp8_e5m2`, `auto` (APF),
và ví dụ dtype tuỳ ý (fp6: 3 exponent bit + 2 mantissa bit) — in ra sai số tối đa
so với bản gốc cho mỗi loại để bạn so sánh mức nén/độ mất mát.

## Lượng tử hoá & load model — API nhanh (`quant.nim` + `nimformer.nim`)
```nim
import nimformer, quant

var model = newNimformerModel(vocab=64, embedDim=32, nHeads=4, nLayers=2, ffMult=4)

# Lưu model dạng int8 / int4 / fp8 / auto(APF):
saveQuantizedModel(model, "model_int8.nimq", qkInt8)
saveQuantizedModel(model, "model_int4.nimq", qkInt4)
saveQuantizedModel(model, "model_fp8.nimq",  qkFp8E4M3)
saveQuantizedModel(model, "model_auto.nimq", qkAuto)   # APF tự build precision theo từng tensor

# Tải lại (tự giải nén về float32 để forward bình thường):
let loaded = loadQuantizedModel("model_int8.nimq")
let logits = loaded.forward(@[1, 5, 9])
```
Muốn dtype tuỳ biến (ví dụ fp6, fp11, fp13...) cho *từng tensor riêng lẻ* thay vì
cả model, gọi thẳng `quantizeTensor(data, shape, qkCustom, myCustomFloat)` /
`dequantizeTensor(qt)` trong `quant.nim` — xem ví dụ trong `test_nimformer.nim`.

Quy ước nén: **trọng số** (Linear/Embedding weight) theo dtype bạn chọn;
**bias và LayerNorm (gamma/beta)** luôn giữ `qkFp32Raw` (không nén) vì chúng
nhỏ và nhạy sai số hơn nhiều so với ma trận trọng số lớn.

## Chỉ test phần thuần số học (không cần Metal/macOS)
```bash
nim c -r customfloat.nim
```
Chạy được trên Linux/Windows luôn, vì đây chỉ là bit-manipulation thuần Nim
(tương đương nhánh `encode_numpy`/`decode_numpy` trong bản Python).

## Build phần GPU (bắt buộc macOS)
File `metal_ai.nim` đã có sẵn pragma tự động link/compile:
```nim
{.passC: "-fobjc-arc".}
{.passL: "-framework Metal -framework Foundation".}
{.compile: "metal_bridge.m".}
```
nên chỉ cần:
```bash
nim c -d:release -r metal_ai.nim
```
Nim sẽ tự gọi clang biên dịch `metal_bridge.m` (Objective-C, có ARC) và link
2 framework, không cần Makefile riêng.

Nếu muốn build thủ công từng bước (debug):
```bash
# 1. Biên dịch riêng file Objective-C thành .o
clang -fobjc-arc -c metal_bridge.m -o metal_bridge.o \
  -framework Metal -framework Foundation

# 2. Build Nim, trỏ tới .o đã có sẵn (thay vì để Nim tự compile lại)
nim c -d:release --passL:"metal_bridge.o -framework Metal -framework Foundation" metal_ai.nim
```

## Mở rộng thêm kernel (matmul, softmax, layernorm, attention, Adam...)
Theo đúng pattern trong `metal_ai.nim`:
1. Thêm source Metal (chuỗi `.metal`) vào hằng `kernelSrc`.
2. Gọi `ctx.pipeline("tên_kernel")` để lấy pipeline.
3. Upload buffer bằng `ctx.uploadF32`/`ctx.uploadU32`/`ctx.uploadBytes`.
4. Gọi `ctx.dispatch(pipeline, bufs, gx, gy, gz, tx, ty, tz)`.
5. Đọc kết quả bằng `downloadF32`/`downloadU8`.

Đây chính là cách `metal_add` và `customfloatEncodeGpu` đã làm — copy y hệt
pattern đó cho `matmul`, `softmax`, `layernorm`, `embedding_lookup`, kernel
attention trong `attention.metal` gốc, v.v. Vì `mtl_dispatch` là generic
(nhận list buffer + tên pipeline), bạn không cần sửa `metal_bridge.m`/`.h`
thêm nữa — chỉ thêm code Nim ở tầng `metal_ai.nim`.

## Ghi chú quan trọng
- `metal_bridge.m` dùng `__bridge_retained` để giữ sống object Objective-C
  qua con trỏ `void*` trả về Nim — phù hợp cho 1 `MetalContext` sống suốt
  vòng đời chương trình (giống cách bản Python giữ `self.device`,
  `self.queue`, `_shader_pipelines` làm global/attribute không giải phóng).
  Nếu cần giải phóng sớm hàng nghìn buffer tạm, nên thêm buffer pool
  (`_pool_get`/`_pool_put` trong bản gốc) ở tầng Nim thay vì alloc mới mỗi lần.
- Phần transformer/Adam/AMP/attention trong bản Python còn nguyên
  `PLACEHOLDER_*` (chưa hoàn chỉnh sẵn) — mình chưa port phần đó vì chưa có
  code gốc để port đúng. Nếu bạn có bản đầy đủ (điền hết placeholder), gửi
  mình port tiếp theo đúng pattern generic-dispatch ở trên.
