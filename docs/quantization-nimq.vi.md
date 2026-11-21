
## 4. Lượng tử hóa (`quant.nim`)

[English](quantization-nimq.md) | [Tiếng Việt](quantization-nimq.vi.md)


### 4.1. Các kiểu lượng tử hóa

```nim
type QuantKind* = enum
  qkFp32Raw          # no compression, kept as plain float32 (used for bias/layernorm — precision-sensitive)
  qkInt8             # symmetric int8, scale = max(|x|)/127
  qkInt4             # symmetric int4 (2 values per byte), scale = max(|x|)/7
  qkInt4Asymmetric   # asymmetric int4, Q4_K_M style, scale=(max-min)/15, with zero_point
  qkFp8E4M3          # CustomFloat(4 exponent, 3 mantissa)
  qkFp8E5M2          # CustomFloat(5 exponent, 2 mantissa)
  qkCustom           # arbitrary user-declared CustomFloat
  qkAuto             # APF — auto-builds a CustomFloat from the tensor itself (buildCustomDtypeForTensor)
  qkFp4KM            # OCP E2M1 FP4 with per-group scale, used by FP4-KM inference paths

type QuantTensor* = object
  kind*: QuantKind
  shape*: seq[int]
  scale*: seq[float32]       # len=1 (per-tensor), len=shape[0] (per-row),
                             # or len=shape[0]*ceil(shape[1]/groupSize) (per-group)
  zero_point*: seq[float32] # parallel to scale
  groupSize*: int           # 0 = per-tensor/per-row; >0 = per-group (GPTQ-style, group=128 is common)
  cf*: CustomFloat          # used for qkFp8E4M3/E5M2/qkCustom/qkAuto
  data*: seq[uint8]
```


### 4.2. Lượng tử hóa và giải lượng tử

```nim
proc quantizeInt8*(arr: openArray[float32], shape: seq[int]): QuantTensor
proc dequantizeInt8*(qt: QuantTensor): seq[float32]
proc quantizeInt4*(arr: openArray[float32], shape: seq[int]): QuantTensor
proc dequantizeInt4*(qt: QuantTensor): seq[float32]
proc quantizeInt4Asymmetric*(arr: openArray[float32], shape: seq[int]): QuantTensor
proc dequantizeInt4Asymmetric*(qt: QuantTensor): seq[float32]
proc quantizeTensor*(arr: openArray[float32], shape: seq[int], kind: QuantKind,
                     customCf: CustomFloat = FP24, gradArr: openArray[float32] = []): QuantTensor
proc dequantizeTensor*(qt: QuantTensor): seq[float32]
proc dequantizeTensorTransposed*(qt: QuantTensor): seq[float32]
```

`quantizeTensor` chọn cách xử lý theo `kind`. `dequantizeTensorTransposed`
giải lượng tử và transpose trong **một lượt**, ghi thẳng layout
`[nCols,nRows]`, tránh hai allocation fp32 toàn kích thước mỗi lần
`Linear.forward` xử lý trọng số lớn như `lm_head`.

```nim
proc quantizeRaw*(arr: openArray[float32], shape: seq[int]): QuantTensor
proc dequantizeRaw*(qt: QuantTensor): seq[float32]
proc quantizeFloatDtype*(arr: openArray[float32], shape: seq[int], cf: CustomFloat, kind: QuantKind): QuantTensor
proc quantizeAuto*(arr: openArray[float32], shape: seq[int], gradArr: openArray[float32] = []): QuantTensor
proc quantizeFp4KM*(arr: openArray[float32], shape: seq[int],
                    groupSize: int = 32, searchSteps: int = 8): QuantTensor
proc dequantizeFp4KM*(qt: QuantTensor): seq[float32]
proc fp4kmDecodeLut*(): array[16, float32]
```

`qkFp4KM` dùng định dạng FP4-KM. Các hàm tiện ích phục vụ nhánh inference cần
layout đóng gói này. Xem khai báo trong `quant.nim` để biết đầy đủ tham số
`quantizeFp4KM`, gồm các tùy chọn scale/group riêng của định dạng.

```nim
import quant

let qt = quantizeTensor(data, shape, qkInt8)
let raw = dequantizeTensor(qt)

# Using APF (automatic)
let qtAuto = quantizeTensor(data, shape, qkAuto, gradArr = grad)
```


### 4.3. Đọc và ghi checkpoint `.nimq`

```nim
proc saveQuantStateDict*(path: string, arch: array[5, int], sd: seq[(string, QuantTensor)])
proc loadQuantStateDict*(path: string): tuple[arch: array[5, int], sd: seq[(string, QuantTensor)]]
```

```nim
saveQuantStateDict("model.nimq", [vocab, embedDim, heads, layers, ffMult], sd)
let (arch, sd) = loadQuantStateDict("model.nimq")
```

---


## 9. Xuất từ Hugging Face sang `.nimq`

`export_to_nimq.py` nhận checkpoint Hugging Face thông thường và GPTQ. Việc
chọn dtype nguồn độc lập với định dạng lượng tử hóa NIMQ. `auto` theo dtype
trong checkpoint/config; có thể ghi đè bằng `fp16`, `bf16` hoặc `fp32`. Trọng
số được chuyển thành tensor torch FP32 để lượng tử hóa, nhờ đó BF16 không cần
chuyển trực tiếp sang NumPy.

```sh
# Mặc định xuất INT8 đối xứng và tự chọn quantization device.
python export_to_nimq.py --model <hf-model-or-path> --output model.nimq \
  --source-dtype auto --int8 --quant-device auto

# Q4_K_M-style INT4 chia nhóm từ checkpoint BF16.
python export_to_nimq.py --model <hf-model-or-path> --output model-q4km.nimq \
  --source-dtype bf16 --q4km --group-size 128 --quant-device cuda

# FP4-KM từ checkpoint FP32 trên bản PyTorch ROCm.
python export_to_nimq.py --model <hf-model-or-path> --output model-fp4km.nimq \
  --source-dtype fp32 --fp4-km --quant-device rocm
```

Chọn một định dạng đầu ra: `--int8` (mặc định), `--q4km`, `--fp4-km`
(`--fp4km` là alias), `--int4` cho INT4 đối xứng hoặc `--raw` để ghi tensor
NIMQ chưa lượng tử hóa. Tensor raw NIMQ được lưu ở FP32. Cờ đầu ra cũ
`--fp16` vẫn là alias của `--raw`; muốn chọn trọng số nguồn FP16, dùng
`--source-dtype fp16` cùng định dạng đầu ra mong muốn.

`--q4km` dùng INT4 bất đối xứng theo nhóm, mặc định 128 giá trị mỗi nhóm;
`--group-size` phải là số chẵn dương. `--fp4-km` dùng nhóm cố định 32, giá trị
OCP E2M1, scale nhóm E4M3 và một scale FP32 cho toàn tensor. FP4-KM yêu cầu
ma trận 2D có số cột chia hết cho 32. Tensor không phù hợp, embedding,
`lm_head` và normalization được giữ ở raw FP32.

`--quant-device auto` tự chọn thiết bị PyTorch đang có. Có thể chỉ định `cpu`,
`mps`, `cuda`, `rocm`, `xpu` hoặc device string PyTorch đã đăng ký, ví dụ
`cuda:1`. ROCm dùng device API CUDA của PyTorch. Nếu backend không chạy được
một phép toán hoặc hết bộ nhớ, exporter thử lại trên CPU và tiếp tục các
tensor còn lại bằng CPU. `--no-accelerator` buộc dùng CPU; `--no-mps` vẫn được
giữ làm alias tương thích.

Để kiểm tra riêng GPTQ, dùng `test/deepseek-inference/verify_gptq_dequant.py`
đối chiếu phép giải lượng tử và
`test/deepseek-inference/dump_reference_states.py` xuất activation tham chiếu
từ PyTorch để so với `test/deepseek-inference/nim_inference.nim`.

---


## 10. Định dạng file `.nimq`

- Header: `NIMQ2` có tiền tố độ dài; loader hiện từ chối magic cũ.
- Kiến trúc: 5 số nguyên (`vocab`, `embedDim`, `nHeads`, `nLayers`, `ffMult`).
- Danh sách các cặp `(name, QuantTensor)`.
- Quy ước: trọng số dùng kiểu lượng tử hóa đã chọn; bias và LayerNorm
  (`gamma`/`beta`) dùng `qkFp32Raw`.

---
