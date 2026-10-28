## 4. Quantization (`quant.nim`)

### 4.1. Quantization kinds

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

### 4.2. Quantize / dequantize

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
`quantizeTensor` is the general dispatcher by `kind`. `dequantizeTensorTransposed` dequantizes + transposes in a **single pass**, writing directly to the `[nCols, nRows]` layout — avoiding two full-size fp32 allocations every time `Linear.forward` is called for a large weight (e.g. `lm_head`).

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

`qkFp4KM` uses the FP4-KM format; its convenience functions are included for inference paths that expect that packed layout. The complete `quantizeFp4KM` parameters are in the declaration in `quant.nim` because they include format-specific scaling/group options.

```nim
import quant

let qt = quantizeTensor(data, shape, qkInt8)
let raw = dequantizeTensor(qt)

# Using APF (automatic)
let qtAuto = quantizeTensor(data, shape, qkAuto, gradArr = grad)
```

### 4.3. Reading/writing `.nimq` checkpoints

```nim
proc saveQuantStateDict*(path: string, arch: array[5, int], sd: seq[(string, QuantTensor)])
proc loadQuantStateDict*(path: string): tuple[arch: array[5, int], sd: seq[(string, QuantTensor)]]
```

```nim
saveQuantStateDict("model.nimq", [vocab, embedDim, heads, layers, ffMult], sd)
let (arch, sd) = loadQuantStateDict("model.nimq")
```

---


---

## 9. Exporting from Hugging Face → `.nimq`

```
python export_to_nimq.py --model TheBloke/deepseek-coder-6.7B-instruct-GPTQ --output model.nimq --int8
```

Supports both standard and GPTQ models. Use `test/deepseek-inference/verify_gptq_dequant.py` to double-check GPTQ dequant math, and `test/deepseek-inference/dump_reference_states.py` to dump reference activations from PyTorch to compare against `test/deepseek-inference/nim_inference.nim`.

---


---

## 10. `.nimq` file format

- Header: `NIMQ1`
- Architecture: 5 integers (`vocab`, `embedDim`, `nHeads`, `nLayers`, `ffMult`)
- A list of `(name, QuantTensor)` entries
- Convention: weights use the chosen quantization kind; biases and LayerNorm (`gamma`/`beta`) always use `qkFp32Raw`

---
