## 3. CustomFloat / APF (`customfloat.nim`)

A custom floating-point format with arbitrary exponent/mantissa bit widths, used to compress data below standard float32 precision, plus an "APF" mechanism that automatically picks the right format.

```nim
proc newCustomFloat*(exponentBits, mantissaBits: int, name = ""): CustomFloat
let
  FP8_E4M3*, FP8_E5M2*, FP11*, FP16C*, FP24*, FP32C*, FP40*, FP48*, FP64C*: CustomFloat
proc fp32ToPackedU64*(x: float32, cf: CustomFloat): uint64
proc packedU64ToFp32*(packed: uint64, cf: CustomFloat): float32
proc buildDecodeLUT*(cf: CustomFloat): array[256, float32]
proc decodeArrayLUT*(buf: openArray[uint8], lut: array[256, float32]): seq[float32]
```
Creates a new CustomFloat format with arbitrary exponent/mantissa bit counts.

```nim
proc cpuEncodeArray*(arr: openArray[float32], cf: CustomFloat): seq[uint8]
proc cpuDecodeArray*(buf: openArray[uint8], cf: CustomFloat): seq[float32]
proc encodeArray*(arr: openArray[float32], cf: CustomFloat): seq[uint8]
proc decodeArray*(buf: openArray[uint8], cf: CustomFloat): seq[float32]
```
Encode/decode a `float32` array to/from bytes in format `cf`. `encodeArray`/`decodeArray` are the main public API.

```nim
proc computeRequiredExponentBits*(arr: openArray[float32], expMargin = APF_EXP_MARGIN_BITS): int
proc computeRequiredMantissaBits*(arr: openArray[float32], relErrorTol = APF_DEFAULT_REL_ERROR_TOL): int
proc buildCustomDtypeForTensor*(arr: openArray[float32], gradArr: openArray[float32] = [],
                               relErrorTol = APF_DEFAULT_REL_ERROR_TOL,
                               expMargin = APF_EXP_MARGIN_BITS): CustomFloat
```
Computes the minimum exponent/mantissa bits needed to represent `arr` (and `gradArr`, if given) within the allowed error, then builds the best-fitting `CustomFloat` — this is the core "APF" (Adaptive Precision Format) algorithm.

```nim
proc apfCastForTraining*(arr: openArray[float32], gradArr: openArray[float32] = [],
                         relErrorTol = APF_DEFAULT_REL_ERROR_TOL,
                         expMargin = APF_EXP_MARGIN_BITS): tuple[data: seq[uint8], cf: CustomFloat]
proc apfDecodeForTraining*(encoded: openArray[uint8], cf: CustomFloat): seq[float32]
```
`apfCastForTraining`: picks a suitable `CustomFloat` for `arr` and encodes it in one step — used inside `apfAdamStep` (section 5.8) to compress tensors during training. `apfDecodeForTraining` is the reverse.

---
