
## 3. CustomFloat và APF (`customfloat.nim`)

[English](customfloat-apf.md) | [Tiếng Việt](customfloat-apf.vi.md)

Định dạng số thực có số bit exponent/mantissa tùy chọn, dùng để nén dữ liệu
xuống dưới độ chính xác float32 tiêu chuẩn; APF tự chọn định dạng phù hợp.

```nim
proc newCustomFloat*(exponentBits, mantissaBits: int, name = ""): CustomFloat
let
  FP8_E4M3*, FP8_E5M2*, FP11*, FP16C*, FP24*, FP32C*, FP40*, FP48*, FP64C*: CustomFloat
proc fp32ToPackedU64*(x: float32, cf: CustomFloat): uint64
proc packedU64ToFp32*(packed: uint64, cf: CustomFloat): float32
proc buildDecodeLUT*(cf: CustomFloat): array[256, float32]
proc decodeArrayLUT*(buf: openArray[uint8], lut: array[256, float32]): seq[float32]
```

Tạo định dạng CustomFloat với số bit exponent và mantissa tùy chọn.

```nim
proc cpuEncodeArray*(arr: openArray[float32], cf: CustomFloat): seq[uint8]
proc cpuDecodeArray*(buf: openArray[uint8], cf: CustomFloat): seq[float32]
proc encodeArray*(arr: openArray[float32], cf: CustomFloat): seq[uint8]
proc decodeArray*(buf: openArray[uint8], cf: CustomFloat): seq[float32]
```

Mã hóa/giải mã mảng `float32` thành/từ bytes theo định dạng `cf`.
`encodeArray`/`decodeArray` là API công khai chính.

```nim
proc computeRequiredExponentBits*(arr: openArray[float32], expMargin = APF_EXP_MARGIN_BITS): int
proc computeRequiredMantissaBits*(arr: openArray[float32], relErrorTol = APF_DEFAULT_REL_ERROR_TOL): int
proc buildCustomDtypeForTensor*(arr: openArray[float32], gradArr: openArray[float32] = [],
                               relErrorTol = APF_DEFAULT_REL_ERROR_TOL,
                               expMargin = APF_EXP_MARGIN_BITS): CustomFloat
```

Tính số bit exponent/mantissa cần để biểu diễn `arr` (và xét `gradArr` nếu
có) theo sai số cho phép, rồi tạo `CustomFloat` phù hợp. Đây là phần cốt lõi
của APF (Adaptive Precision Format).

```nim
proc apfCastForTraining*(arr: openArray[float32], gradArr: openArray[float32] = [],
                         relErrorTol = APF_DEFAULT_REL_ERROR_TOL,
                         expMargin = APF_EXP_MARGIN_BITS): tuple[data: seq[uint8], cf: CustomFloat]
proc apfDecodeForTraining*(encoded: openArray[uint8], cf: CustomFloat): seq[float32]
```

`apfCastForTraining` chọn CustomFloat và mã hóa `arr` trong một bước, dùng
trong `apfAdamStep` ở phần 5.8 để nén tensor khi train.
`apfDecodeForTraining` thực hiện chiều ngược lại.

---
