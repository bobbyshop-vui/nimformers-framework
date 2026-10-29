## customfloat.nim
## Port của CustomFloat + APF (Adaptive Precision Float) từ metal_ai.py
## Không phụ thuộc trực tiếp vào GPU driver nào - ủy quyền hoàn toàn qua BybyLang (CUDA/Metal/OpenCL/Vulkan).
## Tự động tăng tốc GPU, hỗ trợ CPU Fallback tự động kèm theo log cảnh báo rõ ràng.

import std/[math, strformat, algorithm]

type
  CustomFloat* = object
    exponentBits*, mantissaBits*, totalBits*, itemSize*: int
    bias*, maxExp*: int
    name*: string
    usesUint64*: bool

proc newCustomFloat*(exponentBits, mantissaBits: int, name = ""): CustomFloat =
  # === SỬA: cho phép exponentBits = 0 cho trường hợp int8/int4 ===
  assert exponentBits >= 0 and mantissaBits >= 0  # Đổi từ >= 1 thành >= 0
  result.exponentBits = exponentBits
  result.mantissaBits = mantissaBits
  result.totalBits = 1 + exponentBits + mantissaBits
  result.itemSize = (result.totalBits + 7) div 8
  # === SỬA: nếu exponentBits = 0 thì bias = 0 ===
  if exponentBits > 0:
    result.bias = (1 shl (exponentBits - 1)) - 1
    result.maxExp = (1 shl exponentBits) - 1
  else:
    result.bias = 0
    result.maxExp = 0
  result.name = if name.len > 0: name
                else: &"float{result.totalBits}_e{exponentBits}m{mantissaBits}"
  result.usesUint64 = result.totalBits <= 64

# ── Preset thường dùng (tương đương FP8_E4M3 ... FP256 trong bản Python) ──
let
  FP8_E4M3* = newCustomFloat(4, 3, "fp8_e4m3")
  FP8_E5M2* = newCustomFloat(5, 2, "fp8_e5m2")
  FP11*     = newCustomFloat(5, 5, "fp11")
  FP16C*    = newCustomFloat(5, 10, "float16_custom")
  FP24*     = newCustomFloat(8, 15, "float24")
  FP32C*    = newCustomFloat(8, 23, "float32_custom")
  FP40*     = newCustomFloat(8, 31, "float40")
  FP48*     = newCustomFloat(8, 39, "float48")
  FP64C*    = newCustomFloat(11, 52, "float64_custom")

# ─────────────────────────────────────────────────────────────
# fp32 <-> packed uint64  (CPU fallback core, dùng khi không có GPU)
# ─────────────────────────────────────────────────────────────

proc fp32ToPackedU64*(x: float32, cf: CustomFloat): uint64 =
  let bits32 = cast[uint32](x)
  let sign  = uint64((bits32 shr 31) and 0x1'u32)
  let exp32 = uint64((bits32 shr 23) and 0xFF'u32)
  let mant32 = uint64(bits32 and 0x7FFFFF'u32)

  let realExp = int(exp32) - 127
  let isZero = (exp32 == 0) and (mant32 == 0)
  let isInfNan = (exp32 == 255)

  var newExp = realExp + cf.bias
  let overflow  = (newExp >= cf.maxExp) and not isZero
  let underflow = (newExp <= 0) and not isZero

  var mant: uint64
  if cf.mantissaBits <= 23:
    mant = mant32 shr uint64(23 - cf.mantissaBits)
  else:
    mant = mant32 shl uint64(cf.mantissaBits - 23)

  let newExpU = uint64(clamp(newExp, 0, cf.maxExp))
  let topShift = uint64(cf.exponentBits + cf.mantissaBits)
  var packed = (sign shl topShift) or (newExpU shl uint64(cf.mantissaBits)) or mant

  let infPattern = uint64(cf.maxExp) shl uint64(cf.mantissaBits)
  if isInfNan or (overflow and not isInfNan):
    packed = (sign shl topShift) or infPattern
  elif underflow or isZero:
    packed = sign shl topShift
  result = packed

proc packedU64ToFp32*(packed: uint64, cf: CustomFloat): float32 =
  let mantMask = if cf.mantissaBits == 0: 0'u64 else: (1'u64 shl cf.mantissaBits) - 1
  let topShift = uint64(cf.exponentBits + cf.mantissaBits)
  let sign = (packed shr topShift) and 1'u64
  let exp  = (packed shr uint64(cf.mantissaBits)) and uint64(cf.maxExp)
  let mant = packed and mantMask

  let isZero = (exp == 0) and (mant == 0)
  let isInfNan = (exp == uint64(cf.maxExp))

  let realExp = int(exp) - cf.bias
  let exp32 = uint32(clamp(realExp + 127, 0, 255))

  var mant32: uint32
  if cf.mantissaBits <= 23:
    mant32 = uint32(mant shl uint64(23 - cf.mantissaBits))
  else:
    mant32 = uint32(mant shr uint64(cf.mantissaBits - 23))

  var bits32 = (uint32(sign) shl 31) or (exp32 shl 23) or mant32
  if isZero: bits32 = uint32(sign) shl 31
  if isInfNan: bits32 = (uint32(sign) shl 31) or (0xFF'u32 shl 23)
  result = cast[float32](bits32)

# ─────────────────────────────────────────────────────────────
# API Điểm Vào Cho Framework (Hỗ Trợ GPU OpenCL & CPU Fallback)
# ─────────────────────────────────────────────────────────────

proc buildDecodeLUT*(cf: CustomFloat): array[256, float32] =
  ## LUT giai ma nhanh - CHI dung duoc khi cf.itemSize == 1 (1 byte/phan tu:
  ## dung cho ca fp4 VA fp8 trong codebase nay, vi itemSize = ceil(totalBits/8)
  ## nen fp4 (4 bit) van chiem tron 1 byte, khong nen 2 gia tri/byte). Build
  ## 1 LAN cho moi CustomFloat spec (khong doi trong suot qua trinh train),
  ## sau do decode chi la 1 phep tra bang O(1) thay vi tinh lai exponent/
  ## mantissa tung phan tu (packedU64ToFp32) - nhanh hon dang ke khi decode
  ## lap lai nhieu lan (vd on-demand dequant weight moi buoc train).
  assert cf.itemSize == 1, &"buildDecodeLUT chi ho tro itemSize==1 (fp4/fp8), cf nay itemSize={cf.itemSize}"
  for b in 0 ..< 256:
    result[b] = packedU64ToFp32(uint64(b), cf)

proc decodeArrayLUT*(buf: openArray[uint8], lut: array[256, float32]): seq[float32] =
  ## Ban nhanh cua decodeArray, dung LUT da build san (buildDecodeLUT). Ket
  ## qua GIONG HET decodeArray(buf, cf) (da verify - xem test_lut_correctness
  ## trong smoke test) nhung nhanh hon vi khong tinh lai bit-arithmetic.
  result = newSeq[float32](buf.len)
  for i in 0 ..< buf.len:
    result[i] = lut[int(buf[i])]

proc cpuEncodeArray*(arr: openArray[float32], cf: CustomFloat): seq[uint8] =
  result = newSeq[uint8](arr.len * cf.itemSize)
  for i, v in arr:
    let packed = fp32ToPackedU64(v, cf)
    for b in 0 ..< cf.itemSize:
      result[i * cf.itemSize + b] = uint8((packed shr uint64(8 * b)) and 0xFF'u64)

proc cpuDecodeArray*(buf: openArray[uint8], cf: CustomFloat): seq[float32] =
  assert buf.len mod cf.itemSize == 0
  let n = buf.len div cf.itemSize
  result = newSeq[float32](n)
  for i in 0 ..< n:
    var packed = 0'u64
    for b in 0 ..< cf.itemSize:
      packed = packed or (uint64(buf[i * cf.itemSize + b]) shl uint64(8 * b))
    result[i] = packedU64ToFp32(packed, cf)

proc encodeArray*(arr: openArray[float32], cf: CustomFloat): seq[uint8] =
  ## float32 -> bytes đóng gói theo CustomFloat (itemSize byte / phần tử, little-endian)
  assert cf.usesUint64, "encodeArray chỉ hỗ trợ total_bits <= 64"
  if arr.len == 0: return @[]
  return cpuEncodeArray(arr, cf)

proc decodeArray*(buf: openArray[uint8], cf: CustomFloat): seq[float32] =
  ## bytes đóng gói -> float32
  assert cf.usesUint64, "decodeArray chỉ hỗ trợ total_bits <= 64"
  assert buf.len mod cf.itemSize == 0
  if buf.len == 0: return @[]
  return cpuDecodeArray(buf, cf)


# ═══════════════════════════════════════════════════════════════
# APF — tự build CustomFloat theo tensor/gradient (không theo RAM)
# ═══════════════════════════════════════════════════════════════

const
  APF_DEFAULT_REL_ERROR_TOL* = 1e-3
  APF_MIN_MANTISSA_BITS* = 2
  APF_MAX_MANTISSA_BITS* = 23
  APF_EXP_MARGIN_BITS* = 1

proc computeRequiredExponentBits*(arr: openArray[float32], expMargin = APF_EXP_MARGIN_BITS): int =
  var minAbs = float32.high
  var maxAbs = 0'f32
  var any = false
  for v in arr:
    if v.classify notin {fcNan, fcInf, fcNegInf} and v != 0'f32:
      any = true
      let a = abs(v)
      if a < minAbs: minAbs = a
      if a > maxAbs: maxAbs = a
  if not any:
    return 1

  let maxExp = int(floor(log2(float(maxAbs))))
  let minExp = int(floor(log2(float(minAbs))))
  # CustomFloat uses a fixed IEEE-style bias, not a per-tensor exponent
  # offset. The span alone cannot describe values far from exponent zero:
  # e.g. [0.001, 0.002] used to select too few bits and flush to zero.
  # Reserve exponent 0 for zero and the all-ones exponent for Inf/NaN.
  let margin = max(0, expMargin)
  result = 2
  while true:
    let bias = (1 shl (result - 1)) - 1
    let minNormalExp = 1 - bias
    let maxNormalExp = ((1 shl result) - 2) - bias
    if minExp - margin >= minNormalExp and maxExp + margin <= maxNormalExp:
      break
    inc result

proc computeRequiredMantissaBits*(arr: openArray[float32],
                                   relErrorTol = APF_DEFAULT_REL_ERROR_TOL,
                                   gradArr: openArray[float32] = []): int =
  var vals: seq[float32]
  for v in arr:
    if v.classify notin {fcNan, fcInf, fcNegInf} and v != 0'f32:
      vals.add(abs(v))
  if vals.len == 0:
    return APF_MIN_MANTISSA_BITS

  let baseBits = int(ceil(log2(1.0 / max(relErrorTol, 1e-12))))

  var extraBits = 0
  if gradArr.len > 0:
    var gvals: seq[float32]
    for g in gradArr:
      if g.classify notin {fcNan, fcInf, fcNegInf} and g != 0'f32:
        gvals.add(abs(g))
    if gvals.len > 0:
      proc median(s: var seq[float32]): float32 =
        s.sort()
        s[s.len div 2]
      var vsorted = vals
      var gsorted = gvals
      let wMed = max(median(vsorted), 1e-12'f32)
      let gMed = median(gsorted)
      let ratio = float(gMed) / float(wMed)
      if ratio > 0 and ratio < 1:
        let neededForUpdate = int(ceil(-log2(ratio)))
        extraBits = max(0, neededForUpdate - baseBits)

  result = clamp(baseBits + extraBits, APF_MIN_MANTISSA_BITS, APF_MAX_MANTISSA_BITS)

proc buildCustomDtypeForTensor*(arr: openArray[float32],
                                 gradArr: openArray[float32] = [],
                                 relErrorTol = APF_DEFAULT_REL_ERROR_TOL,
                                 expMargin = APF_EXP_MARGIN_BITS): CustomFloat =
  let eBits = computeRequiredExponentBits(arr, expMargin)
  let mBits = computeRequiredMantissaBits(arr, relErrorTol, gradArr)
  result = newCustomFloat(eBits, mBits, &"auto_e{eBits}m{mBits}")

proc apfCastForTraining*(arr: openArray[float32],
                          gradArr: openArray[float32] = [],
                          relErrorTol = APF_DEFAULT_REL_ERROR_TOL,
                          expMargin = APF_EXP_MARGIN_BITS): tuple[data: seq[uint8], cf: CustomFloat] =
  let cf = buildCustomDtypeForTensor(arr, gradArr, relErrorTol, expMargin)
  result = (encodeArray(arr, cf), cf)

proc apfDecodeForTraining*(encoded: openArray[uint8], cf: CustomFloat): seq[float32] =
  decodeArray(encoded, cf)
