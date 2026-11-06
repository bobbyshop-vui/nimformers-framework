## quant.nim
## Tensor quantization for Nimformer: int8, int4, fp8 (e4m3/e5m2), and "auto"
## (APF builds a CustomFloat from the tensor itself — reusing customfloat.nim).
## Plus a simple binary file format to save/load quantized state dicts.
##
## No main/test here — a pure library, imported elsewhere.

import std/math
import mapped_weights
export mapped_weights
import customfloat

type
  QuantKind* = enum
    qkFp32Raw   ## uncompressed, plain float32 (for bias/layernorm — error sensitive)
    qkInt8      ## symmetric int8, scale = max(|x|)/127
    qkInt4      ## symmetric int4 (2 values packed per byte), scale = max(|x|)/7
    qkInt4Asymmetric  ## THÊM: asymmetric int4 (Q4_K_M style), scale=(max-min)/15, zero_point
    qkFp8E4M3   ## CustomFloat(4,3) — uses customfloat.nim
    qkFp8E5M2   ## CustomFloat(5,2)
    qkCustom    ## any user-declared CustomFloat (any number of bits)
    qkAuto      ## APF: builds a CustomFloat from the tensor itself (buildCustomDtypeForTensor)
    qkFp4KM     ## NEW — "Q4_K_M but FP4": STANDARD OCP E2M1 (no codes wasted on
                ## inf/nan, round-to-nearest) + one FP8-E4M3 scale per GROUP of 32
                ## elements + one per-tensor FP32 scale. NO zero_point because FP4
                ## already has a sign bit -> exactly 0.25 bit/weight cheaper than Q4_K_M.
                ## 4 + 8/32 = 4.25 bits/weight. See quantizeFp4KM.

  QuantTensor* = object
    kind*: QuantKind
    shape*: seq[int]
    scale*: seq[float32]      ## ADDED: an array - len=1 (per-tensor), len=shape[0] (per-row),
                              ## or len=shape[0]*ceil(shape[1]/groupSize) (per-group, see groupSize)
    zero_point*: seq[float32] ## ADDED: an array - parallel to scale
    groupSize*: int      ## ADDED: 0 = per-tensor/per-row (the old behaviour, still compatible).
                         ## >0 = per-group: each row is split into ceil(nCols/groupSize) groups,
                         ## each group with its own scale/zero_point — almost as fine as the original GPTQ (group=128)
                         ## instead of one scale for a WHOLE 4096-column row (the old --q4km, too large an error
                         ## from a coarse double quantization — see quantize_int4_asymmetric_per_group
                         ## in export_to_nimq.py).
    cf*: CustomFloat     ## used for qkFp8E4M3/E5M2/qkCustom/qkAuto
    mappedData*: MappedView[uint8]
    data*: seq[uint8]

template bytes*(qt: QuantTensor): untyped =
  weightValues(qt.data, qt.mappedData)

proc clearPayload*(qt: var QuantTensor) =
  qt.data = @[]
  qt.mappedData = default(MappedView[uint8])

proc numelQ(shape: seq[int]): int =
  result = 1
  for s in shape: result *= s

# ─────────────────────────────────────────────────────────────
# int8 / int4 — plain hand-written symmetric quantization, no CustomFloat needed
# ─────────────────────────────────────────────────────────────

proc maxAbs(arr: openArray[float32]): float32 =
  result = 0'f32
  for v in arr:
    let a = abs(v)
    if a > result: result = a

proc quantizeInt8*(arr: openArray[float32], shape: seq[int]): QuantTensor =
  let m = maxAbs(arr)
  let scale = if m == 0'f32: 1'f32 else: m / 127'f32
  result.kind = qkInt8
  result.shape = shape
  result.scale = @[scale]
  result.zero_point = @[0'f32]
  result.data = newSeq[uint8](arr.len)
  for i, v in arr:
    let q = clamp(int(round(v / scale)), -127, 127)
    result.data[i] = cast[uint8](int8(q))

proc dequantizeInt8*(qt: QuantTensor): seq[float32] =
  let sc = qt.scale[0]
  result = newSeq[float32](qt.bytes.len)
  for i, b in qt.bytes:
    result[i] = float32(cast[int8](b)) * sc

proc quantizeInt4*(arr: openArray[float32], shape: seq[int]): QuantTensor =
  let m = maxAbs(arr)
  let scale = if m == 0'f32: 1'f32 else: m / 7'f32
  result.kind = qkInt4
  result.shape = shape
  result.scale = @[scale]
  result.zero_point = @[0'f32]
  let n = arr.len
  result.data = newSeq[uint8]((n + 1) div 2)
  for i in 0 ..< n:
    let q = clamp(int(round(arr[i] / scale)), -7, 7)
    let nibble = uint8(q and 0x0F)
    let byteIdx = i div 2
    if i mod 2 == 0:
      result.data[byteIdx] = (result.data[byteIdx] and 0xF0'u8) or nibble
    else:
      result.data[byteIdx] = (result.data[byteIdx] and 0x0F'u8) or (nibble shl 4)

proc dequantizeInt4*(qt: QuantTensor): seq[float32] =
  ## ADDED LUT: this path (not Q4_K_M/asymmetric-per-group, see
  ## dequantizeInt4Asymmetric below which already has a LUT) uses ONE
  ## scale for the whole tensor -> the LUT is built only ONCE for the whole function (not
  ## rebuilt per group like the asymmetric version), even cheaper. Before,
  ## every element computed (signed-8)*sc itself - now 16 computations once
  ## and a table lookup for all n elements.
  let sc = qt.scale[0]
  let n = numelQ(qt.shape)
  result = newSeq[float32](n)
  var lut: array[16, float32]
  for k in 0 ..< 16:
    var signed = k
    if signed >= 8: signed -= 16
    lut[k] = float32(signed) * sc
  for i in 0 ..< n:
    let byteIdx = i div 2
    var nibble: uint8
    if i mod 2 == 0: nibble = qt.bytes[byteIdx] and 0x0F'u8
    else: nibble = (qt.bytes[byteIdx] shr 4) and 0x0F'u8
    result[i] = lut[int(nibble)]

# ─────────────────────────────────────────────────────────────
# THÊM: asymmetric int4 (Q4_K_M style)
# ─────────────────────────────────────────────────────────────

proc quantizeInt4Asymmetric*(arr: openArray[float32], shape: seq[int]): QuantTensor =
  var minVal = float32.high
  var maxVal = -float32.high
  for v in arr:
    if v < minVal: minVal = v
    if v > maxVal: maxVal = v
  
  let range = maxVal - minVal
  let scale = if range < 1e-12'f32: 1'f32 else: range / 15'f32
  let zeroPoint = -minVal / scale
  
  result.kind = qkInt4Asymmetric
  result.shape = shape
  result.scale = @[scale]
  result.zero_point = @[zeroPoint]
  
  let n = arr.len
  result.data = newSeq[uint8]((n + 1) div 2)
  for i in 0 ..< n:
    let q = clamp(int(round(arr[i] / scale + zeroPoint)), 0, 15)
    let nibble = uint8(q and 0x0F)
    let byteIdx = i div 2
    if i mod 2 == 0:
      result.data[byteIdx] = (result.data[byteIdx] and 0xF0'u8) or nibble
    else:
      result.data[byteIdx] = (result.data[byteIdx] and 0x0F'u8) or (nibble shl 4)

proc quantizeInt4AsymmetricGrouped*(arr: openArray[float32], shape: seq[int],
                                    groupSize = 32): QuantTensor =
  ## Resident Q4 ABI: [out,in], low nibble first, rows padded to whole bytes,
  ## and one floating scale/zero point per contiguous input group.
  if shape.len != 2 or shape[0] <= 0 or shape[1] <= 0 or
      arr.len != shape[0] * shape[1] or groupSize <= 0 or groupSize mod 2 != 0:
    raise newException(ValueError, "grouped Q4 requires a matrix and an even positive group size")
  let rows = shape[0]
  let cols = shape[1]
  let groups = (cols + groupSize - 1) div groupSize
  let rowBytes = (cols + 1) div 2
  result = QuantTensor(kind: qkInt4Asymmetric, shape: shape, groupSize: groupSize,
    data: newSeq[uint8](rows * rowBytes),
    scale: newSeq[float32](rows * groups),
    zero_point: newSeq[float32](rows * groups))
  for row in 0 ..< rows:
    for group in 0 ..< groups:
      let first = group * groupSize
      let count = min(groupSize, cols - first)
      let base = row * cols + first
      let blockQ = quantizeInt4Asymmetric(arr.toOpenArray(base, base + count - 1), @[count])
      let gi = row * groups + group
      result.scale[gi] = blockQ.scale[0]
      result.zero_point[gi] = blockQ.zero_point[0]
      for i, value in blockQ.data:
        result.data[row * rowBytes + first div 2 + i] = value

proc dequantizeInt4Asymmetric*(qt: QuantTensor): seq[float32] =
  ## Per-row support: if qt.scale.len > 1 (== shape[0]), each row uses its own
  ## scale/zero_point (the packed data is per row too: each row rounds up to
  ## its own bytes - see quantize_int4_asymmetric_per_row in export_to_nimq.py).
  ## Per-group support (qt.groupSize > 0): each row is split further into
  ## ceil(nCols/groupSize) groups, one scale/zero_point per group (see
  ## quantize_int4_asymmetric_per_group in export_to_nimq.py) — almost as fine
  ## as the original GPTQ, avoiding the too coarse double-quantization error of per-row.
  let n = numelQ(qt.shape)
  result = newSeq[float32](n)
  var lut: array[16, float32]   ## ADDED: a 16-value lookup table (llama.cpp style) — int4 has only
                                 ## 16 possible codes (0..15) within a row/group (sharing
                                 ## one scale/zero_point), so the 16 values are computed ONCE and
                                 ## LOOKED UP per element — much faster than recomputing the
                                 ## (nibble-zp)*scale subtract + multiply for EVERY element.
  if qt.groupSize > 0:
    doAssert qt.shape.len == 2, "per-group int4 supports only 2D tensors"
    let nRows = qt.shape[0]
    let nCols = qt.shape[1]
    let bytesPerRow = (nCols + 1) div 2
    let nGroupsPerRow = (nCols + qt.groupSize - 1) div qt.groupSize
    for r in 0 ..< nRows:
      let rowByteOff = r * bytesPerRow
      let rowGroupOff = r * nGroupsPerRow
      var g = -1
      for c in 0 ..< nCols:
        let gNew = c div qt.groupSize
        if gNew != g:
          g = gNew
          let sc = qt.scale[rowGroupOff + g]
          let zp = qt.zero_point[rowGroupOff + g]
          for k in 0 ..< 16: lut[k] = (float32(k) - zp) * sc   # LUT recomputed only when entering a new group
        let byteIdx = rowByteOff + c div 2
        var nibble: uint8
        if c mod 2 == 0: nibble = qt.bytes[byteIdx] and 0x0F'u8
        else: nibble = (qt.bytes[byteIdx] shr 4) and 0x0F'u8
        result[r * nCols + c] = lut[int(nibble)]
  elif qt.scale.len > 1:
    doAssert qt.shape.len == 2, "per-row int4 supports only 2D tensors"
    let nRows = qt.shape[0]
    let nCols = qt.shape[1]
    let bytesPerRow = (nCols + 1) div 2
    for r in 0 ..< nRows:
      let sc = qt.scale[r]
      let zp = qt.zero_point[r]
      for k in 0 ..< 16: lut[k] = (float32(k) - zp) * sc   # LUT computed once per row, looked up for the whole row
      let rowByteOff = r * bytesPerRow
      for c in 0 ..< nCols:
        let byteIdx = rowByteOff + c div 2
        var nibble: uint8
        if c mod 2 == 0: nibble = qt.bytes[byteIdx] and 0x0F'u8
        else: nibble = (qt.bytes[byteIdx] shr 4) and 0x0F'u8
        result[r * nCols + c] = lut[int(nibble)]
  else:
    let sc = qt.scale[0]
    let zp = qt.zero_point[0]
    for i in 0 ..< n:
      let byteIdx = i div 2
      var nibble: uint8
      if i mod 2 == 0: nibble = qt.bytes[byteIdx] and 0x0F'u8
      else: nibble = (qt.bytes[byteIdx] shr 4) and 0x0F'u8
      result[i] = (float32(nibble) - zp) * sc

# ─────────────────────────────────────────────────────────────
# fp8 / custom / auto — built entirely on CustomFloat (customfloat.nim)
# ─────────────────────────────────────────────────────────────

proc customFloatMaxMagnitude(cf: CustomFloat): float32 =
  ## The largest magnitude this CustomFloat format CAN represent
  ## (excluding inf/nan - exponent field = maxExp is reserved for inf/nan).
  ## Used to compute the scale before quantizing - WITHOUT this step narrow-exponent
  ## formats (e.g. fp4 e2m1: bias=1, maxExp=3 -> the largest representable magnitude
  ## is only ~3.0, the SMALLEST (normal) ~1.0) silently send ALL real
  ## weights (usually of magnitude << 1.0) to 0 when not scaled first.
  if cf.exponentBits == 0:
    return 1'f32   # int8/int4 do not take this path (they have their own quantize functions)
  let maxRealExp = (cf.maxExp - 1) - cf.bias   # maxExp is for inf/nan, the largest valid exp is maxExp-1
  let maxMantissa = 2'f32 - pow(2'f32, -float32(cf.mantissaBits))  # the maximum 1.mantissa (implicit leading bit)
  result = pow(2'f32, float32(maxRealExp)) * maxMantissa

proc quantizeFloatDtype*(arr: openArray[float32], shape: seq[int], cf: CustomFloat, kind: QuantKind): QuantTensor =
  ## CHANGED: REALLY computes the scale (it used to hard-code scale=[1.0] and do NOTHING
  ## - narrow-exponent formats like fp4 quantized almost all real weights to 0, see
  ## the explanation in customFloatMaxMagnitude). Now: scale = maxAbs(arr) /
  ## maxMagnitude(cf), the data is divided by scale BEFORE encoding (into the
  ## representable range of the format), and the scale is stored so dequantizeTensor multiplies it back.
  ## BACKWARD SAFE: OLD .nimq files with scale=[1.0] (from before this fix)
  ## still work (dequantizeTensor now multiplies by qt.scale[0], 1.0 is a no-op).
  var buf: seq[float32]
  for v in arr: buf.add(v)
  let maxMag = customFloatMaxMagnitude(cf)
  let dataMax = maxAbs(buf)
  # scale > 1 means "shrink the data before encoding" (dataMax larger
  # than the representable range) - scale < 1 means "enlarge" (dataMax smaller than
  # the representable range, for fp4 with real weights ~0.01-0.1). If the data is
  # all 0 (dataMax=0), keep scale=1 to avoid dividing by 0.
  let scale = if dataMax > 0'f32: dataMax / maxMag else: 1'f32
  if scale != 0'f32:
    for i in 0 ..< buf.len: buf[i] = buf[i] / scale
  result.kind = kind
  result.shape = shape
  result.scale = @[scale]
  result.zero_point = @[0'f32]
  result.cf = cf
  result.data = encodeArray(buf, cf)

proc quantizeAuto*(arr: openArray[float32], shape: seq[int],
                    gradArr: openArray[float32] = []): QuantTensor =
  ## APF: builds a CustomFloat from the tensor itself (+gradient if any)
  var buf: seq[float32]
  for v in arr: buf.add(v)
  let cf = buildCustomDtypeForTensor(buf, gradArr)
  result = quantizeFloatDtype(buf, shape, cf, qkAuto)

# ─────────────────────────────────────────────────────────────
# Dispatcher chung
# ─────────────────────────────────────────────────────────────

proc quantizeRaw*(arr: openArray[float32], shape: seq[int]): QuantTensor =
  result.kind = qkFp32Raw
  result.shape = shape
  result.scale = @[1'f32]
  result.zero_point = @[0'f32]
  result.data = newSeq[uint8](arr.len * 4)
  for i, v in arr:
    var vv = v
    copyMem(addr result.data[i*4], addr vv, 4)

proc dequantizeRaw*(qt: QuantTensor): seq[float32] =
  let n = qt.bytes.len div 4
  result = newSeq[float32](n)
  for i in 0 ..< n:
    copyMem(addr result[i], unsafeAddr qt.bytes[i*4], 4)

# ═════════════════════════════════════════════════════════════
# FP4-KM  —  "Q4_K_M but FP4"
# ═════════════════════════════════════════════════════════════
#
# WHY: `quantizeFloatDtype` + FP4_E2M1 (the old path) uses ONE scale for the
# WHOLE tensor, plus 2 flaws of the E2M1 codec in customfloat.nim:
#   (a) maxExp is reserved for inf/nan -> 4/16 codes lost, the largest magnitude is only 3.0
#       instead of 6.0, a dynamic range of 4x instead of 12x;
#   (b) encodeArray TRUNCATES to 0 instead of rounding (0.90 -> 0.00).
# Measured result: ~99.9% of the weights of a 7B matrix forced to exactly 0, rel-L2 0.94.
#
# FP4-KM fixes all three:
#   1. Uses the STANDARD OCP E2M1 level set: {0, .5, 1, 1.5, 2, 3, 4, 6} x sign,
#      all 16 codes, NONE wasted on inf/nan.
#   2. Round-to-nearest, plus a minimum-MSE scale search (like llama.cpp's
#      make_qx_quants).
#   3. A scale per GROUP of 32 elements, stored as one FP8-E4M3 byte, normalised by
#      one per-tensor FP32 scale (exactly Blackwell's NVFP4 layout).
#
# HOW IT DIFFERS FROM Q4_K_M: Q4_K_M must store BOTH a scale AND a min per sub-block because
# int4 has no sign (6+6 bits / 32 elements = 4.5 bpw). FP4 already has a sign bit, so it is
# symmetric around 0 and needs only a scale -> 4 + 8/32 = 4.25 bpw.
#
# LAYOUT of `data` (one flat byte block, the .nimq file format is unchanged):
#     [0                 , R*C/2            )  nibbles, 2 elements/byte
#     [R*C/2             , R*C/2 + R*nGroups)  one E4M3 scale byte per group
# Within each group of 32 elements, byte j (j=0..15) holds element j in the LOW nibble and
# element j+16 in the HIGH nibble — llama.cpp style, so lanes 0..15 of a
# simdgroup read 16 contiguous bytes (coalesced).
# `qt.scale = @[globalScale]`, `qt.groupSize = 32`.

const Fp4kmMags*: array[8, float32] =
  [0'f32, 0.5'f32, 1.0'f32, 1.5'f32, 2.0'f32, 3.0'f32, 4.0'f32, 6.0'f32]

proc fp4kmDecodeLut*(): array[16, float32] =
  ## nibble = (sign shl 3) or magIdx.  magIdx = (e shl 1) or m of standard E2M1.
  for i in 0 ..< 8:
    result[i] = Fp4kmMags[i]
    result[8 + i] = -Fp4kmMags[i]

proc fp4kmMagIndex(a: float32): int {.inline.} =
  ## The NEAREST magnitude index (round-to-nearest). Thresholds = midpoints.
  if a < 0.25'f32: 0
  elif a < 0.75'f32: 1
  elif a < 1.25'f32: 2
  elif a < 1.75'f32: 3
  elif a < 2.5'f32: 4
  elif a < 3.5'f32: 5
  elif a < 5.0'f32: 6
  else: 7

proc fp4kmGroupErr(g: openArray[float32], n: int, invS, s: float32): float32 {.inline.} =
  var acc = 0'f32
  for i in 0 ..< n:
    let t = g[i] * invS
    let a = abs(t)
    let q = Fp4kmMags[fp4kmMagIndex(a)] * s
    let d = (if t < 0'f32: -q else: q) - g[i]
    acc += d * d
  acc

proc fp4kmMaxAbs(g: openArray[float32], n: int): float32 {.inline.} =
  result = 0'f32
  for i in 0 ..< n:
    if abs(g[i]) > result: result = abs(g[i])

proc fp4kmBestScaleByte(g: openArray[float32], n: int, globalScale: float32,
                        e4m3Lut: array[256, float32], searchSteps: int):
                        tuple[b: uint8, s: float32] =
  ## Minimum-MSE scale search, but ONLY over the scales that CAN BE STORED
  ## (E4M3 values after decoding). Searching in fp32 and then forcing to E4M3 is
  ## USELESS - measurements show no improvement, because rounding the
  ## scale wipes out the search result. This is the difference from llama.cpp's make_qx_quants
  ## (which stores the scale in fp16, almost continuous).
  let mx = fp4kmMaxAbs(g, n)
  if mx <= 0'f32 or globalScale <= 0'f32: return (0'u8, 0'f32)
  let base = mx / 6.0'f32
  var bestB = 0'u8
  var bestS = 0'f32
  var bestE = Inf.float32
  let steps = max(searchSteps, 1)
  for k in 0 ..< steps:
    let r = if steps == 1: 1.0'f32
            else: 0.62'f32 + 0.62'f32 * float32(k) / float32(steps - 1)
    let cand = base * r
    let enc = encodeArray([cand / globalScale], FP8_E4M3)
    let b = enc[0]
    let dec = e4m3Lut[int(b)]
    if dec <= 0'f32 or dec > 1e30'f32 or dec != dec: continue
    let s = dec * globalScale
    if s <= 0'f32: continue
    let e = fp4kmGroupErr(g, n, 1'f32 / s, s)
    if e < bestE:
      bestE = e; bestB = b; bestS = s
  (bestB, bestS)

proc quantizeFp4KM*(arr: openArray[float32], shape: seq[int],
                    groupSize: int = 32, searchSteps: int = 8): QuantTensor =
  doAssert shape.len == 2, "quantizeFp4KM: only for 2D tensors [rows, cols]"
  let R = shape[0]
  let C = shape[1]
  doAssert C mod groupSize == 0,
    "quantizeFp4KM: cols (" & $C & ") must be divisible by groupSize (" & $groupSize & ")"
  doAssert groupSize mod 2 == 0, "quantizeFp4KM: groupSize must be even"
  doAssert arr.len == R * C, "quantizeFp4KM: arr.len does not match shape"
  let nG = C div groupSize
  let half = groupSize div 2

  # ── pass 1: only max|w| per group is needed to pick the per-tensor scale ──
  var buf = newSeq[float32](groupSize)
  var maxScale = 0'f32
  for r in 0 ..< R:
    for j in 0 ..< nG:
      let off = r * C + j * groupSize
      var mx = 0'f32
      for t in 0 ..< groupSize:
        if abs(arr[off + t]) > mx: mx = abs(arr[off + t])
      let s = mx / 6.0'f32
      if s > maxScale: maxScale = s

  # ── per-tensor scale to bring the group scales into the "sweet spot" of E4M3 ──
  # (exactly the NVFP4 layout: one per-tensor FP32 + one FP8 per block).
  # 224 is a safe E4M3 value below the 240 ceiling of the repo's codec.
  let globalScale = if maxScale > 0'f32: maxScale / 224.0'f32 else: 1'f32

  # ── pass 2: picks the minimum-MSE E4M3 scale byte DIRECTLY for each group ──
  # The weights MUST be quantized with EXACTLY the scale that will be stored, not an
  # intermediate fp32 scale - otherwise the two would disagree.
  var scaleBytes = newSeq[uint8](R * nG)
  let e4m3Lut = buildDecodeLUT(FP8_E4M3)
  var gsReal = newSeq[float32](R * nG)
  for r in 0 ..< R:
    for j in 0 ..< nG:
      let off = r * C + j * groupSize
      for t in 0 ..< groupSize: buf[t] = arr[off + t]
      let (b, s) = fp4kmBestScaleByte(buf, groupSize, globalScale, e4m3Lut, searchSteps)
      scaleBytes[r * nG + j] = b
      gsReal[r * nG + j] = s

  # ── pass 3: pack the nibbles ──
  var data = newSeq[uint8](R * C div 2 + R * nG)
  for r in 0 ..< R:
    for j in 0 ..< nG:
      let gi = r * nG + j
      let s = gsReal[gi]
      let off = r * C + j * groupSize
      let blk = gi * half
      if s <= 0'f32:
        for t in 0 ..< half: data[blk + t] = 0'u8
        continue
      let invS = 1'f32 / s
      for t in 0 ..< half:
        let vLo = arr[off + t] * invS
        let vHi = arr[off + t + half] * invS
        let cLo = uint8(fp4kmMagIndex(abs(vLo))) or (if vLo < 0'f32: 8'u8 else: 0'u8)
        let cHi = uint8(fp4kmMagIndex(abs(vHi))) or (if vHi < 0'f32: 8'u8 else: 0'u8)
        data[blk + t] = cLo or (cHi shl 4)
  copyMem(addr data[R * C div 2], addr scaleBytes[0], R * nG)

  result.kind = qkFp4KM
  result.shape = shape
  result.scale = @[globalScale]
  result.zero_point = @[0'f32]
  result.groupSize = groupSize
  result.cf = FP8_E4M3            # records the codec used for the scale
  result.data = data

proc dequantizeFp4KM*(qt: QuantTensor): seq[float32] =
  doAssert qt.shape.len == 2, "dequantizeFp4KM: needs a 2D tensor"
  let R = qt.shape[0]
  let C = qt.shape[1]
  let g = if qt.groupSize > 0: qt.groupSize else: 32
  let nG = C div g
  let half = g div 2
  let nibEnd = R * C div 2
  let globalScale = if qt.scale.len > 0: qt.scale[0] else: 1'f32
  let lut = fp4kmDecodeLut()
  let e4m3Lut = buildDecodeLUT(FP8_E4M3)
  result = newSeq[float32](R * C)
  for r in 0 ..< R:
    for j in 0 ..< nG:
      let gi = r * nG + j
      let sb = qt.bytes[nibEnd + gi]
      var s = e4m3Lut[int(sb)] * globalScale
      if s > 1e30'f32 or s != s: s = 0'f32
      let off = r * C + j * g
      let blk = gi * half
      for t in 0 ..< half:
        let b = qt.bytes[blk + t]
        result[off + t]        = lut[int(b and 0x0F'u8)] * s
        result[off + t + half] = lut[int(b shr 4)] * s

proc quantizeTensor*(arr: openArray[float32], shape: seq[int], kind: QuantKind,
                      customCf: CustomFloat = FP24, gradArr: openArray[float32] = []): QuantTensor =
  case kind
  of qkFp32Raw: quantizeRaw(arr, shape)
  of qkInt8:    quantizeInt8(arr, shape)
  of qkInt4:    quantizeInt4(arr, shape)
  of qkInt4Asymmetric: quantizeInt4Asymmetric(arr, shape)  # THÊM
  of qkFp8E4M3: quantizeFloatDtype(arr, shape, FP8_E4M3, qkFp8E4M3)
  of qkFp8E5M2: quantizeFloatDtype(arr, shape, FP8_E5M2, qkFp8E5M2)
  of qkCustom:  quantizeFloatDtype(arr, shape, customCf, qkCustom)
  of qkAuto:    quantizeAuto(arr, shape, gradArr)
  of qkFp4KM:   quantizeFp4KM(arr, shape)

proc dequantizeTensor*(qt: QuantTensor): seq[float32] =
  case qt.kind
  of qkFp32Raw: dequantizeRaw(qt)
  of qkInt8:    dequantizeInt8(qt)
  of qkInt4:    dequantizeInt4(qt)
  of qkInt4Asymmetric: dequantizeInt4Asymmetric(qt)  # THÊM
  of qkFp4KM:   dequantizeFp4KM(qt)
  of qkFp8E4M3, qkFp8E5M2, qkCustom, qkAuto:
    # CHANGED: MULTIPLIES BACK by qt.scale[0] (it was ignored entirely, always treated as
    # scale=1 - right for OLD files (the real scale was 1.0, a no-op) but WRONG for
    # NEW files from the fixed quantizeFloatDtype (with a real scale, see the explanation
    # there) - without multiplying back here, decoding gives tiny/zero values completely
    # wrong compared to the original weights.
    let raw = decodeArray(qt.bytes, qt.cf)
    let s = if qt.scale.len > 0: qt.scale[0] else: 1'f32
    if s == 1'f32: raw
    else:
      var res = newSeq[float32](raw.len)
      for i in 0 ..< raw.len: res[i] = raw[i] * s
      res

proc dequantizeTensorTransposed*(qt: QuantTensor): seq[float32] =
  ## ADDED: dequant + transpose MERGED INTO ONE PASS, written directly into the target layout
  ## [nCols, nRows] (transposed) instead of dequantizeTensor() (giving [nRows,nCols])
  ## and a separate transpose() — the old way allocated 2 full-size fp32 arrays at once
  ## (Linear.forward calls this on every forward for large weights, e.g. lm_head/
  ## the q/k/v/o/gate/up/down projections of a 6.7B model), now only 1 array.
  ## Uses a 16-value LUT (llama.cpp style) for int4/int4-asymmetric because each
  ## row/group has only 16 possible codes — the 16 values are precomputed and looked up for
  ## each element, instead of recomputing the subtract+multiply per element.
  ## FIX: supports 1D tensors (bias, LN gamma/beta) — treated as a single row (1 x n)
  ## to avoid a crash when Linear.useQuant=true but weightQ is 1D (quantBase
  ## holds 1D bias/gamma too from NIMBIN4 checkpoints).
  if qt.shape.len <= 1:
    return dequantizeTensor(qt)  # 0D/1D: no transpose needed, returned as is
  let nRows = qt.shape[0]
  let nCols = qt.shape[1]
  result = newSeq[float32](nRows * nCols)
  case qt.kind
  of qkInt4Asymmetric:
    var lut: array[16, float32]
    let bytesPerRow = (nCols + 1) div 2
    if qt.groupSize > 0:
      let nGroupsPerRow = (nCols + qt.groupSize - 1) div qt.groupSize
      for r in 0 ..< nRows:
        let rowByteOff = r * bytesPerRow
        let rowGroupOff = r * nGroupsPerRow
        var g = -1
        for c in 0 ..< nCols:
          let gNew = c div qt.groupSize
          if gNew != g:
            g = gNew
            let sc = qt.scale[rowGroupOff + g]
            let zp = qt.zero_point[rowGroupOff + g]
            for k in 0 ..< 16: lut[k] = (float32(k) - zp) * sc
          let byteIdx = rowByteOff + c div 2
          var nibble: uint8
          if c mod 2 == 0: nibble = qt.bytes[byteIdx] and 0x0F'u8
          else: nibble = (qt.bytes[byteIdx] shr 4) and 0x0F'u8
          result[c * nRows + r] = lut[int(nibble)]   # written straight in the transposed layout
    elif qt.scale.len > 1:
      for r in 0 ..< nRows:
        let sc = qt.scale[r]
        let zp = qt.zero_point[r]
        for k in 0 ..< 16: lut[k] = (float32(k) - zp) * sc
        let rowByteOff = r * bytesPerRow
        for c in 0 ..< nCols:
          let byteIdx = rowByteOff + c div 2
          var nibble: uint8
          if c mod 2 == 0: nibble = qt.bytes[byteIdx] and 0x0F'u8
          else: nibble = (qt.bytes[byteIdx] shr 4) and 0x0F'u8
          result[c * nRows + r] = lut[int(nibble)]
    else:
      let sc = qt.scale[0]
      let zp = qt.zero_point[0]
      for k in 0 ..< 16: lut[k] = (float32(k) - zp) * sc
      for r in 0 ..< nRows:
        let rowByteOff = r * bytesPerRow
        for c in 0 ..< nCols:
          let byteIdx = rowByteOff + c div 2
          var nibble: uint8
          if c mod 2 == 0: nibble = qt.bytes[byteIdx] and 0x0F'u8
          else: nibble = (qt.bytes[byteIdx] shr 4) and 0x0F'u8
          result[c * nRows + r] = lut[int(nibble)]
  of qkInt4:
    let sc = qt.scale[0]
    let bytesPerRow = (nCols + 1) div 2
    var lut: array[16, float32]
    for k in 0 ..< 16:
      var signed = int(k)
      if signed >= 8: signed -= 16
      lut[k] = float32(signed) * sc
    for r in 0 ..< nRows:
      let rowByteOff = r * bytesPerRow
      for c in 0 ..< nCols:
        let byteIdx = rowByteOff + c div 2
        var nibble: uint8
        if c mod 2 == 0: nibble = qt.bytes[byteIdx] and 0x0F'u8
        else: nibble = (qt.bytes[byteIdx] shr 4) and 0x0F'u8
        result[c * nRows + r] = lut[int(nibble)]
  of qkInt8:
    let sc = qt.scale[0]
    for r in 0 ..< nRows:
      let rowOff = r * nCols
      for c in 0 ..< nCols:
        result[c * nRows + r] = float32(cast[int8](qt.bytes[rowOff + c])) * sc
  else:
    # Rare (fp32 raw / fp8 / custom / auto) — not a hot path
    # (the hot q/k/v/o/gate/up/down/lm_head path always uses per-group
    # int4 asymmetric), so a normal dequant then transpose (2 temporary arrays) is acceptable.
    let flat = dequantizeTensor(qt)
    for r in 0 ..< nRows:
      let rowOff = r * nCols
      for c in 0 ..< nCols:
        result[c * nRows + r] = flat[rowOff + c]

# ─────────────────────────────────────────────────────────────
# Binary I/O — saves/loads one state dict (list of (name, QuantTensor)) + the architecture
# ─────────────────────────────────────────────────────────────

const MagicStr = "NIMQ2"   ## ADDED: bumped NIMQ1 -> NIMQ2 for the groupSize field (per-group int4).
                            ## Old .nimq files (NIMQ1) must be re-exported with the new export_to_nimq.py.
const MagicAligned* = "NIMQ3"
  ## NIMQ2 plus an int32 payload alignment after the magic, padding up to that
  ## alignment before the arch fields, and each tensor's packed payload
  ## starting at a multiple of it. Lets a GPU on unified memory wrap each
  ## payload of the mmap in place. The header padding also guarantees every
  ## NIMQ2 byte moves forward by >= align - 9 bytes when a file is converted
  ## in place (convert_nimq_aligned).
const NimqPayloadAlign* = 65536
  ## Default for NIMQ3: covers 4 KiB (x86) and 16 KiB (Apple Silicon) pages and
  ## Vulkan host-pointer import alignments.

proc writeString(f: File, s: string) =
  var n = int32(s.len)
  discard f.writeBuffer(addr n, sizeof(n))
  if s.len > 0: discard f.writeBuffer(unsafeAddr s[0], s.len)

proc readString(f: File): string =
  var n: int32
  discard f.readBuffer(addr n, sizeof(n))
  result = newString(n)
  if n > 0: discard f.readBuffer(addr result[0], n)

proc writeQuantTensor(f: File, qt: QuantTensor; payloadAlign = 0) =
  var kindByte = uint8(ord(qt.kind))
  discard f.writeBuffer(addr kindByte, 1)
  var ndims = int32(qt.shape.len)
  discard f.writeBuffer(addr ndims, 4)
  for d in qt.shape:
    var dv = int32(d)
    discard f.writeBuffer(addr dv, 4)
  var nScales = int32(qt.scale.len)
  discard f.writeBuffer(addr nScales, 4)
  if qt.scale.len > 0:
    discard f.writeBuffer(unsafeAddr qt.scale[0], qt.scale.len * 4)
  if qt.zero_point.len > 0:
    discard f.writeBuffer(unsafeAddr qt.zero_point[0], qt.zero_point.len * 4)
  var gs = int32(qt.groupSize)
  discard f.writeBuffer(addr gs, 4)
  var eb = int32(max(qt.cf.exponentBits, 1))
  var mb = int32(qt.cf.mantissaBits)
  discard f.writeBuffer(addr eb, 4)
  discard f.writeBuffer(addr mb, 4)
  var dlen = int32(qt.bytes.len)
  discard f.writeBuffer(addr dlen, 4)
  if payloadAlign > 1:
    let pad = (payloadAlign - int(f.getFilePos() mod payloadAlign)) mod payloadAlign
    if pad > 0:
      var zeros = newSeq[uint8](pad)
      discard f.writeBuffer(addr zeros[0], pad)
  if qt.bytes.len > 0:
    discard f.writeBuffer(unsafeAddr qt.bytes[0], qt.bytes.len)

proc readQuantTensor(f: File): QuantTensor =
  var kindByte: uint8
  discard f.readBuffer(addr kindByte, 1)
  result.kind = QuantKind(kindByte)
  var ndims: int32
  discard f.readBuffer(addr ndims, 4)
  result.shape = newSeq[int](ndims)
  for i in 0 ..< ndims:
    var dv: int32
    discard f.readBuffer(addr dv, 4)
    result.shape[i] = int(dv)
  var nScales: int32
  discard f.readBuffer(addr nScales, 4)
  result.scale = newSeq[float32](nScales)
  if nScales > 0:
    discard f.readBuffer(addr result.scale[0], nScales * 4)
  result.zero_point = newSeq[float32](nScales)
  if nScales > 0:
    discard f.readBuffer(addr result.zero_point[0], nScales * 4)
  var gs: int32
  discard f.readBuffer(addr gs, 4)
  result.groupSize = int(gs)
  var eb, mb: int32
  discard f.readBuffer(addr eb, 4)
  discard f.readBuffer(addr mb, 4)
  result.cf = newCustomFloat(int(eb), int(mb))
  var dlen: int32
  discard f.readBuffer(addr dlen, 4)
  result.data = newSeq[uint8](dlen)
  if dlen > 0:
    discard f.readBuffer(addr result.data[0], dlen)

proc rawFloatView*(qt: QuantTensor): MappedView[float32] =
  ## The float32 payload of a mapped qkFp32Raw tensor, read in place (NIMQ2
  ## payloads may be unaligned; callers copy rows with copyMem).
  ## owner == nil when the tensor is not mapped raw fp32.
  if qt.kind == qkFp32Raw and qt.mappedData.owner != nil and qt.mappedData.len mod 4 == 0:
    result = mappedView[float32](qt.mappedData.owner, qt.mappedData.offset,
      qt.mappedData.len div 4)

proc readMappedQuantTensor*(c: var WeightCursor; mapped = true;
                            payloadAlign = 0): QuantTensor =
  let kind = readValue[uint8](c)
  if kind > uint8(high(QuantKind)): raise newException(IOError, "Invalid quant kind")
  result.kind = QuantKind(kind)
  let ndims = c.readCount(4)
  if ndims > 8: raise newException(IOError, "Invalid tensor rank")
  for i in 0 ..< ndims:
    let d = int(readValue[int32](c))
    if d <= 0: raise newException(IOError, "Invalid tensor dimension")
    result.shape.add d
  let ns = c.readCount(4)
  result.scale = readSeq[float32](c, ns)
  result.zero_point = readSeq[float32](c, ns)
  result.groupSize = int(readValue[int32](c))
  let eb = int(readValue[int32](c))
  let mb = int(readValue[int32](c))
  if eb < 0 or mb < 0 or eb > 8 or mb > 23:
    raise newException(IOError, "Invalid custom float format")
  result.cf = newCustomFloat(eb, mb)
  let n = c.readCount()
  if payloadAlign > 1:
    c.pos += (payloadAlign - c.pos mod payloadAlign) mod payloadAlign
    c.owner.checkRange(c.pos, n)
  if mapped:
    result.mappedData = mappedView[uint8](c.owner, c.pos, n)
    c.pos += n
  else:
    result.data = readSeq[uint8](c, n)

proc saveQuantStateDict*(path: string, arch: array[5, int], sd: seq[(string, QuantTensor)];
                         payloadAlign = 0) =
  ## payloadAlign > 0 writes NIMQ3 (payloads aligned for in-place GPU use);
  ## 0 keeps the NIMQ2 layout that export_to_nimq.py produces.
  if payloadAlign < 0 or payloadAlign > (1 shl 24) or
      (payloadAlign > 0 and (payloadAlign < 64 or (payloadAlign and (payloadAlign - 1)) != 0)):
    raise newException(ValueError, "Invalid NIMQ payload alignment")
  var f = open(path, fmWrite)
  defer: f.close()
  if payloadAlign > 0:
    writeString(f, MagicAligned)
    var a = int32(payloadAlign)
    discard f.writeBuffer(addr a, 4)
    var headerPad = newSeq[uint8](payloadAlign - int(f.getFilePos() mod payloadAlign))
    discard f.writeBuffer(addr headerPad[0], headerPad.len)
  else:
    writeString(f, MagicStr)
  var archI32: array[5, int32]
  for i, v in arch: archI32[i] = int32(v)
  discard f.writeBuffer(addr archI32[0], sizeof(archI32))
  var n = int32(sd.len)
  discard f.writeBuffer(addr n, 4)
  for (name, qt) in sd:
    writeString(f, name)
    writeQuantTensor(f, qt, payloadAlign)
  if payloadAlign > 1:
    # GPUs wrap each payload rounded up to their page/import alignment; the
    # last one must not run past the end of the mapped file.
    let pad = (payloadAlign - int(f.getFilePos() mod payloadAlign)) mod payloadAlign
    if pad > 0:
      var zeros = newSeq[uint8](pad)
      discard f.writeBuffer(addr zeros[0], pad)

proc loadQuantStateDict*(path: string; mapped = false): tuple[arch: array[5, int],
    sd: seq[(string, QuantTensor)], payloadAlign: int] =
  ## Reads NIMQ2 and NIMQ3. payloadAlign is 0 for NIMQ2.
  var c = WeightCursor(owner: mapWeights(path))
  let magic = c.readString()
  if magic == MagicAligned:
    result.payloadAlign = int(readValue[int32](c))
    if result.payloadAlign <= 13 or result.payloadAlign > (1 shl 24):
      raise newException(IOError, "Invalid NIMQ payload alignment: " & path)
    c.pos = result.payloadAlign
  elif magic != MagicStr:
    raise newException(IOError, "Invalid NIMQ checkpoint: " & path)
  for i in 0 ..< 5: result.arch[i] = int(readValue[int32](c))
  let n = c.readCount()
  for i in 0 ..< n:
    let name = c.readString()
    result.sd.add((name, readMappedQuantTensor(c, mapped, result.payloadAlign)))
