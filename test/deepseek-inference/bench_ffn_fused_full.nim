## bench_ffn_fused_full.nim
## Đo throughput của TOÀN BỘ pipeline FFN gộp session (giống hệt
## fusedResidualNormFFN trong nim_inference.nim: RMSNorm scale + gate-matmul
## + up-matmul + bias + sigmoid + mul + mul + down-matmul + bias), TRONG
## CÙNG 1 sessionBegin/sessionEnd - khác với bench_ffn_matmul.nim chỉ đo
## 1 matmul đơn lẻ mỗi lần.
##
## Mục đích: bench_ffn_matmul.nim cho median ~6ms/matmul (3 matmul/layer
## ~18ms/layer), nhưng DECODE_PROFILE đo được FFN thật là ~42ms/layer.
## Bench này đo ĐÚNG những gì fusedResidualNormFFN làm để xác định 24ms
## chênh lệch nằm ở: overhead sessionUpload/sessionAllocScratch/sVecOp/
## sActivation xen giữa các matmul, hay driver Metal Intel không overlap
## được các lệnh liên tiếp trong 1 command buffer.
##
## Build:  nim c -d:release --hints:off --gc:arc --opt:speed -d:backend=metal bench_ffn_fused_full.nim
## Chạy:   ./bench_ffn_fused_full metal

import std/[times, random, algorithm, strformat, os]
import backend

randomize(1234)

proc buildFakeQ4Weight(inF, outF, groupSize: int):
    tuple[wq: seq[uint8], scales, zeros: seq[float32], nGroupsPerRow: int] =
  let nGroupsPerRow = (inF + groupSize - 1) div groupSize
  var wq = newSeq[uint8](outF * ((inF + 1) div 2))
  for i in 0 ..< wq.len: wq[i] = uint8(rand(255))
  var scales = newSeq[float32](outF * nGroupsPerRow)
  var zeros = newSeq[float32](outF * nGroupsPerRow)
  for i in 0 ..< scales.len:
    scales[i] = 0.01'f32 + rand(0.02'f32)
    zeros[i] = rand(15.0'f32)
  result = (wq, scales, zeros, nGroupsPerRow)

proc sVecOp(be: GpuBackend, op: string, a, b: SessionHandle, n: int): SessionHandle =
  result = sessionAllocScratch(be, n)
  doAssert sessionVecOp(be, op, a, b, result, n), "sVecOp(" & op & ") that bai"

proc sActivation(be: GpuBackend, op: string, a: SessionHandle, n: int): SessionHandle =
  result = sessionAllocScratch(be, n)
  doAssert sessionActivation(be, op, a, result, n), "sActivation(" & op & ") that bai"

proc benchFusedFFN(ctx: Backend, hidden, interF, groupSize, rows, nIter: int) =
  echo &"--- FFN fused full: hidden={hidden} inter={interF} rows={rows} groupSize={groupSize} ---"
  let be = ctx.kind.toByby()
  let n = rows * hidden
  let nInter = rows * interF

  let (gwq, gscales, gzeros, gGroups) = buildFakeQ4Weight(hidden, interF, groupSize)
  var wGate: ResidentWeightHandle = beResidentUploadQ4(ctx, gwq, gscales, gzeros, hidden, interF, groupSize, gGroups)
  let (uwq, uscales, uzeros, uGroups) = buildFakeQ4Weight(hidden, interF, groupSize)
  var wUp: ResidentWeightHandle = beResidentUploadQ4(ctx, uwq, uscales, uzeros, hidden, interF, groupSize, uGroups)
  let (dwq, dscales, dzeros, dGroups) = buildFakeQ4Weight(interF, hidden, groupSize)
  var wDown: ResidentWeightHandle = beResidentUploadQ4(ctx, dwq, dscales, dzeros, interF, hidden, groupSize, dGroups)

  var x2 = newSeq[float32](n)
  for i in 0 ..< n: x2[i] = rand(1.0'f32) - 0.5'f32
  var invBroadcast = newSeq[float32](n)
  var wBroadcast = newSeq[float32](n)
  for i in 0 ..< n:
    invBroadcast[i] = 1.0'f32
    wBroadcast[i] = 1.0'f32

  proc runOnce(): seq[float32] =
    doAssert sessionBegin(be), "sessionBegin that bai"
    var hX2 = sessionUpload(be, x2)
    var hInv = sessionUpload(be, invBroadcast)
    var hW = sessionUpload(be, wBroadcast)
    var hT1 = sVecOp(be, "mul", hX2, hInv, n)
    var hNormed = sVecOp(be, "mul", hT1, hW, n)
    var hGate = sessionAllocScratch(be, nInter)
    var hUp = sessionAllocScratch(be, nInter)
    doAssert sessionMatmulQ4(be, hNormed, wGate, hGate, rows)
    doAssert sessionMatmulQ4(be, hNormed, wUp, hUp, rows)
    var hSig = sActivation(be, "sigmoid", hGate, nInter)
    var hSilu = sVecOp(be, "mul", hGate, hSig, nInter)
    var hGated = sVecOp(be, "mul", hSilu, hUp, nInter)
    var hOut = sessionAllocScratch(be, rows * hidden)
    doAssert sessionMatmulQ4(be, hGated, wDown, hOut, rows)
    doAssert sessionEnd(be), "sessionEnd that bai"
    result = sessionRead(be, hOut, rows * hidden)
    sessionFree(be, hX2); sessionFree(be, hInv); sessionFree(be, hW)
    sessionFree(be, hT1); sessionFree(be, hNormed)
    sessionFree(be, hGate); sessionFree(be, hUp)
    sessionFree(be, hSig); sessionFree(be, hSilu); sessionFree(be, hGated)
    sessionFree(be, hOut)

  for i in 0 ..< 5:
    discard runOnce()

  var times: seq[float] = @[]
  for i in 0 ..< nIter:
    let t0 = epochTime()
    let y = runOnce()
    let t1 = epochTime()
    doAssert y.len == rows * hidden, "sai kich thuoc output"
    times.add((t1 - t0) * 1000.0)

  times.sort()
  let median = times[times.len div 2]
  let p10 = times[times.len div 10]
  let p90 = times[(times.len * 9) div 10]
  var total = 0.0
  for t in times: total += t
  let avg = total / times.len.float
  echo &"  median={median:.2f}ms  avg={avg:.2f}ms  p10={p10:.2f}ms  p90={p90:.2f}ms  " &
       &"(min={times[0]:.2f}ms max={times[^1]:.2f}ms, n={nIter})"

  beResidentFreeQ4(wGate)
  beResidentFreeQ4(wUp)
  beResidentFreeQ4(wDown)

when isMainModule:
  let backendName = if paramCount() > 0: paramStr(1) else: "metal"
  setForbidCpuFallback(true)
  echo &"Backend: {backendName}"
  let ctx = newBackend(backendName)

  discard beMatmul(ctx, @[0'f32, 0, 0, 0], 2, 2, @[0'f32, 0, 0, 0], 2, 2)

  const groupSize = 128
  const nIter = 200

  # shape thật deepseek-coder-6.7B: hidden=4096, intermediate=11008, rows=1 (decode)
  benchFusedFFN(ctx, 4096, 11008, groupSize, 1, nIter)

  echo ""
  echo "So sanh median o day voi 41.98ms cua DECODE_PROFILE FFN that."
  echo "Neu KHOP (~40ms) -> chinh viec gop nhieu lenh (RMSNorm+2matmul+"
  echo "sigmoid+2mul+matmul) trong 1 session la nguyen nhan, khong phai"
  echo "rieng kernel Q4 GEMV (da chung minh nhanh o bench_ffn_matmul.nim)."
  echo "Neu THAP HON NHIEU -> con nguyen nhan khac ngoai pipeline nay"
  echo "(vd. RMSNorm that dung sessionRmsnormEnc chua duoc bench o day,"
  echo "hoac chi phi ngoai FFN nhu addT/residual/dequant CPU con sot lai)."