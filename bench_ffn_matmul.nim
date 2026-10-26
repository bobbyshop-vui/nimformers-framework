## bench_ffn_matmul.nim
## Đo throughput THẬT của sessionMatmulQ4 (kernel Q4 GEMV, M=1 - đúng shape
## dùng trong decode) cô lập hoàn toàn khỏi: load model 4.4GB, tokenizer,
## Python/torch warmup, và "lần gọi matmul đầu tiên" (Metal compile pipeline
## + pickMatmulPipeline - xem comment ở nim_inference.nim dòng ~2349).
##
## Mục đích: tách 2 giả thuyết ra khỏi nhau -
##   (a) 40ms/gọi là chi phí TÍNH TOÁN thật của kernel Q4 GEMV trên Iris 640
##   (b) 40ms/gọi là nhiễu (thermal throttle, background load, GC Nim,...)
##      lẫn vào phép đo DECODE_PROFILE (chỉ đo layer 0, 2 token đầu - mẫu quá
##      nhỏ, dễ dính spike)
##
## Build:  nim c -d:release -o:bench_ffn_matmul bench_ffn_matmul.nim
## Chạy:   ./bench_ffn_matmul
##
## Đọc kết quả: so sánh median ms/call ở ĐÂY với 40ms DECODE_PROFILE đo được.
## Nếu khớp nhau (~cùng độ lớn) -> (a) đúng, kernel Q4 GEMV chính là bottleneck,
## cần tối ưu bên bybylang (tiling/threadgroup/vectorize dequant int4).
## Nếu ở đây thấp hơn NHIỀU (vd. <10ms) -> (b) đúng, đường resident/kernel
## không phải vấn đề, đi tìm nhiễu hệ thống (Activity Monitor GPU History,
## `pmset -g therm`, tắt bớt app nền, cắm sạc thay vì chạy pin).

import std/[times, random, algorithm, strformat, os]
import backend

randomize(1234)

proc packedInt4Row(inF, groupSize: int): seq[uint8] =
  ## nGroupsPerRow*ceil(groupSize/2) byte/hàng thực ra layout thật là
  ## ceil(inF/2) byte/hàng (2 giá trị int4 gói / byte), như
  ## quantize_int4_asymmetric_per_group xuất ra ở export_to_nimq.py.
  result = newSeq[uint8]((inF + 1) div 2)
  for i in 0 ..< result.len:
    result[i] = uint8(rand(255))

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

proc benchOne(ctx: Backend, label: string, inF, outF, groupSize, nIter: int) =
  echo &"--- {label}: inF={inF} outF={outF} groupSize={groupSize} (M=1, giong het shape decode) ---"
  let (wq, scales, zeros, nGroupsPerRow) = buildFakeQ4Weight(inF, outF, groupSize)
  let wHandle = beResidentUploadQ4(ctx, wq, scales, zeros, inF, outF, groupSize, nGroupsPerRow)
  var w: ResidentWeightHandle = wHandle

  var x = newSeq[float32](inF)
  for i in 0 ..< inF: x[i] = rand(1.0'f32) - 0.5'f32

  # Warmup: cho Metal compile pipeline + pickMatmulPipeline chạy xong TRƯỚC
  # khi tính giờ - đúng lý do generate() gọi 1 matmul "mồi" trước lúc đo
  # thật (xem nim_inference.nim khoi_dong_gpu_pipeline).
  for i in 0 ..< 5:
    discard beResidentLinearQ4(ctx, x, w, 1)

  var times: seq[float] = @[]
  for i in 0 ..< nIter:
    let t0 = epochTime()
    let y = beResidentLinearQ4(ctx, x, w, 1)
    let t1 = epochTime()
    doAssert y.len == outF, "matmul tra ve sai kich thuoc - kernel loi, ket qua benchmark khong dang tin"
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
  beResidentFreeQ4(w)

when isMainModule:
  let backendName = if paramCount() > 0: paramStr(1) else: "metal"
  setForbidCpuFallback(true)
  echo &"Backend: {backendName}"
  let ctx = newBackend(backendName)

  # Mồi pipeline chung 1 lần trước tất cả (giống hệt generate() thật) để
  # không tính nhầm chi phí "compile lần đầu" vào bench nào cả.
  discard beMatmul(ctx, @[0'f32, 0, 0, 0], 2, 2, @[0'f32, 0, 0, 0], 2, 2)

  const groupSize = 128
  const nIter = 200

  # 3 shape thật của FFN deepseek-coder-6.7B (hidden=4096, intermediate=11008)
  benchOne(ctx, "FFN gate/up proj (4096 -> 11008)", 4096, 11008, groupSize, nIter)
  benchOne(ctx, "FFN down proj (11008 -> 4096)", 11008, 4096, groupSize, nIter)
  # shape attention oProj/qProj (hidden -> hidden) de doi chieu voi so lieu
  # "Attention-loop + oProj" trong DECODE_PROFILE
  benchOne(ctx, "Attn q/k/v/o proj (4096 -> 4096)", 4096, 4096, groupSize, nIter)

  echo ""
  echo "So sanh: 1 layer FFN that = 2x(gate/up, cung shape) + 1x(down)."
  echo "Tong uoc tinh 1 layer FFN = 2*median(gate/up) + median(down)."
  echo "Nhan x32 layer se ra uoc tinh FFN toan bo 1 token - doi chieu voi"
  echo "129.694s / 99 token = ~1310ms/token ma log thuc te dang cho."