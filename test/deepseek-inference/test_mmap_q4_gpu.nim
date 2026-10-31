## Weights served from a model mmap (BYBY_INFER_BACKEND=metal|vulkan). On
## unified-memory GPUs both NIMQ2 (payloads at any byte offset) and NIMQ3
## (page-aligned) payloads are wrapped in place and the kernels take the byte
## offset; elsewhere they are copied from the mapping. Every path must match
## the CPU reference. Run once more with BYBY_Q4_TUNED=wide to force the
## tuned wide Q4 kernels instead of the legacy ones.
import std/[os, math, strformat, sets]
import backend, quant, mapped_weights
import vendor/bybylang/gpubackend as byby

proc checkQ4(be: GpuBackend; w: ResidentWeightHandle; wDense: seq[float32];
             K, N, M: int; label: string) =
  var x = newSeq[float32](M * K)
  for i in 0 ..< x.len: x[i] = sin(float32(i) * 0.37'f32)
  var hx = byby.sessionUpload(be, x)
  var hy: SessionHandle
  doAssert byby.sessionBegin(be)
  doAssert sessionMatmulQ4(be, hx, w, hy, M)
  doAssert byby.sessionEnd(be), byby.sessionGetLastError(be)
  let got = byby.sessionRead(be, hy, M * N)
  byby.sessionFree(be, hx); byby.sessionFree(be, hy)
  var err, peak = 0'f64
  for m in 0 ..< M:
    for n in 0 ..< N:
      var s = 0'f64
      for k in 0 ..< K: s += float64(x[m * K + k]) * float64(wDense[n * K + k])
      err = max(err, abs(s - float64(got[m * N + n]))); peak = max(peak, abs(s))
  doAssert err <= max(1e-4, peak * 1e-4), &"{label} M={M}: error={err} peak={peak}"

proc q4FromMmap() =
  let ctx = newBackend(getEnv("BYBY_INFER_BACKEND", "metal"))
  let be = ctx.kind.toByby()
  let dir = getTempDir() / ("mmap-q4-" & $getCurrentProcessId())
  createDir(dir)
  defer: removeDir(dir)
  const group = 128
  var sd: seq[(string, QuantTensor)]
  var seed = 12345'u32
  proc rnd(): uint32 =
    seed = seed * 1664525'u32 + 1013904223'u32
    seed shr 8
  # Names of growing length push successive NIMQ2 payloads to unaligned
  # byte offsets and off any page boundary.
  for (name, K, N) in [("a.q_proj", 512, 96), ("bb.k_proj.weight", 1024, 64), ("ccc.o", 256, 520),
                       ("d", 512, 33), ("ee", 512, 33), ("fff", 512, 33), ("gggg", 512, 33)]:
    var qt = QuantTensor(kind: qkInt4Asymmetric, shape: @[N, K], groupSize: group)
    qt.data = newSeq[uint8](N * ((K + 1) div 2))
    for b in qt.data.mitems: b = uint8(rnd() and 255)
    let groups = (K + group - 1) div group
    for i in 0 ..< N * groups:
      qt.scale.add 0.002'f32 + float32(rnd() mod 100) * 0.0001'f32
      qt.zero_point.add float32(rnd() mod 16)
    sd.add((name, qt))
  saveQuantStateDict(dir / "v2.nimq", [1, 2, 3, 4, 5], sd)
  saveQuantStateDict(dir / "v3.nimq", [1, 2, 3, 4, 5], sd, NimqPayloadAlign)
  var zeroCopyByFile: seq[bool]
  for file in ["v2.nimq", "v3.nimq"]:
    let loaded = loadQuantStateDict(dir / file, mapped = true)
    doAssert loaded.payloadAlign == (if file == "v3.nimq": NimqPayloadAlign else: 0)
    var shifts = initHashSet[int]()
    var zeroCopy = true
    for (name, qt) in loaded.sd:
      doAssert qt.data.len == 0 and qt.mappedData.owner != nil
      let N = qt.shape[0]
      let K = qt.shape[1]
      let groups = (K + group - 1) div group
      let up = residentUploadQ4Mapped(be, qt.mappedData.dataPtr, qt.mappedData.len,
        qt.scale, qt.zero_point, K, N, group, groups)
      var w = up.w
      zeroCopy = zeroCopy and up.zeroCopy
      shifts.incl w.wqOffset mod 4
      let wDense = dequantizeTensor(qt)       # [N, K]
      for M in [1, 5]: checkQ4(be, w, wDense, K, N, M, &"{file} {name}")
      # Tuned plans (wide kernels with BYBY_Q4_TUNED=wide) on the same buffer.
      let tuned = residentPrepareQ4(w, 1) and residentPrepareQ4(w, 8)
      for M in [1, 5]: checkQ4(be, w, wDense, K, N, M, &"{file} {name} tuned")
      residentFreeQ4(w)
      echo &"PASS {file} {name} {N}x{K}: zeroCopy={up.zeroCopy} wqOffset={w.wqOffset} tuned={tuned}"
    if file == "v2.nimq" and zeroCopy:
      # Shift 0 is covered by the page-aligned v3 file.
      doAssert toHashSet([1, 2, 3]) <= shifts, "v2 payloads should cover every unaligned shift: " & $shifts
    zeroCopyByFile.add zeroCopy
  doAssert zeroCopyByFile[0] == zeroCopyByFile[1], "NIMQ2 must wrap in place wherever NIMQ3 does"

proc fp32Lmhead() =
  ## Raw fp32 [N, K] in a model mmap used as the NT GEMM weight (the lm_head
  ## path): wrapped in place when the payload is float aligned, else copied.
  let ctx = newBackend(getEnv("BYBY_INFER_BACKEND", "metal"))
  let be = ctx.kind.toByby()
  let dir = getTempDir() / ("mmap-fp32-" & $getCurrentProcessId())
  createDir(dir)
  defer: removeDir(dir)
  const K = 384
  const N = 1000
  var w = newSeq[float32](N * K)
  for i in 0 ..< w.len: w[i] = cos(float32(i) * 0.011'f32) * 0.05'f32
  var wrappedAny = false
  # NIMQ2 with the payload float aligned and not, then NIMQ3.
  for (file, align) in [("lm_v2a.nimq", 0), ("lm_v2b.nimq", 0), ("lm_v3.nimq", NimqPayloadAlign)]:
    var name = "odd.name"
    var loaded: tuple[arch: array[5, int], sd: seq[(string, QuantTensor)], payloadAlign: int]
    for extra in 0 ..< 4:
      saveQuantStateDict(dir / file, [1, 2, 3, 4, 5],
        @[(name, quantizeRaw(@[1'f32], @[1])), ("lm_head.weight", quantizeRaw(w, @[N, K]))], align)
      loaded = loadQuantStateDict(dir / file, mapped = true)
      let aligned = loaded.sd[1][1].mappedData.offset mod 4 == 0
      if align > 0 or aligned == (file == "lm_v2a.nimq"): break
      name.add "x"
    let qt = loaded.sd[1][1]
    let floatAligned = qt.mappedData.offset mod 4 == 0
    doAssert floatAligned == (file != "lm_v2b.nimq")
    let view = rawFloatView(qt)
    doAssert view.owner != nil and view.values[5] == w[5]
    let wrapped = beResidentWrapFp32(ctx, qt.mappedData.dataPtr, qt.mappedData.len)
    var hw = wrapped.h
    if not floatAligned: doAssert not wrapped.zeroCopy, file & ": unaligned floats are copied"
    if align > 0: doAssert wrapped.zeroCopy == wrappedAny, "NIMQ2 must wrap in place wherever NIMQ3 does"
    wrappedAny = wrappedAny or wrapped.zeroCopy
    for M in [1, 3]:
      var x = newSeq[float32](M * K)
      for i in 0 ..< x.len: x[i] = sin(float32(i) * 0.21'f32)
      var hx = byby.sessionUpload(be, x)
      var hy = byby.sessionAllocScratch(be, M * N)
      doAssert byby.sessionBegin(be)
      doAssert byby.sessionMatmulNT(be, hx, hw, hy, M, K, N, wrapped.offsetElems)
      doAssert byby.sessionEnd(be), byby.sessionGetLastError(be)
      let got = byby.sessionRead(be, hy, M * N)
      byby.sessionFree(be, hx); byby.sessionFree(be, hy)
      var err = 0'f64
      for m in 0 ..< M:
        for n in 0 ..< N:
          var s = 0'f64
          for k in 0 ..< K: s += float64(x[m * K + k]) * float64(w[n * K + k])
          err = max(err, abs(s - float64(got[m * N + n])))
      doAssert err < 1e-3, &"{file} fp32 NT M={M}: error={err}"
    byby.sessionFree(be, hw)
    echo &"PASS {file} fp32 lm_head {N}x{K} NT from mmap: zeroCopy={wrapped.zeroCopy} " &
      &"offset={wrapped.offsetElems} floats"

q4FromMmap()
fp32Lmhead()
