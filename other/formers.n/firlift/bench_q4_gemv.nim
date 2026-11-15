## Resident Vulkan Q4 GEMV comparison: prior Q4 threadgroup kernel, 32-bit-word GEMV,
## and the native sessionMatmulQ4 path. All JIT compilation and uploads are outside timers.
## Run from the repository root with BACKEND=vulkan.
import std/[algorithm, math, times, monotimes]
import session_vm
import vendor/bybylang/gpubackend as byby
import vendor/bybylang/micro_kernel/session_vm_dispatch
import vendor/bybylang/micro_kernel/jit_vm
import ./kernel_gen
import ./kernel_asm

const GroupSize = 128
const Samples = 5
const DispatchesPerSample = 4

proc fill(n: int; seed: float; scale: float32): seq[float32] =
  result = newSeq[float32](n)
  for i in 0 ..< n: result[i] = float32(sin(float(i) * 0.37 + seed)) * scale

proc bytesAsFloats(b: seq[uint8]): seq[float32] =
  result = newSeq[float32]((b.len + 3) div 4)
  if b.len > 0: copyMem(addr result[0], unsafeAddr b[0], b.len)

proc maxRel(a, b: seq[float32]): float32 =
  if a.len != b.len: return Inf.float32
  for i in 0 ..< a.len:
    if a[i] != a[i] or b[i] != b[i] or a[i] == Inf.float32 or a[i] == -Inf.float32 or
        b[i] == Inf.float32 or b[i] == -Inf.float32:
      return Inf.float32
    result = max(result, abs(a[i] - b[i]) / max(1.0'f32, abs(b[i])))

proc median(xs: var seq[float64]): float64 =
  xs.sort()
  xs[xs.len div 2]

proc dispatchResident(be: byby.GpuBackend; name, source: string; g: Gen;
                      output, weights, activations: SessionHandle): bool =
  let vk = be == byby.gbVulkan
  if not sessionEncPipeline(vk, name, source): return false
  template bindHandle(slot: int; h: SessionHandle) =
    if not sessionEncBindVkOrMetal(vk, slot, h.metalHandle, h.vulkanHandle): return false
  bindHandle(0, output)
  bindHandle(1, weights)
  bindHandle(2, weights)
  bindHandle(3, activations)
  if not sessionEncTgBytes(vk, 0, g.tgBytes): return false
  sessionEncDispatch(vk, g.threads, g.tg, g.tgBytes)

proc compileResident(g: Gen; k, n: int; name: string): tuple[name, source: string] =
  let program = assemble(g.body, g.ins, g.outs, @[k, n * (k div 2) div 4 + 2 * n * (k div GroupSize)], @[n], 1e-5'f32,
    slotLayout = true)
  result = (name, lowerBytecodeToGlslCached(program, name))

proc benchCase(be: byby.GpuBackend; k, n: int) =
  let gs = GroupSize
  let groups = k div gs
  let rowBytes = k div 2
  var qbytes = newSeq[uint8](n * rowBytes)
  for i in 0 ..< qbytes.len: qbytes[i] = uint8((i * 37 + i div 11 + 13) mod 256)
  let scales = fill(n * groups, 0.3, 0.07'f32)
  var zeros = newSeq[float32](n * groups)
  for i in 0 ..< zeros.len: zeros[i] = 5.25'f32 + float32(i mod 7) * 0.37'f32
  let a = fill(k, 0.9, 0.65'f32)
  let packed = bytesAsFloats(qbytes) & scales & zeros

  let oldG = genMatmulQ4Par(1, k, n, gs, groups, 32, false, true)
  let wordG = genMatmulQ4WordPar(k, n, gs, groups, 32, true)
  let oldName = "q4old_k" & $k & "_n" & $n
  let wordName = "q4word_k" & $k & "_n" & $n
  let oldJob = compileResident(oldG, k, n, oldName)
  let wordJob = compileResident(wordG, k, n, wordName)

  doAssert sessionBegin(be), "sessionBegin failed"
  var weights = sessionUploadPrivate(be, packed)
  var activations = sessionUploadPrivate(be, a)
  var oldOut = sessionAllocScratch(be, n)
  var wordOut = sessionAllocScratch(be, n)
  var nativeOut: SessionHandle
  var nativeWeights = residentUploadQ4(be, qbytes, scales, zeros, k, n, gs, groups)
  sessionJitPrewarm(be, [oldJob, wordJob])

  doAssert dispatchResident(be, oldJob.name, oldJob.source, oldG, oldOut, weights, activations), "old Q4 dispatch failed"
  doAssert dispatchResident(be, wordJob.name, wordJob.source, wordG, wordOut, weights, activations), "word Q4 dispatch failed"
  doAssert sessionMatmulQ4(be, activations, nativeWeights, nativeOut, 1), "native Q4 dispatch failed"
  doAssert nativeOut.vulkanHandle.buf != nil, "native Q4 output was not allocated"
  doAssert sessionEnd(be), "warmup sessionEnd failed"
  let native = sessionRead(be, nativeOut, n)
  let oldValues = sessionRead(be, oldOut, n)
  let wordValues = sessionRead(be, wordOut, n)
  let oldErr = maxRel(oldValues, native)
  let wordErr = maxRel(wordValues, native)
  doAssert oldErr <= 5e-3'f32, "old Q4 numerical mismatch: " & $oldErr
  doAssert wordErr <= 5e-3'f32, "word Q4 numerical mismatch: " & $wordErr

  var oldTimes, wordTimes, nativeTimes: seq[float64]
  for sample in 0 ..< Samples:
    # Alternate candidate order to reduce drift from clock and thermal changes.
    let oldFirst = sample mod 2 == 0
    for candidate in 0 ..< 3:
      let which = if oldFirst: candidate else: (2 - candidate + 3) mod 3
      let start = getMonoTime()
      doAssert sessionBegin(be), "timed sessionBegin failed"
      for _ in 0 ..< DispatchesPerSample:
        case which
        of 0: doAssert dispatchResident(be, oldJob.name, oldJob.source, oldG, oldOut, weights, activations)
        of 1: doAssert dispatchResident(be, wordJob.name, wordJob.source, wordG, wordOut, weights, activations)
        else: doAssert sessionMatmulQ4(be, activations, nativeWeights, nativeOut, 1)
      doAssert sessionEnd(be), "timed sessionEnd failed"
      let elapsed = float64((getMonoTime() - start).inNanoseconds) / 1.0e6 / float64(DispatchesPerSample)
      case which
      of 0: oldTimes.add elapsed
      of 1: wordTimes.add elapsed
      else: nativeTimes.add elapsed

  echo "K=", k, " N=", n, " groups=", groups,
    " maxRel(old/native)=", oldErr, " maxRel(word/native)=", wordErr,
    " ms old=", median(oldTimes), " word=", median(wordTimes), " native=", median(nativeTimes),
    " speedup(word/old)=", median(oldTimes) / median(wordTimes)

  sessionFree(be, activations)
  sessionFree(be, weights)
  sessionFree(be, oldOut)
  sessionFree(be, wordOut)
  sessionFree(be, nativeOut)
  residentFreeQ4(nativeWeights)

proc main() =
  let be = byby.gbVulkan
  echo "Resident Q4 GEMV Vulkan benchmark; median of ", Samples,
    " samples x ", DispatchesPerSample, " dispatches, uploads/JIT excluded"
  for (k, n) in [(4096, 4096), (4096, 11008), (11008, 4096)]:
    benchCase(be, k, n)

when isMainModule:
  main()
