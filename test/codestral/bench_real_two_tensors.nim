## Benchmark GPU merge on two real Codestral tensors (inProj and outProj)
import std/[tables, strformat, strutils, monotimes, times, os]
import backend, quant, lora, fp4_gemv_gpu, nimformer
import merge_gpu, rawbin, quantize_lora_bin_4bit
import vendor/bybylang/gpubackend as byby

const NimqPath = "test/codestral/models/codestral_mamba_only.nimq"
const CkptPath = "test/codestral/models/jamba_parallel_lora.bin"

proc run() =
  let backendName = getEnv("BYBY_INFER_BACKEND", "metal")
  echo &"=== Benchmarking GPU Merge on Two Real Codestral Tensors ({backendName}) ==="
  let ctx = newBackend(backendName)
  let be = ctx.kind.toByby()

  # 1. Load two quant tensors from .nimq
  echo &"[bench] Mapping {NimqPath} ..."
  let t0 = getMonoTime()
  let sd = quant.loadQuantStateDict(NimqPath, mapped = true).sd
  var quantMap = initTable[string, QuantTensor]()
  for (name, qt) in sd:
    if name in ["backbone.layers.0.mixer.in_proj.weight", "backbone.layers.0.mixer.out_proj.weight",
                "block0.mambaReal.inProj.weight", "block0.mambaReal.outProj.weight"]:
      quantMap[name] = qt
  echo &"[bench] Mapped .nimq in {(getMonoTime()-t0).inMilliseconds} ms"

  var inProjBase, outProjBase: QuantTensor
  if "block0.mambaReal.inProj.weight" in quantMap:
    inProjBase = quantMap["block0.mambaReal.inProj.weight"]
  elif "backbone.layers.0.mixer.in_proj.weight" in quantMap:
    inProjBase = quantMap["backbone.layers.0.mixer.in_proj.weight"]

  if "block0.mambaReal.outProj.weight" in quantMap:
    outProjBase = quantMap["block0.mambaReal.outProj.weight"]
  elif "backbone.layers.0.mixer.out_proj.weight" in quantMap:
    outProjBase = quantMap["backbone.layers.0.mixer.out_proj.weight"]

  echo &"[bench] inProj shape: {inProjBase.shape}, bytes: {inProjBase.bytes.len div (1024*1024)} MiB"
  echo &"[bench] outProj shape: {outProjBase.shape}, bytes: {outProjBase.bytes.len div (1024*1024)} MiB"

  # 2. Load adapters from .bin
  echo &"[bench] Mapping {CkptPath} with loadMappedQLora4Bin ..."
  let t1 = getMonoTime()
  let q = loadMappedQLora4Bin(CkptPath)
  echo &"[bench] Mapped .bin in {(getMonoTime()-t1).inMilliseconds} ms"

  proc getAdapter(adName: string; rows, cols: int): LoraAdapter =
    let r = parseInt(q.meta["adapter." & adName & ".r"])
    let alpha = parseFloat(q.meta["adapter." & adName & ".alpha"]).float32
    result.inF = cols
    result.outF = rows
    result.r = r
    result.alpha = alpha
    result.scale = alpha / float32(r)
    for t in q.fp32Tensors:
      if t.name == "adapter::" & adName & "::A":
        result.A = Tensor(shape: @[r, cols], data: t.data, mappedData: t.mappedData)
        result.mappedA = t.mappedData
      elif t.name == "adapter::" & adName & "::B":
        result.B = Tensor(shape: @[rows, r], data: t.data, mappedData: t.mappedData)
        result.mappedB = t.mappedData

  let inProjAdapter = getAdapter("block0.mambaReal.inProj.weight", inProjBase.shape[0], inProjBase.shape[1])
  let outProjAdapter = getAdapter("block0.mambaReal.outProj.weight", outProjBase.shape[0], outProjBase.shape[1])
  echo &"[bench] inProj adapter r={inProjAdapter.r}, scale={inProjAdapter.scale}, A={inProjAdapter.aValues.len}, B={inProjAdapter.bValues.len}"
  echo &"[bench] outProj adapter r={outProjAdapter.r}, scale={outProjAdapter.scale}, A={outProjAdapter.aValues.len}, B={outProjAdapter.bValues.len}"

  let scratchBytes = 64 * 1024 * 1024

  for (name, base, ad) in [
    ("block0.mambaReal.outProj.weight", outProjBase, outProjAdapter),
    ("block0.mambaReal.inProj.weight", inProjBase, inProjAdapter)
  ]:
    echo &"\n[bench] --- Benchmarking {name} ({base.shape[0]}x{base.shape[1]}) ---"
    let rows = base.shape[0]
    let cols = base.shape[1]
    var cache = Fp4LinCacheGpu(
      rows: rows,
      cols: cols,
      wBuf: sessionUploadBytesShared(be, base.bytes),
      lutBuf: byby.sessionUploadPrivate(be, buildFp4KmLutQ(base))
    )

    let mStart = getMonoTime()
    let scale = mergeFp4ResidentGpu(ctx, cache, ad, scratchBytes)
    let mElapsed = float64((getMonoTime() - mStart).inNanoseconds) / 1e9
    echo &"[bench] GPU in-place merge {name}: {mElapsed:.3f} s (globalScale={scale})"

    sessionFree(be, cache.wBuf)
    sessionFree(be, cache.lutBuf)

run()
