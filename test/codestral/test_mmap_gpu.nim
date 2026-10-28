## GPU regression: mapped FP4 + mapped LoRA + mapped dense sidecar.
## BYBY_BENCH_BACKEND=vulkan runs the same checks on a T4/Vulkan host.
import std/[os, tables, math]
import backend, nimformer, quant, rawbin
import gpu_layers, fp4_gemv_gpu
import vendor/bybylang/gpubackend as byby

proc main() =
  const K = 8192
  const N = 16
  let dir = getTempDir() / ("mmap-gpu-" & $getCurrentProcessId())
  createDir(dir)
  defer: removeDir(dir)
  var source = newSeq[float32](K * N)
  var x = newSeq[float32](K)
  var a = newSeq[float32](K)
  var b = newSeq[float32](N)
  for i in 0 ..< source.len: source[i] = sin(float32(i) * 0.017'f32) * 0.03'f32
  for i in 0 ..< K:
    x[i] = cos(float32(i) * 0.013'f32)
    a[i] = sin(float32(i) * 0.019'f32) * 0.01'f32
  for i in 0 ..< N: b[i] = float32(i + 1) * 0.1'f32
  let qt = quantizeFp4KM(source, @[N, K])
  saveQuantStateDict(dir / "base.nimq", [N, K, 1, 1, 1], @[("proj", qt)])
  saveRawBin(dir / "adapter.bin", initTable[string, string](), @[
    RawBinTensor(name: "adapter::proj::A", shape: @[1, K], data: a),
    RawBinTensor(name: "adapter::proj::B", shape: @[N, 1], data: b),
    RawBinTensor(name: "newBlock::dense.weight", shape: @[N, K], data: source)])
  let base = loadQuantStateDict(dir / "base.nimq", mapped = true)
  let adapter = loadMappedRawBin(dir / "adapter.bin")
  let ctx = newBackend(getEnv("BYBY_BENCH_BACKEND", "metal"))
  let be = ctx.kind.toByby()
  var lin = Linear(inF: K, outF: N, useQuant: true, weightQ: base.sd[0][1],
    bias: newTensor(@[N]), loraR: 1, loraScale: 0.5'f32,
    mappedLoraA: adapter.tensors[0].mappedData, mappedLoraB: adapter.tensors[1].mappedData)
  discard prepareGpuLinForLinear(ctx, "mapped-fp4", lin)
  doAssert lin.weightQ.bytes.len == 0
  var dense = Linear(inF: K, outF: N, bias: newTensor(@[N]),
    weight: Tensor(shape: @[N, K], mappedData: adapter.tensors[2].mappedData))
  discard prepareGpuLinForLinear(ctx, "mapped-dense", dense)
  doAssert dense.weight.floatValues.len == 0
  var hx = byby.sessionUpload(be, x)
  var hy = byby.sessionAllocScratch(be, N)
  defer:
    byby.sessionFree(be, hx)
    byby.sessionFree(be, hy)
  let decoded = dequantizeTensor(qt)
  for name in ["mapped-fp4", "mapped-dense"]:
    doAssert byby.sessionBegin(be)
    doAssert encodeLinByName(name, hx, 0, hy, 0)
    doAssert byby.sessionEnd(be)
    let got = byby.sessionRead(be, hy, N)
    var xa = 0.0
    for i in 0 ..< K: xa += float64(x[i]) * float64(a[i])
    for n in 0 ..< N:
      var expected = if name == "mapped-fp4": xa * float64(b[n]) * 0.5 else: 0.0
      for i in 0 ..< K:
        expected += float64(x[i]) * float64(if name == "mapped-fp4": decoded[n * K + i] else: source[n * K + i])
      doAssert abs(float64(got[n]) - expected) < max(1e-4, abs(expected) * 1e-4)
    echo "PASS: ", name, " GPU output matches CPU reference"
  # Batch entry reaches cooperative matrix on supporting Vulkan drivers.
  var batchX = newSeq[float32](8 * K)
  for i in 0 ..< batchX.len: batchX[i] = x[i mod K]
  var hb = byby.sessionUpload(be, batchX)
  var ho = byby.sessionAllocScratch(be, 8 * N)
  defer:
    byby.sessionFree(be, hb)
    byby.sessionFree(be, ho)
  doAssert byby.sessionBegin(be)
  doAssert encodeFp4GemmRows(gFp4CacheGpu["mapped-fp4"], hb, ho, 8)
  doAssert byby.sessionEnd(be)
  let gotBatch = byby.sessionRead(be, ho, 8 * N)
  for row in 0 ..< 8:
    for n in 0 ..< N:
      var expected = 0.0
      for i in 0 ..< K: expected += float64(x[i]) * float64(decoded[n * K + i])
      doAssert abs(float64(gotBatch[row * N + n]) - expected) < max(1e-4, abs(expected) * 0.005)
  echo "PASS: mapped FP4 batch GPU output matches reference"
main()
