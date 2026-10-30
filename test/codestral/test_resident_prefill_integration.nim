## Exercise the actual infer_lora entry point, nonzero LoRA, R/P/G state carry.
## Pass RRR or RPG as the first argument; each process owns one GPU model.
import std/[os, math, tables, strutils, strformat]
import backend, nimformer, quant, jamba_hybrid, lora, infer_lora
import mamba2_real, mamba2_fused_step, gpu_layers, prefill_gpu

setForbidCpuFallback(true)
let ctx=newBackend(getEnv("BYBY_INFER_BACKEND","metal"))
initMicroVmJit("resident-prefill-integration",ctx.kind.toByby())
let pattern=if paramCount()>0: paramStr(1) else: "RPG"
var m=newNimformerModelJamba(128,64,8,2,pattern,mambaRealDState=8,
  mambaRealNGroups=2,mambaRealExpand=2,initializeWeights=false)
for (name,t) in paramList(m):
  var count=1
  for d in t[].shape: count *= d
  t[].data=newSeq[float32](count)
  for i in 0..<count:
    t[].data[i]=if name.endsWith("gamma") or name.endsWith("normWeight") or name.endsWith("dSkip"): 1'f32
      elif name.endsWith("aLog"): -1'f32
      elif name.endsWith("dtBias"): -2'f32
      elif name.endsWith("bias") or name.endsWith("beta"): 0'f32
      else: sin(float32(i+name.len)*0.13'f32)*0.07'f32
for name,lin in paramLinears(m):
  if lin[].weight.shape.len != 2: continue
  let qt=quantizeFp4KM(lin[].weight.data,lin[].weight.shape)
  setQuantWeight(lin[],qt)
  if name.startsWith("block"):
    lin[].loraR=4; lin[].loraScale=0.5
    lin[].loraA=newSeq[float32](4*lin[].inF)
    lin[].loraB=newSeq[float32](4*lin[].outF)
    for i in 0..<lin[].loraA.len: lin[].loraA[i]=sin(float32(i)*0.2'f32)*0.01'f32
    for i in 0..<lin[].loraB.len: lin[].loraB[i]=cos(float32(i)*0.3'f32)*0.02'f32
    if ".attn." in name or ".fuseGate." in name:
      for i in 0..<lin[].bias.data.len: lin[].bias.data[i]=0.03'f32
var lo:JambaLora
discard setGpuNormPreset("mamba-ssm")
prepareGpuModel(m,lo,ctx)
putEnv("BYBY_PREFILL_BATCH_SIZE","8")

proc states():seq[float32]=
  for i,blk in m.blocks:
    let r=gLayersV2[i]
    let mr=blk.mambaReal
    let data=sessionRead(ctx.kind.toByby(),r.hWork,r.off.offSsm+mr.numHeads*mr.headDim*mr.dState)
    for j in 0..<mr.convDim*(mr.convKernel-1):result.add data[r.off.offConvSt+j]
    for j in 0..<mr.numHeads*mr.headDim*mr.dState:result.add data[r.off.offSsm+j]
    if blk.kind in {bkMambaRealPlusAttn,bkMambaRealGated}:
      let a=gAttnLayers[i]
      let n=a.lay.H*a.lay.D
      let d=sessionRead(ctx.kind.toByby(),a.hWork,max(a.lay.offZ+n,a.lay.offS+n*a.lay.D))
      for j in 0..<n: result.add d[a.lay.offZ+j]
      for j in 0..<n*a.lay.D: result.add d[a.lay.offS+j]

proc error(a,b:seq[float32]):float32=
  doAssert a.len==b.len
  var scale=1'f32
  for i in 0..<a.len:
    doAssert classify(b[i]) notin {fcNan,fcInf,fcNegInf}
    result=max(result,abs(a[i]-b[i])); scale=max(scale,abs(a[i]))
  result /= scale

for count in [2,10,17,65,129]:
  var ids:seq[int]
  for i in 0..<count:ids.add((i*7+3) mod m.vocab)
  warmupGpuForParallelPrefill(m,ctx)
  for id in ids:discard forwardStepJambaGpu(m,id,ctx,produceLogits=false)
  let expected=states()
  let expectedLogits=forwardStepJambaGpu(m,11,ctx).data
  for mode in ["auto","m8"]:
    putEnv("BYBY_PREFILL_GEMM",mode)
    for scan in ["serial","coalesced","parallel"]:
      putEnv("BYBY_PREFILL_SCAN",scan)
      resetGpuGenState(m,ctx)
      forwardPrefillBatch(m,lo,ids,ctx)
      let released=releaseResidentPrefill()
      doAssert released>0
      let stateError=error(expected,states())
      let logitError=error(expectedLogits,forwardStepJambaGpu(m,11,ctx).data)
      doAssert stateError<3e-4'f32 and logitError<3e-4'f32,
        &"pattern={pattern} T={count} {mode}/{scan} state={stateError} logits={logitError}"
      echo &"PASS infer_lora {pattern} + LoRA T={count} {mode}/{scan} state={stateError:.3g} logits={logitError:.3g}"
# Full-sized prefix blocks, including a terminal one-token remainder.
putEnv("BYBY_PREFILL_BATCH_SIZE","128")
putEnv("BYBY_PREFILL_SCAN","serial")
putEnv("BYBY_PREFILL_GEMM","auto")
var longIds: seq[int]
for i in 0..<257: longIds.add((i*7+3) mod m.vocab)
warmupGpuForParallelPrefill(m,ctx)
for id in longIds: discard forwardStepJambaGpu(m,id,ctx,produceLogits=false)
let longStates=states()
let longLogits=forwardStepJambaGpu(m,11,ctx).data
resetGpuGenState(m,ctx)
forwardPrefillBatch(m,lo,longIds,ctx)
doAssert releaseResidentPrefill()>0
doAssert error(longStates,states())<3e-4'f32
doAssert error(longLogits,forwardStepJambaGpu(m,11,ctx).data)<3e-4'f32
echo "PASS infer_lora 257 tokens in 128 + 128 + 1 with R/P/G and LoRA"
# Existing disable switch still selects the tokenwise fallback.
putEnv("BYBY_PREFILL_PARALLEL","0")
resetGpuGenState(m,ctx)
forwardPrefillBatch(m,lo,@[3,10,17],ctx)
echo "PASS existing prefill disable switch"
