import std/[math, os, tables, strutils]
import backend, quant, lora, fp4_gemv_gpu, inplace_lora_merge
import nimformer, jamba_hybrid, mamba2_real, gpu_layers
import merge_gpu
import vendor/bybylang/gpubackend as byby

var base=QuantTensor(kind:qkFp4KM,shape: @[8,64],scale: @[1'f32],groupSize:32,
  data:newSeq[uint8](8*64 div 2+8*64 div 32))
let lut=buildFp4KmLutQ(base)
var ad=newLoraAdapter(64,8,1,1)
let delta=lut[17]*0.5'f32
doAssert delta>0
for x in ad.A.data.mitems:x=delta
for x in ad.B.data.mitems:x=1
let candidate=mergeFp4Candidate(base,ad)
doAssert candidate.accepted and candidate.deltaRelativeError<1e-7
var merged=base
merged.data=candidate.packed
for x in dequantizeFp4KM(merged):doAssert abs(x-delta)<1e-7
for x in base.bytes:doAssert x==0 # original payload never mutated
for x in ad.A.data.mitems:x=1e-10'f32
let rejected=mergeFp4Candidate(base,ad)
doAssert not rejected.accepted and rejected.packed.len==0
var refused=false
try:discard mergeFp4Candidate(base,ad,scratchBytes=1)
except ValueError:refused=true
doAssert refused

# Oracle is the original merge loop and the existing quantizeFp4KM codec.
for rank in [1,4,8]:
  var dense=newSeq[float32](8*64)
  for i in 0..<dense.len:dense[i]=sin(float32(i)*0.13'f32)*0.2'f32
  let oldBase=quantizeFp4KM(dense,@[8,64])
  var oldAdapter=newLoraAdapter(64,8,rank,2*rank.float32)
  for i in 0..<oldAdapter.A.data.len:oldAdapter.A.data[i]=cos(float32(i)*0.3'f32)*0.03'f32
  for i in 0..<oldAdapter.B.data.len:oldAdapter.B.data[i]=sin(float32(i)*0.2'f32)*0.02'f32
  dense=dequantizeTensor(oldBase)
  for row in 0..<8:
    for r in 0..<rank:
      let b=oldAdapter.bValues[row*rank+r]*oldAdapter.scale
      if b==0:continue
      for col in 0..<64:dense[row*64+col]+=b*oldAdapter.aValues[r*64+col]
  let expected=quantizeTensor(dense,@[8,64],oldBase.kind,customCf=oldBase.cf)
  let streamed=mergeLegacyFp4Candidate(oldBase,oldAdapter)
  doAssert streamed.accepted and streamed.globalScale==expected.scale[0]
  doAssert streamed.packed==expected.data,"legacy merge packed byte mismatch rank=" & $rank
echo "PASS original BYBY_MERGE_LORA=1 vs streamed merge: global scale and every packed byte match for ranks 1/4/8"
refused=false
try:checkMergeMemory(1,1)
except ValueError:refused=true
doAssert refused

let ctx=newBackend(getEnv("BYBY_INFER_BACKEND","metal"))
let be=ctx.kind.toByby()
for rank in [1,4,8]:
  var dense=newSeq[float32](8*64)
  for i in 0..<dense.len:dense[i]=sin(float32(i)*0.13'f32)*0.2'f32
  let qt=quantizeFp4KM(dense,@[8,64])
  var adapter=newLoraAdapter(64,8,rank,2*rank.float32)
  for i in 0..<adapter.A.data.len:adapter.A.data[i]=cos(float32(i)*0.3'f32)*0.03'f32
  for i in 0..<adapter.B.data.len:adapter.B.data[i]=sin(float32(i)*0.2'f32)*0.02'f32
  let expected=mergeLegacyFp4Candidate(qt,adapter)
  var cache=Fp4LinCacheGpu(rows:8,cols:64,wBuf:sessionUploadBytesShared(be,qt.bytes),
    lutBuf:byby.sessionUploadPrivate(be,buildFp4KmLutQ(qt)))
  # Force four-group tiles to exercise in-place overwrites across boundaries.
  let scale=mergeFp4ResidentGpu(ctx,cache,adapter,2*(rank*(64+8)+256)*4+4096+80)
  doAssert scale==expected.globalScale,"GPU merge scale mismatch rank=" & $rank &
    " got=" & $scale & "/" & $cast[uint32](scale) &
    " expected=" & $expected.globalScale & "/" & $cast[uint32](expected.globalScale)
  var readback=sessionAllocScratch(be,qt.bytes.len div 4)
  doAssert sessionBegin(be)
  doAssert sessionCopyInto(be,cache.wBuf,readback,0,0,qt.bytes.len div 4)
  doAssert sessionEnd(be)
  let packedFloats=sessionRead(be,readback,qt.bytes.len div 4)
  var packedBytes=newSeq[uint8](qt.bytes.len)
  copyMem(addr packedBytes[0],unsafeAddr packedFloats[0],packedBytes.len)
  doAssert packedBytes==expected.packed,"GPU merge packed mismatch rank=" & $rank
  var lutReadback=sessionAllocScratch(be,272)
  doAssert sessionBegin(be)
  doAssert sessionCopyInto(be,cache.lutBuf,lutReadback,0,0,272)
  doAssert sessionEnd(be)
  let gpuLut=sessionRead(be,lutReadback,272)
  var mergedQt=qt;mergedQt.scale= @[scale]
  doAssert gpuLut==buildFp4KmLutQ(mergedQt),"GPU merge LUT mismatch"
  sessionFree(be,lutReadback);sessionFree(be,readback)
  sessionFree(be,cache.wBuf);sessionFree(be,cache.lutBuf)
echo "PASS GPU merge packed bytes, global scale and LUT match CPU legacy at ranks 1/4/8"
var weight=sessionUploadBytesShared(be,base.bytes)
let originalMetal=weight.metalHandle
let originalVulkan=weight.vulkanHandle.buf
overwritePackedWeight(ctx,weight,candidate.packed)
doAssert weight.metalHandle==originalMetal and weight.vulkanHandle.buf==originalVulkan
let floats=sessionRead(be,weight,candidate.packed.len div 4)
var bytes=newSeq[uint8](candidate.packed.len)
copyMem(addr bytes[0],unsafeAddr floats[0],bytes.len)
doAssert bytes==candidate.packed
# Packed bytes can encode NaNs as floats. Writes must remain bit-exact.
for i in 0..<bytes.len:bytes[i]=uint8((i*251+127) mod 256)
overwritePackedWeight(ctx,weight,bytes)
let raw=sessionRead(be,weight,bytes.len div 4)
var roundtrip=newSeq[uint8](bytes.len)
copyMem(addr roundtrip[0],unsafeAddr raw[0],bytes.len)
doAssert bytes==roundtrip
sessionFree(be,weight)
var m=newNimformerModelJamba(128,64,8,2,"R",mambaRealDState=8,
  mambaRealNGroups=2,mambaRealExpand=2,initializeWeights=false)
for (name,t) in paramList(m):
  var count=1
  for d in t[].shape:count*=d
  t[].data=newSeq[float32](count)
  for i in 0..<count:
    t[].data[i]=if name.endsWith("gamma") or name.endsWith("normWeight") or name.endsWith("dSkip"):1'f32
      elif name.endsWith("aLog"): -1'f32
      elif name.endsWith("dtBias"): -2'f32
      else:sin(float32(i)*0.1'f32)*0.03'f32
var lo:JambaLora
const targetName="block0.mambaReal.outProj.weight"
for name,lin in paramLinears(m):
  if lin[].weight.shape.len!=2:continue
  var qt=quantizeFp4KM(lin[].weight.data,lin[].weight.shape)
  if name==targetName:
    qt.scale= @[1'f32]
    for x in qt.data.mitems:x=0
    var adapter=newLoraAdapter(lin[].inF,lin[].outF,1,1)
    for x in adapter.A.data.mitems:x=delta
    for x in adapter.B.data.mitems:x=1
    lo.quantBase[name]=qt;lo.adapters[name]=adapter
    lin[].loraR=1;lin[].loraScale=1
    lin[].loraA=adapter.A.data;lin[].loraB=adapter.B.data
  setQuantWeight(lin[],qt)
discard setGpuNormPreset("mamba-ssm")
prepareGpuModel(m,lo,ctx)
var expected:seq[float32]
for id in [3,10,17]:expected=forwardStepJambaGpu(m,id,ctx).data
let originalHandle=gFp4CacheGpu[targetName].wBuf
doAssert mergeResidentLoraInplace(m,lo,ctx)==1
doAssert lo.adapters.len==0 and targetName notin gLoraGpu
doAssert originalHandle.metalHandle==gFp4CacheGpu[targetName].wBuf.metalHandle
doAssert originalHandle.vulkanHandle.buf==gFp4CacheGpu[targetName].wBuf.vulkanHandle.buf
resetGpuGenState(m,ctx)
var actual:seq[float32]
for id in [3,10,17]:actual=forwardStepJambaGpu(m,id,ctx).data
for i in 0..<actual.len:doAssert abs(actual[i]-expected[i])<1e-5'f32
echo "PASS real model decode after accepted in-place merge: adapter removed once, GPU weight buffer reused, logits match separate LoRA"
echo "PASS guarded FP4 merge, unsafe-delta rejection, budget rejection, in-place GPU buffer and packed-bit preservation"
