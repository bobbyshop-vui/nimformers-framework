## Bounded FP4-KM merge. One packed tensor candidate, no dense matrix
## or duplicate GPU model. The original checkpoint mapping is never written.
import std/[math, os, tables, strformat, strutils, algorithm, monotimes, times]
import backend, quant, lora, gpu_layers, fp4_gemv_gpu, nimformer, jamba_hybrid, customfloat
import merge_gpu

when defined(macosx):
  {.compile: "inplace_weight_io.m".}
  proc writeShared(handle, data: pointer; offset, count: csize_t): cint
    {.cdecl, importc: "codestral_write_shared_bytes".}

type
  MergeCandidate* = object
    packed*: seq[uint8]
    globalScale*: float32
    deltaRelativeError*, weightRelativeError*: float64
    accepted*: bool
  MergeUsage {.importc: "struct rusage", header: "<sys/resource.h>", bycopy.} = object
    ru_maxrss {.importc: "ru_maxrss".}: clong

proc getUsage(who: cint; usage: ptr MergeUsage): cint
  {.importc: "getrusage", header: "<sys/resource.h>".}

proc peakResidentBytes*(): int64 =
  var usage: MergeUsage
  if getUsage(0, addr usage) != 0:
    raise newException(OSError, "cannot read process memory usage for merge budget")
  when defined(macosx): int64(usage.ru_maxrss)
  else: int64(usage.ru_maxrss)*1024

proc checkMergeMemory*(scratchBytes: int64; limitBytes = 7_000_000_000'i64) =
  # Reserve driver/allocator overhead as well as all planned merge buffers.
  # `peakResidentBytes` is the historical PEAK (ru_maxrss), so a machine under
  # memory pressure fails here even when the live footprint is far smaller.
  # Report the number so the failure is diagnosable, and let
  # BYBY_MEMORY_LIMIT_MB raise the cap on a machine with headroom.
  if scratchBytes < 0 or limitBytes <= 0 or limitBytes > 12_000_000_000'i64 or
      peakResidentBytes() > limitBytes - scratchBytes - 256*1024*1024:
    raise newException(ValueError, "in-place merge cannot fit the " &
      $(limitBytes div 1_000_000_000) & " GB process budget (peak RSS=" &
      $(peakResidentBytes() div (1024*1024)) & " MiB, scratch=" &
      $(scratchBytes div (1024*1024)) & " MiB)")

proc magIndex(v: float32): uint8 {.inline.} =
  if v<0.25:0 elif v<0.75:1 elif v<1.25:2 elif v<1.75:3
  elif v<2.5:4 elif v<3.5:5 elif v<5:6 else:7

proc mergeLegacyFp4Candidate*(base: QuantTensor; ad: LoraAdapter;
    scratchBytes = 64*1024*1024): MergeCandidate =
  ## Same arithmetic/codec as BYBY_MERGE_LORA=1 in commit 24111a9.
  ## Recompute one group per pass instead of storing a dense matrix.
  if base.kind!=qkFp4KM or base.shape.len!=2 or base.groupSize notin [0,32]:
    raise newException(ValueError,"legacy in-place merge requires FP4-KM group32")
  let rows=base.shape[0];let cols=base.shape[1]
  let elements=rows*cols;let groups=elements div 32
  let packedBytes=elements div 2+groups
  if cols mod 32!=0 or ad.inF!=cols or ad.outF!=rows or ad.r<=0 or
      ad.aValues.len!=ad.r*cols or ad.bValues.len!=rows*ad.r or base.bytes.len!=packedBytes:
    raise newException(ValueError,"legacy merge shape mismatch")
  if scratchBytes<4*1024*1024 or packedBytes>scratchBytes-4*1024*1024:
    raise newException(ValueError,"packed tensor exceeds bounded merge scratch")
  let lut=buildFp4KmLutQ(base)
  let e4m3=buildDecodeLUT(FP8_E4M3)
  var values,original:array[32,float32]
  proc loadGroup(group:int; values:var array[32,float32]) =
    let row=group div (cols div 32)
    let first=(group mod (cols div 32))*32
    let scale=lut[16+int(base.bytes[elements div 2+group])]
    for lane in 0..<16:
      let byte=base.bytes[group*16+lane]
      values[lane]=lut[int(byte and 15)]*scale
      values[lane+16]=lut[int(byte shr 4)]*scale
    original=values
    # Preserve rank order and FP32 rounding of the original in-place += loop.
    for rank in 0..<ad.r:
      let b=ad.bValues[row*ad.r+rank]*ad.scale
      if b==0:continue
      for lane in 0..<32:
        values[lane]+=b*ad.aValues[rank*cols+first+lane]
    for value in values:
      if classify(value) in {fcNan,fcInf,fcNegInf}:
        raise newException(ValueError,"nonfinite legacy merge target")
  var maxScale=0'f32
  for group in 0..<groups:
    loadGroup(group,values)
    var mx=0'f32
    for value in values:mx=max(mx,abs(value))
    maxScale=max(maxScale,mx/6'f32)
  result.globalScale=if maxScale>0:maxScale/224'f32 else:1'f32
  result.packed=newSeq[uint8](packedBytes)
  var errSq,deltaSq,weightSq:float64
  for group in 0..<groups:
    loadGroup(group,values)
    var mx=0'f32
    for lane,value in values:
      mx=max(mx,abs(value))
      let d=float64(value)-float64(original[lane])
      deltaSq+=d*d;weightSq+=float64(value)*float64(value)
    var bestError=Inf.float32
    var bestByte=0'u8
    var bestScale=0'f32
    if mx>0:
      for step in 0..<8:
        let ratio=0.62'f32+0.62'f32*float32(step)/7'f32
        let candidate=(mx/6'f32)*ratio
        # encodeArray([x], FP8_E4M3) uses this exact exported scalar codec.
        let byte=uint8(fp32ToPackedU64(candidate/result.globalScale,FP8_E4M3))
        let decoded=e4m3[int(byte)]
        if decoded<=0 or decoded>1e30'f32 or decoded!=decoded:continue
        let scale=decoded*result.globalScale
        if scale<=0:continue
        var error=0'f32
        let invScale=1'f32/scale
        for value in values:
          let normalized=value*invScale
          let q=Fp4kmMags[int(magIndex(abs(normalized)))]*scale
          let d=(if normalized<0: -q else:q)-value
          error+=d*d
        if error<bestError:bestError=error;bestByte=byte;bestScale=scale
    result.packed[elements div 2+group]=bestByte
    if bestScale>0:
      let invScale=1'f32/bestScale
      for lane in 0..<16:
        let low=values[lane]*invScale
        let high=values[lane+16]*invScale
        let lo=magIndex(abs(low)) or (if low<0:8'u8 else:0'u8)
        let hi=magIndex(abs(high)) or (if high<0:8'u8 else:0'u8)
        result.packed[group*16+lane]=lo or (hi shl 4)
    if mx>0:errSq+=float64(bestError)
  result.deltaRelativeError=if deltaSq>0:sqrt(errSq/deltaSq) elif errSq==0:0.0 else:Inf
  result.weightRelativeError=if weightSq>0:sqrt(errSq/weightSq) elif errSq==0:0.0 else:Inf
  result.accepted=true

proc mergeFp4Candidate*(base: QuantTensor; ad: LoraAdapter;
    scratchBytes = 64*1024*1024; maxDeltaError = 0.001;
    maxWeightError = 0.02): MergeCandidate =
  ## Keep per-tensor scale/layout unchanged, search the original and nearby
  ## representable group scales. Guard against losing small LoRA deltas.
  if base.kind != qkFp4KM or base.shape.len != 2 or
      base.groupSize notin [0,32] or base.shape[1] mod 32 != 0:
    raise newException(ValueError, "in-place merge requires FP4-KM group32")
  let rows=base.shape[0]; let cols=base.shape[1]
  let elements=rows*cols
  let packedBytes=elements div 2+elements div 32
  if ad.r<=0 or ad.inF!=cols or ad.outF!=rows or
      ad.aValues.len!=ad.r*cols or ad.bValues.len!=rows*ad.r or
      base.bytes.len!=packedBytes or maxDeltaError<0 or maxWeightError<0:
    raise newException(ValueError, "invalid in-place merge shape or error bound")
  if scratchBytes<4*1024*1024 or packedBytes>scratchBytes-4*1024*1024:
    raise newException(ValueError, "packed tensor exceeds bounded merge scratch")
  result.packed=newSeq[uint8](packedBytes)
  result.globalScale=if base.scale.len>0:base.scale[0] else:1'f32
  let lut=buildFp4KmLutQ(base)
  var target:array[32,float32]
  var exactTarget:array[32,float64]
  var codes,bestCodes:array[32,uint8]
  var deltaSq,weightSq,errSq:float64
  for group in 0..<elements div 32:
    let row=group div (cols div 32)
    let col=(group mod (cols div 32))*32
    let oldScale=int(base.bytes[elements div 2+group])
    var groupDeltaSq=0.0
    for lane in 0..<32:
      let packed=base.bytes[group*16+(lane mod 16)]
      let code=if lane<16:int(packed and 15) else:int(packed shr 4)
      let old=lut[code]*lut[16+oldScale]
      var d=0'f32
      for rank in 0..<ad.r:
        d+=ad.scale*ad.bValues[row*ad.r+rank]*ad.aValues[rank*cols+col+lane]
      target[lane]=old+d
      exactTarget[lane]=float64(old)+float64(d)
      if classify(target[lane]) in {fcNan,fcInf,fcNegInf}:
        raise newException(ValueError, "nonfinite LoRA merge target")
      deltaSq+=float64(d)*float64(d)
      groupDeltaSq+=float64(d)*float64(d)
      weightSq+=float64(target[lane])*float64(target[lane])
    var bestErr=Inf
    var bestScale=oldScale
    for scaleByte in max(0,oldScale-2)..min(255,oldScale+2):
      let scale=lut[16+scaleByte]
      if scale<=0 or classify(scale) in {fcNan,fcInf,fcNegInf}:continue
      var error=0.0
      for lane in 0..<32:
        codes[lane]=magIndex(abs(target[lane]/scale)) or
          (if target[lane]<0:8'u8 else:0'u8)
        let got=lut[int(codes[lane])]*scale
        let e=float64(got)-exactTarget[lane]
        error+=e*e
      if error<bestErr:
        bestErr=error;bestScale=scaleByte;bestCodes=codes
    if bestErr==Inf:raise newException(ValueError,"no finite FP4 merge scale")
    # Require every quant group to preserve its own delta. This also rejects
    # an unsafe tensor early, before scanning billions of unaffected weights.
    if bestErr>maxDeltaError*maxDeltaError*groupDeltaSq:
      result.deltaRelativeError=if groupDeltaSq>0:sqrt(bestErr/groupDeltaSq) else:Inf
      result.packed.setLen(0)
      return
    errSq+=bestErr
    for lane in 0..<16:
      result.packed[group*16+lane]=bestCodes[lane] or (bestCodes[lane+16] shl 4)
    result.packed[elements div 2+group]=uint8(bestScale)
  result.deltaRelativeError=if deltaSq>0:sqrt(errSq/deltaSq) elif errSq==0:0.0 else:Inf
  result.weightRelativeError=if weightSq>0:sqrt(errSq/weightSq) elif errSq==0:0.0 else:Inf
  result.accepted=deltaSq>0 and result.deltaRelativeError<=maxDeltaError and
    result.weightRelativeError<=maxWeightError
  if not result.accepted:result.packed.setLen(0)

proc overwritePackedWeight*(ctx: Backend; weight: var SessionHandle;
    packed: openArray[uint8]) =
  ## Preserve the buffer identity held by every cached GEMV/GEMM plan.
  let be=ctx.kind.toByby()
  when defined(macosx):
    if ctx.kind==bkMetal:
      if packed.len==0 or writeShared(weight.metalHandle,unsafeAddr packed[0],0,csize_t(packed.len))==0:
        raise newException(IOError,"in-place merge write to idle Metal buffer failed")
      return
  if ctx.kind!=bkVulkan or packed.len mod 4!=0:
    raise newException(ValueError,"in-place Vulkan merge needs aligned packed bytes")
  for offset in countup(0,packed.len-1,1024*1024):
    let count=min(1024*1024,packed.len-offset)
    var staging=sessionUploadBytesShared(be,packed.toOpenArray(offset,offset+count-1))
    try:
      if not sessionBegin(be):raise newException(IOError,"merge transfer begin failed")
      let copied=sessionCopyInto(be,staging,weight,0,offset div 4,count div 4)
      let finished=sessionEnd(be)
      if not copied or not finished:raise newException(IOError,"in-place merge transfer failed")
    finally:sessionFree(be,staging)

proc mergeResidentLoraInplace*(m: var NimformerModelJamba; lo: var JambaLora; ctx: Backend): int =
  let scratchMiB=min(64,parseInt(getEnv("BYBY_MERGE_SCRATCH_MIB","64")))
  let scratchBytes=scratchMiB*1024*1024
  let limitBytes=min(12_000_000_000'i64,parseBiggestInt(getEnv("BYBY_MEMORY_LIMIT_MB","7000"))*1_000_000)
  checkMergeMemory(scratchBytes.int64,limitBytes)
  when not defined(noOneDispatch):
    if gOdActive:raise newException(ValueError,"in-place merge requires BYBY_ONE_DISPATCH=0")
  var names:seq[string]
  for name in lo.adapters.keys:names.add name
  names.sort()
  for name in names:
    if name notin lo.quantBase or name notin gFp4CacheGpu:continue
    let base=lo.quantBase[name]
    if base.kind!=qkFp4KM:continue
    checkMergeMemory(scratchBytes.int64,limitBytes)
    echo &"[lora-merge] begin {name}; mode=GPU legacy/in-place; scratch<=8 MiB"
    let mergeStart=getMonoTime()
    let globalScale=mergeFp4ResidentGpu(ctx,gFp4CacheGpu[name],lo.adapters[name],scratchBytes)
    lo.quantBase[name].scale= @[globalScale]
    if name in gLoraGpu:
      var ad=gLoraGpu[name]
      sessionFree(ctx.kind.toByby(),ad.aBuf)
      sessionFree(ctx.kind.toByby(),ad.bBuf)
      sessionFree(ctx.kind.toByby(),ad.hT)
      gLoraGpu.del(name)
    lo.adapters.del(name)
    for linearName,lin in paramLinears(m):
      if linearName==name:
        lin[].weightQ.scale= @[globalScale]
        lin[].loraR=0;lin[].loraA= @[];lin[].loraB= @[]
        lin[].mappedLoraA=default(MappedView[float32])
        lin[].mappedLoraB=default(MappedView[float32])
    inc result
    let seconds=float64((getMonoTime()-mergeStart).inNanoseconds)/1e9
    echo &"[lora-merge] GPU in-place {name}: {seconds:.3f}s; buffer and LUT reused"
  echo &"[lora-merge] {result}/{names.len} merged; peak RSS={peakResidentBytes() div 1_000_000} MB; limit=7000 MB"
