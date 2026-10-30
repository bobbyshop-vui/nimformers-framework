## GPU-only legacy FP4 merge, bounded scratch; CPU schedules dispatches only.
import std/[strutils, math, os]
import backend, lora, fp4_gemv_gpu, customfloat, session_vm_dispatch
import vendor/bybylang/gpubackend as byby

const stages=["max","scale","quant","commit","advance","lut"]
const common=staticRead("merge_gpu_common.h")

proc shader(stage:int; metal:bool):string =
  let sections=common.split("// ENTRY_")
  let name="codestral_merge_gpu_" & stages[stage]
  if metal:
    result="""
#include <metal_stdlib>
using namespace metal;
#pragma clang fp contract(off)
#define PRECISE
#define INL __attribute__((always_inline))
#define F2U(x) as_type<uint>(x)
#define U2F(x) as_type<float>(x)
#define FLOAT_INF INFINITY
#define BYTE_AT(i) ((weights[(i)>>2u]>>(((i)&3u)*8u))&255u)
#define PARAMS , device uint* scratch, device uint* weights, device float* lut, device const float* adapter, device uint* state
#define ARGS ,scratch,weights,lut,adapter,state
#define OUT_VALUES thread float* values
#define SYNC threadgroup_barrier(mem_flags::mem_threadgroup)
#define BAD_VALUE atomic_fetch_or_explicit((device atomic_uint*)&state[9],1u,memory_order_relaxed)
#define MAX_VALUE(v) atomic_fetch_max_explicit((device atomic_uint*)&state[0],F2U(v),memory_order_relaxed)
"""
    result.add sections[0]
    result.add "kernel void " & name & "(device uint* scratch [[buffer(0)]],device uint* weights [[buffer(1)]],device float* lut [[buffer(2)]],device const float* adapter [[buffer(3)]],device uint* state [[buffer(4)]],uint gid [[thread_position_in_grid]],uint lid [[thread_index_in_threadgroup]])\n"
    let body=sections[stage+1].split('\n',1)[1]
    result.add(if stage==0:"{\n threadgroup float maxmem[128];\n" & body[2..^1] else:body)
  else:
    result="""
#version 450
layout(local_size_x_id=0) in;
layout(std430,binding=0) buffer Scratch { uint scratch[]; };
layout(std430,binding=1) buffer Weights { uint weights[]; };
layout(std430,binding=2) buffer Lut { float lut[]; };
layout(std430,binding=3) readonly buffer Adapter { float adapter[]; };
layout(std430,binding=4) buffer State { uint state[]; };
shared float maxmem[128];
#define PRECISE precise
#define INL
#define F2U(x) floatBitsToUint(x)
#define U2F(x) uintBitsToFloat(x)
#define FLOAT_INF uintBitsToFloat(0x7f800000u)
#define isfinite(x) (!isnan(x)&&!isinf(x))
#define BYTE_AT(i) ((weights[(i)>>2u]>>(((i)&3u)*8u))&255u)
#define PARAMS
#define ARGS
#define OUT_VALUES precise out float values[32]
#define SYNC barrier()
#define BAD_VALUE atomicOr(state[9],1u)
#define MAX_VALUE(v) atomicMax(state[0],F2U(v))
"""
    result.add sections[0]
    result.add "void main() { uint gid=gl_GlobalInvocationID.x, lid=gl_LocalInvocationID.x;\n"
    result.add sections[stage+1].split('\n',1)[1] & "\n}"

when defined(macosx):
  {.compile:"merge_gpu_metal.m".}
  proc runMetal(sources:ptr cstring; scratch,weights,lut,adapter,state:pointer;
    groups,tile:cint):cint {.cdecl,importc:"codestral_merge_gpu_metal".}

## One bounded scratch buffer for every tensor: merges are sequential, so
## re-allocating up to 8 MiB per weight only added allocator traffic
## (128 tensors per run).
var gMergeScratch: SessionHandle
var gMergeScratchFloats = 0
var gMergeScratchBe = byby.gbAuto

proc releaseMergeScratch*() =
  if gMergeScratchFloats > 0:
    sessionFree(gMergeScratchBe, gMergeScratch)
    gMergeScratchFloats = 0
    gMergeScratchBe = byby.gbAuto

proc mergeScratch(be: byby.GpuBackend; floats: int): SessionHandle =
  let live = if be == byby.gbVulkan: gMergeScratch.vulkanHandle.buf != nil
             else: gMergeScratch.metalHandle != nil
  if gMergeScratchFloats >= floats and gMergeScratchBe == be and live:
    return gMergeScratch
  releaseMergeScratch()
  gMergeScratch = sessionAllocScratch(be, floats)
  gMergeScratchFloats = floats
  gMergeScratchBe = be
  result = gMergeScratch

proc mergeFp4ResidentGpu*(ctx:Backend; cache:var Fp4LinCacheGpu;
    ad:LoraAdapter; scratchBytes:int):float32 =
  let be=ctx.kind.toByby()
  let elements=cache.rows*cache.cols
  let groups=elements div 32
  if ctx.kind notin {bkMetal,bkVulkan} or cache.cols mod 32!=0 or groups mod 4!=0 or
      ad.r<=0 or ad.inF!=cache.cols or ad.outF!=cache.rows or
      ad.aValues.len!=ad.r*cache.cols or ad.bValues.len!=ad.r*cache.rows:
    raise newException(ValueError,"GPU merge requires aligned FP4-KM group32 and valid LoRA")
  let adapterBytes=(ad.r*(cache.cols+cache.rows)+256)*4
  let tileBytes=min(scratchBytes-2*adapterBytes-4096,8*1024*1024)
  if tileBytes<80:raise newException(ValueError,"GPU merge scratch cannot fit adapter and one tile")
  let tile=min(groups,((tileBytes div 20) div 4)*4)
  var data= @(ad.aValues)
  data.add(ad.bValues)
  data.add(buildDecodeLUT(FP8_E4M3))
  var config=[0'u32,0,0,uint32(elements),uint32(cache.cols),uint32(cache.rows),
    uint32(ad.r),cast[uint32](ad.scale),uint32(tile),0]
  var configFloats=newSeq[float32](config.len)
  copyMem(addr configFloats[0],addr config[0],config.len*4)
  var hScratch=mergeScratch(be,tile*5)
  var hAdapter=sessionUpload(be,data)
  var hState=sessionUpload(be,configFloats)
  defer:
    sessionFree(be,hAdapter);sessionFree(be,hState)
  if ctx.kind==bkMetal:
    when defined(macosx):
      var sources:array[6,string]
      var sourcePtrs:array[6,cstring]
      for i in 0..<6:sources[i]=shader(i,true);sourcePtrs[i]=sources[i].cstring
      if runMetal(addr sourcePtrs[0],hScratch.metalHandle,cache.wBuf.metalHandle,
          cache.lutBuf.metalHandle,hAdapter.metalHandle,hState.metalHandle,
          cint(groups),cint(tile))==0:
        raise newException(IOError,"Metal GPU merge failed; inference aborted")
  else:
    let handles=[hScratch,cache.wBuf,cache.lutBuf,hAdapter,hState]
    proc dispatch(stage,n:int) =
      let name="codestral_merge_gpu_" & stages[stage]
      if not sessionEncPipeline(true,name,shader(stage,false)):
        raise newException(IOError,"Vulkan GPU merge pipeline failed")
      for idx,h in handles:
        if not sessionBindBufferVk(idx,h.vulkanHandle):raise newException(IOError,"GPU merge bind failed")
      if not sessionEncDispatch(true,n,128):raise newException(IOError,"GPU merge dispatch failed")
    if not sessionBegin(be):raise newException(IOError,"GPU merge begin failed")
    dispatch(0,((groups+127) div 128)*128);dispatch(1,1)
    if not sessionEnd(be):raise newException(IOError,"GPU merge max failed")
    let status=sessionRead(be,hState,10)
    if cast[uint32](status[9])!=0:raise newException(ValueError,"nonfinite GPU merge target")
    for first in countup(0,groups-1,tile):
      if not sessionBegin(be):raise newException(IOError,"GPU merge tile begin failed")
      dispatch(2,min(tile,groups-first));dispatch(3,min(tile,groups-first)*4);dispatch(4,1)
      if not sessionEnd(be):raise newException(IOError,"GPU merge tile failed")
    if not sessionBegin(be):raise newException(IOError,"GPU merge LUT begin failed")
    dispatch(5,256)
    if not sessionEnd(be):raise newException(IOError,"GPU merge LUT failed")
  result=sessionRead(be,hState,2)[1]
  if result<=0 or classify(result) in {fcNan,fcInf,fcNegInf}:
    raise newException(IOError,"invalid GPU merge global scale")
