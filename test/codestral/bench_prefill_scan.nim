## Actual Codestral scan shapes; compares identical seeded inputs on one GPU.
import std/[os, math, times, monotimes, algorithm, strformat, strutils]
import backend, mamba2_full_vm, prefill_inference_kernels, mamba2_ssm_gpu

let ctx=newBackend(getEnv("BYBY_INFER_BACKEND","metal"))
let be=ctx.kind.toByby()
let T=if paramCount()>0:parseInt(paramStr(1)) else:64
let L=prefixLayout(1,T,128,64,128,8,4,8192,1e-5'f32,8,true)
var seed=newSeq[float32](L.fwdEnd)
for i in 0..<T*L.cDim:seed[L.conv+i]=sin(float32(i mod 1009)*0.13'f32)*0.2'f32
for i in 0..<T*L.H:seed[L.dt+i]=0.07'f32
for i in 0..<L.H:seed[L.aLog+i]= -1;seed[L.dSkip+i]=0.5
for i in 0..<L.HD*L.N:seed[L.states+i]=0.01'f32
var initial=sessionUpload(be,seed)
var work=sessionAllocScratch(be,seed.len)
var reference:seq[float32]
for mode in ["serial","coalesced"]:
  var steps:seq[MvmStep]
  if mode=="coalesced":steps.add((buildPrefixStateTransposeProgram(L,true),L.HD*L.N,4,256,"@scan_in" & $T))
  steps.add((buildCompactPrefixScanProgram(L,mode=="coalesced"),L.HD,4,256,"@scan_" & mode & $T))
  if mode=="coalesced":steps.add((buildPrefixStateTransposeProgram(L,false),L.HD*L.N,4,256,"@scan_out" & $T))
  var names:seq[string]
  for s in steps:names.add kernelNameFor(ctx,s.prog,s.tag)
  var samples:seq[float64]
  for repeat in 0..<7:
    doAssert sessionBegin(be)
    doAssert sessionCopyInto(be,initial,work,0,0,seed.len)
    doAssert sessionEnd(be)
    let start=getMonoTime()
    doAssert sessionBegin(be)
    for i,s in steps:encodeMicroVm(ctx,s.prog,work,s.nThreads,s.tgBytes,s.tag,s.tgSize,names[i])
    doAssert sessionEnd(be)
    if repeat>=2:samples.add float64((getMonoTime()-start).inNanoseconds)/1e6
  let result=sessionRead(be,work,L.fwdEnd)
  var output:seq[float32]
  for i in 0..<T*L.C:output.add result[L.yPre+i]
  let off=L.states+(if mode=="serial":L.HD*L.N else:0)
  for i in 0..<L.HD*L.N:output.add result[off+i]
  if mode=="serial":reference=output
  else:
    var err=0'f32
    for i in 0..<output.len:err=max(err,abs(output[i]-reference[i]))
    doAssert err<2e-5'f32, $err
  samples.sort()
  echo &"SCAN {ctx.kind} T={T} {mode} median_ms={samples[2]:.3f}"
sessionFree(be,initial);sessionFree(be,work)
