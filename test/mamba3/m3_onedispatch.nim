## Persistent Mamba-3 R/P/G stack. Reuses Codestral's virtual threadgroups,
## grid barrier, FP4/FP32 GEMV and LoRA bytecode, with Mamba-3 kernels.
import std/[tables, math, os, strutils]
import backend, nimformer, m3_hybrid, m3_inference_kernels
import gpu_layers, fp4_gemv_gpu, fp4_gemv_vm, session_vm, od_core
import vendor/bybylang/gpubackend as byby

type M3OneDispatch* = object
  ready*:bool
  kernel*:string
  weights,constants:SessionHandle
  ntg:int

proc release*(st:var M3OneDispatch;be:GpuBackend)=
  if st.ready:
    sessionFree(be,st.weights);sessionFree(be,st.constants)
  st=default(M3OneDispatch)

proc eligible*(m:var NimformerModelJamba;ctx:Backend):string =
  if ctx.kind!=bkMetal:return "persistent grid barriers require Metal"
  for name,lin in paramLinears(m):
    if lin[].inF<=0 or lin[].outF<=0 or name=="outProj.weight":continue
    if name notin gLinCache:return "missing GPU projection " & name
    let gpu=gLinCache[name]
    if gpu.isInt4:return "int4 sidecar uses the resident multi-dispatch path"
    if gpu.isFp4 and (gpu.K mod 32!=0 or gpu.N mod 2!=0):return "FP4 projection requires K%32=0 and even N"
    if name in gLoraGpu and gLoraGpu[name].r>32:return "LoRA rank>32 uses resident multi-dispatch"

proc prepareOneDispatch*(m:var NimformerModelJamba;ctx:Backend;a:M3Arena;
    states:seq[M3DeviceState];params:seq[M3Params];paramData:seq[float32]):M3OneDispatch =
  let reason=eligible(m,ctx)
  if reason.len>0:raise newException(ValueError,reason)
  let be=ctx.kind.toByby()
  var data=paramData
  var packedBytes=0
  var weightOffset,lutOffset,biasOffset,aOffset,bOffset:Table[string,int]
  var copies:seq[tuple[source:SessionHandle,src,dst,count:int,packed:bool]]
  template add(v:seq[float32]):int =
    block:
      let offset=data.len
      data.add(v)
      offset
  for name,lin in paramLinears(m):
    if lin[].inF<=0 or lin[].outF<=0 or name=="outProj.weight":continue
    let g=gLinCache[name]
    if g.isFp4:
      let cache=gFp4CacheGpu[name]
      let count=(g.K*g.N div 2+g.K*g.N div 32+3) div 4
      weightOffset[name]=packedBytes
      copies.add((cache.wBuf,0,packedBytes div 4,count,true))
      packedBytes+=count*4
      # Copy the GPU LUT too: in-place LoRA merge may have changed its scale.
      lutOffset[name]=data.len;data.setLen(data.len+272)
      copies.add((cache.lutBuf,0,lutOffset[name],272,false))
    else:
      lutOffset[name]=data.len;data.setLen(data.len+g.K*g.N)
      copies.add((g.wF32,0,lutOffset[name],g.K*g.N,false))
    biasOffset[name]=add(lin[].bias.data)
    if name in gLoraGpu:
      let ad=gLoraGpu[name]
      aOffset[name]=data.len;data.setLen(data.len+ad.r*g.K)
      copies.add((ad.aBuf,0,aOffset[name],ad.r*g.K,false))
      bOffset[name]=data.len;data.setLen(data.len+ad.r*g.N)
      copies.add((ad.bBuf,0,bOffset[name],ad.r*g.N,false))
  var p:VmProgram
  proc bases(mem,input,lut,weight:int)=
    p.emitUConst(31,uint32(mem));p.emitBaseF32(31)
    p.emitUConst(31,uint32(input));p.emitBaseIn(31)
    p.emitUConst(31,uint32(lut));p.emitBaseLut(31)
    p.emitUConst(31,uint32(weight));p.emitBaseU8(31)
  proc stage(body:VmProgram;mode,n,param:int)=
    odStage(p,body,uint32(mode),uint32(n),uint32(param))
    p.emitGridBarrier(0)
  proc element(body:VmProgram;n:int)=
    bases(0,0,0,0);stage(body,2,(n+255) div 256,n)
  proc linear(name:string;input,output:int)=
    let g=gLinCache[name]
    bases(output,input,lutOffset[name],weightOffset.getOrDefault(name))
    if g.isFp4:stage(buildFp4KmGemv2RowProgram(g.K,g.N),1,g.N div 2,336)
    else:stage(buildF32GemvProgram(g.K,g.N,1'f32,false),1,g.N,336)
    if name in gLoraGpu:
      let ad=gLoraGpu[name]
      bases(a.lora,input,aOffset[name],0)
      stage(buildF32GemvProgram(g.K,ad.r,1'f32,false),1,ad.r,336)
      bases(output,a.lora,bOffset[name],0)
      stage(buildF32GemvSmallKProgram(ad.r,g.N,ad.scale,true),2,(g.N+255) div 256,g.N)
    bases(output,0,biasOffset[name],0)
    stage(biasRows(g.N,0),2,(g.N+255) div 256,g.N)
  for i,blk in m.blocks:
    let w=params[i];let s=states[i];let root="block" & $i
    bases(0,0,0,0);stage(rmsRows(a.hidden,a.x,a.norm,w.norm,blk.rms1.eps),0,1,0)
    linear(root & ".mambaReal.inProj.weight",a.norm,a.proj)
    bases(0,0,0,0);stage(prepareSiso(a,s,w,1,blk.mambaReal.normEps),0,a.H,0)
    bases(0,0,0,0);stage(scanSiso(a,s,w,1),1,a.C,336)
    element(carrySiso(a,s,1),max(a.C,a.H*a.N))
    linear(root & ".mambaReal.outProj.weight",a.inner,a.mix)
    if blk.kind!=bkMambaReal:
      linear(root & ".attn.qkvProj.weight",a.norm,a.attnProj)
      bases(0,0,0,0);stage(linearPlusScan(a,s,blk.attn.nHeads,1),1,a.hidden,336)
      element(carryLinearPlus(a,s),a.hidden)
      linear(root & ".attn.outProj.weight",a.attnInner,a.attnOut)
      if blk.kind==bkMambaRealGated:
        element(gateInput(a),a.hidden)
        linear(root & ".fuseGate.weight",a.gateIn,a.gate)
        element(mixGate(a),a.hidden)
      else:element(residual(a.mix,a.attnOut,a.mix),a.hidden)
    element(residual(a.x,a.mix,a.x),a.hidden)
    bases(0,0,0,0);stage(rmsRows(a.hidden,a.x,a.norm,w.norm2,blk.mambaReal.normEps),0,1,0)
    linear(root & ".mambaReal.mlpFc1.weight",a.norm,a.mlp)
    element(siluMlp(a),a.F)
    linear(root & ".mambaReal.mlpFc2.weight",a.gated,a.mix)
    element(residual(a.x,a.mix,a.x),a.hidden)
  p.emitGridEpilogue(0);p.emitHalt()
  result.kernel=odCompile(p,"mamba3_persistent_" & m.pattern & "_" & $a.hidden)
  result.ntg=parseInt(getEnv("BYBY_OD_TG","1"))
  if result.ntg<1 or result.ntg>16:raise newException(ValueError,"BYBY_OD_TG must be 1..16")
  result.constants=sessionUpload(be,data)
  result.weights=sessionAllocScratch(be,max(1,(packedBytes+3) div 4))
  var opened=false
  try:
    if not sessionBegin(be):raise newException(IOError,"OD pack sessionBegin")
    opened=true
    for cp in copies:
      var target=if cp.packed:result.weights else:result.constants
      if not sessionCopyInto(be,cp.source,target,cp.src,cp.dst,cp.count):
        raise newException(IOError,"OD device weight copy failed")
    let ok=sessionEnd(be);opened=false
    if not ok:raise newException(IOError,"OD pack sessionEnd")
    result.ready=true
  finally:
    if opened:discard sessionEnd(be)
    if not result.ready:
      sessionFree(be,result.weights);sessionFree(be,result.constants)

proc encodeOneDispatch*(st:M3OneDispatch;arena:SessionHandle)=
  odDispatch(st.kernel,arena,st.weights,st.constants,st.ntg,8*336*4)
