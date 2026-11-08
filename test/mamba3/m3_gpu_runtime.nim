## GPU-resident R/P/G stack: batched FP4 prefill and one command buffer/token.
## The Mamba-3 state, prefix scratch, sidecar statistics and activations never
## round-trip through the host between layers or between prefix/decode.
import std/[tables, strutils, os, math, monotimes]
import backend, nimformer, quant, m3_hybrid, m3_inference_kernels
import gpu_layers, fp4_gemv_gpu, fp4_gemv_vm, session_vm
import m3_onedispatch
import prefill_inference_kernels
import vendor/bybylang/gpubackend as byby

type
  M3ProgramRequest* = object
    role*: string
    arena*: M3Arena
    state*: M3DeviceState
    params*: M3Params
    T*, finalGamma*, layers*: int
    epsilon*: float32
    dynamicOffsets*: bool
  M3ProvidedProgram* = object
    program*: VmProgram
    threads*, tgSize*, tgBytes*: int
  M3ProgramProvider* = proc(request: M3ProgramRequest): M3ProvidedProgram {.closure.}
  M3BlockStat* = object
    idx*:int
    kind*:char
    rms*,maxAbs*,mean*,inRms*,sideRms*,sideMax*:float32
    nonFinite*:int
  Kernel = object
    name:string
    threads,tg,bytes:int
  LayerPrograms = object
    norm,prepare,scan,carry,norm2,mlp,residual,lp,lpCarry,gateIn,gate,mix:Kernel
  RuntimePrograms = object
    layers:seq[LayerPrograms]
    finalNorm:Kernel
    bias,loraDown,loraUp,f32Batch:Table[string,Kernel]
  M3Runtime* = object
    ready*,poisoned*:bool
    be:GpuBackend
    a*:M3Arena
    states*:seq[M3DeviceState]
    params:seq[M3Params]
    paramData:seq[float32]
    layerOffsets:seq[SessionHandle]
    arena*,constants:SessionHandle
    readback:SessionHandle
    zeros:SessionHandle
    stateStart,stateEnd:int
    programs:Table[int,RuntimePrograms]
    modelKey:string
    tokens*:int
    prefillCalls*,decodeCalls*:int
    kernelSetupSeconds*:float64
    skipAttn*:bool
    od:M3OneDispatch
    lastTrace*:seq[M3BlockStat]

var gM3Runtime*:M3Runtime
var gM3ProgramProvider*:M3ProgramProvider

proc setM3ProgramProvider*(provider:M3ProgramProvider) =
  # Cached programs must be rebuilt when the source of their bytecode changes.
  gM3Runtime.programs.clear()
  gM3ProgramProvider=provider

proc require(ok:bool;message:string)=
  if not ok:raise newException(CatchableError,"Mamba-3 GPU: " & message)

proc compile(p:VmProgram;tag:string;threads,tg:int;bytes=1024;be=gbMetal):Kernel =
  Kernel(name:vmKernelForBe(p,tag,be),threads:threads,tg:tg,bytes:bytes)

proc encode(k:Kernel;st:M3Runtime;constant:SessionHandle)=
  encodeVmKernel(k.name,k.threads,k.tg,k.bytes,st.arena,0,
    gDummyFor(st.be),constant,st.arena,0)

proc encode(k:Kernel;st:M3Runtime;constant,offsets:SessionHandle)=
  encodeVmKernel(k.name,k.threads,k.tg,k.bytes,st.arena,0,
    offsets,constant,st.arena,0)

proc releaseM3Runtime*() =
  if gM3Runtime.ready:
    release(gM3Runtime.od,gM3Runtime.be)
    sessionFree(gM3Runtime.be,gM3Runtime.arena)
    sessionFree(gM3Runtime.be,gM3Runtime.constants)
    for offsets in gM3Runtime.layerOffsets.mitems:sessionFree(gM3Runtime.be,offsets)
    sessionFree(gM3Runtime.be,gM3Runtime.zeros)
    sessionFree(gM3Runtime.be,gM3Runtime.readback)
  gM3Runtime=default(M3Runtime)

proc prepareM3Runtime*(m:var NimformerModelJamba;ctx:Backend)=
  releaseM3Runtime()
  var st:M3Runtime
  st.be=ctx.kind.toByby();require(st.be in {gbMetal,gbVulkan},"resident runtime requires Metal/Vulkan")
  ensureDummy(st.be)
  let mr=m.blocks[0].mambaReal
  var rank=1
  for _,ad in gLoraGpu:rank=max(rank,ad.r)
  let capacity=max(4,parseInt(getEnv("MAMBA3_PREFILL_CHUNK","128")))
  require(capacity<=2048,"MAMBA3_PREFILL_CHUNK must be <=2048")
  st.a=arenaLayout(m.embedDim,mr.intermediateSize,mr.numHeads,mr.dState,m.vocab,((capacity+7) div 8)*8,rank)
  st.stateStart=st.a.total
  var endState=st.stateStart
  template param(v:seq[float32]):int =
    block:
      let offset=st.paramData.len
      st.paramData.add(v)
      offset
  for i,blk in m.blocks:
    require(blk.kind in {bkMambaReal,bkMambaRealPlusAttn,bkMambaRealGated},"R/P/G patterns only")
    let mm=blk.mambaReal
    require(mm.intermediateSize==st.a.C and mm.numHeads==st.a.H and mm.dState==st.a.N,"mixed mixer dimensions")
    let heads=if blk.kind==bkMambaReal:0 else:blk.attn.nHeads
    let s=stateLayout(st.a,endState,heads)
    st.states.add(s);endState=s.total
    var w:M3Params
    w.norm=param(blk.rms1.gamma.data);w.norm2=param(mm.mlpNorm.data)
    w.bNorm=param(mm.bNorm.data);w.cNorm=param(mm.cNorm.data)
    w.bBias=param(mm.bBias.data);w.cBias=param(mm.cBias.data)
    w.dt=param(mm.dtBias.data);w.skip=param(mm.dSkip.data)
    st.params.add(w)
    var offsets=newSeq[float32](ord(high(M3OffsetSlot))+1)
    template setOffset(slot:M3OffsetSlot; value:int)=
      offsets[ord(slot)]=cast[float32](uint32(value))
    setOffset(moNorm,w.norm);setOffset(moNorm2,w.norm2)
    setOffset(moPhase,s.phase);setOffset(moPrevK,s.prevK)
    setOffset(moPrevV,s.prevV);setOffset(moSsm,s.ssm)
    setOffset(moBNorm,w.bNorm);setOffset(moCNorm,w.cNorm)
    setOffset(moBBias,w.bBias);setOffset(moCBias,w.cBias)
    setOffset(moDt,w.dt);setOffset(moSkip,w.skip)
    setOffset(moLpS,s.lpS);setOffset(moLpZ,s.lpZ);setOffset(moLpZNext,s.lpZNext)
    st.layerOffsets.add(sessionUpload(st.be,offsets))
  st.stateEnd=endState
  discard param(m.rmsF.gamma.data)
  st.constants=sessionUpload(st.be,st.paramData)
  st.arena=sessionUpload(st.be,newSeq[float32](endState))
  st.zeros=sessionUpload(st.be,newSeq[float32](endState-st.stateStart))
  st.readback=sessionAllocScratch(st.be,m.vocab+4)
  st.ready=true;st.modelKey=m.pattern & ":" & $m.embedDim & ":" & $m.vocab
  gM3Runtime=st

proc resetM3Runtime*(ctx:Backend)=
  if not gM3Runtime.ready:return
  let be=ctx.kind.toByby()
  require(be==gM3Runtime.be,"reset backend mismatch")
  var opened=false
  try:
    require(sessionBegin(be),"reset sessionBegin")
    opened=true
    require(sessionCopyInto(be,gM3Runtime.zeros,gM3Runtime.arena,0,0,64),"reset persistent control")
    require(sessionCopyInto(be,gM3Runtime.zeros,gM3Runtime.arena,0,
      gM3Runtime.stateStart,gM3Runtime.stateEnd-gM3Runtime.stateStart),"reset state copy")
    let ok=sessionEnd(be);opened=false;require(ok,"reset sessionEnd")
    gM3Runtime.tokens=0;gM3Runtime.poisoned=false
  finally:
    if opened:discard sessionEnd(be)

proc preparePrograms(m:NimformerModelJamba;T:int)=
  if T in gM3Runtime.programs:return
  let a=gM3Runtime.a;let be=gM3Runtime.be
  var pr:RuntimePrograms
  template provided(roleName:string; stateValue:M3DeviceState; paramsValue:M3Params;
      epsValue:float32; dynamicValue:bool; gammaValue:int; tagValue:string;
      fallback:untyped):Kernel =
    block:
      if gM3ProgramProvider.isNil:
        fallback
      else:
        let built=gM3ProgramProvider(M3ProgramRequest(role:roleName,arena:a,
          state:stateValue,params:paramsValue,T:T,epsilon:epsValue,
          dynamicOffsets:dynamicValue,finalGamma:gammaValue,layers:m.blocks.len))
        require(built.program.words.len>1 and built.threads>0 and built.tgSize>0 and
          built.threads mod built.tgSize==0 and built.tgBytes>=0,"invalid provider program: " & roleName)
        compile(built.program,tagValue,built.threads,built.tgSize,built.tgBytes,be)
  for i,blk in m.blocks:
    let s=gM3Runtime.states[i];let w=gM3Runtime.params[i]
    # The cache also hashes every bytecode word (including embedded state and
    # parameter offsets). Stable tags let identical programs share a pipeline
    # across layers and chunk lengths; dispatch sizes remain in each Kernel.
    let tag="m3resident_" & $a.hidden & "_" & $a.C & "_" & $a.H & "_" & $a.N
    var lp:LayerPrograms
    lp.norm=provided("norm",s,w,blk.rms1.eps,true,0,tag & "_norm"):
      compile(rmsRows(a.hidden,a.x,a.norm,w.norm,blk.rms1.eps,true,moNorm),tag & "_norm",T*256,256,1024,be)
    lp.prepare=provided("prepare",s,w,blk.mambaReal.normEps,true,0,tag & "_prepare"):
      compile(prepareSiso(a,s,w,T,blk.mambaReal.normEps,true),tag & "_prepare",a.H*256,256,1024,be)
    lp.scan=provided("scan",s,w,blk.mambaReal.normEps,true,0,tag & "_scan"):
      compile(scanSiso(a,s,w,T,true),tag & "_scan",a.C*32,32,128,be)
    lp.carry=provided("carry",s,w,blk.mambaReal.normEps,true,0,tag & "_carry"):
      compile(carrySiso(a,s,T,true),tag & "_carry",max(a.C,a.H*a.N),256,16,be)
    lp.norm2=provided("norm2",s,w,blk.mambaReal.normEps,true,0,tag & "_norm2"):
      compile(rmsRows(a.hidden,a.x,a.norm,w.norm2,blk.mambaReal.normEps,true,moNorm2),tag & "_norm2",T*256,256,1024,be)
    lp.mlp=provided("mlp",s,w,blk.mambaReal.normEps,true,0,tag & "_mlp"):
      compile(siluMlp(a),tag & "_mlp",T*a.F,256,16,be)
    lp.residual=provided("residual",s,w,blk.mambaReal.normEps,true,0,tag & "_residual"):
      compile(residual(a.x,a.mix,a.x),tag & "_residual",T*a.hidden,256,16,be)
    if blk.kind!=bkMambaReal:
      lp.lp=compile(linearPlusScan(a,s,blk.attn.nHeads,T,true),tag & "_lp",a.hidden*32,32,128,be)
      lp.lpCarry=compile(carryLinearPlus(a,s,true),tag & "_lpCarry",a.hidden,256,16,be)
      lp.mix=compile(residual(a.mix,a.attnOut,a.mix),tag & "_sum",T*a.hidden,256,16,be)
      if blk.kind==bkMambaRealGated:
        lp.gateIn=compile(gateInput(a),tag & "_gateIn",T*a.hidden,256,16,be)
        lp.gate=compile(mixGate(a),tag & "_gate",T*a.hidden,256,16,be)
    pr.layers.add(lp)
  let finalGamma=gM3Runtime.paramData.len-a.hidden
  pr.finalNorm=provided("finalNorm",default(M3DeviceState),default(M3Params),m.rmsF.eps,false,finalGamma,"m3final_" & $a.hidden):
    compile(rmsRows(a.hidden,a.x,a.norm,finalGamma,m.rmsF.eps),"m3final_" & $a.hidden,T*256,256,1024,be)
  for name,lin in gLinCache:
    if not name.startsWith("block") and name!="outProj.weight":continue
    pr.bias[name]=compile(biasRows(lin.N,0),"m3_bias_" & $lin.N,T*lin.N,256,16,be)
    if not lin.isFp4 and not lin.isInt4 and lin.K mod 32==0:
      pr.f32Batch[name]=compile(buildF32GemmBatchProgram(lin.K,lin.N,32,8),
        "m3_f32batch8_" & $lin.K & "_" & $lin.N,lin.N*32,32,8*32*4,be)
    if name in gLoraGpu:
      let ad=gLoraGpu[name]
      # The down-projection writes [T,rank] with one 32-lane group per row/rank.
      pr.loraDown[name]=compile(loraDown(lin.K,ad.r),
        "m3_ldown_" & $lin.K & "_" & $ad.r,T*ad.r*32,32,128,be)
      pr.loraUp[name]=compile(loraRows(lin.N,ad.r,0,a.lora,ad.scale),
        "m3_lup_" & $lin.N & "_" & $ad.r & "_" & $ad.scale,T*lin.N,256,16,be)
  gM3Runtime.programs[T]=pr

proc encodeLinear(st:M3Runtime;pr:RuntimePrograms;name:string;input,output,T:int;bias=true)=
  require(name in gLinCache,"unprepared projection " & name)
  let lin=gLinCache[name]
  if T==1:
    encodeLin(lin,st.arena,input,st.arena,output,bias=bias)
    return
  if lin.isFp4:
    let cache=gFp4CacheGpu[name]
    var row=0
    if cache.m8JitName.len>0 and getEnv("BYBY_PREFILL_FP4_KERNEL","auto")!="m4":
      let tg=max(32,cache.m8TgSize)
      while row+8<=T:
        encodeVmKernel(cache.m8JitName,lin.N*tg,tg,(272+8*tg)*4,
          st.arena,output+row*lin.N,cache.wBuf,cache.lutBuf,st.arena,input+row*lin.K)
        row+=8
    while row+4<=T and cache.m4JitName.len>0:
      require(encodeFp4GemmM4(cache,st.arena,input+row*lin.K,st.arena,output+row*lin.N),"FP4 M4 " & name)
      row+=4
    while row<T:
      require(encodeFp4Gemv(cache,st.arena,input+row*lin.K,st.arena,output+row*lin.N),"FP4 tail " & name)
      inc row
  elif not lin.isInt4:
    # Native NT matrix projection for the fully trained Linear++ sidecar.
    # Offset-aware VM rows avoid host copies and work on both backends.
    var row=0
    if name in pr.f32Batch:
      let k=pr.f32Batch[name]
      while row+8<=T:
        encodeVmKernel(k.name,k.threads,k.tg,k.bytes,st.arena,output+row*lin.N,
          gDummyFor(st.be),lin.wF32,st.arena,input+row*lin.K)
        row+=8
    while row<T:
      encodeVmKernel(lin.knF32,lin.N*32,32,f32GemvTgBytes(),st.arena,output+row*lin.N,
        gDummyFor(st.be),lin.wF32,st.arena,input+row*lin.K)
      inc row
  else:
    for row in 0..<T:
      encodeLin(lin,st.arena,input+row*lin.K,st.arena,output+row*lin.N,bias=bias)
    return # encodeLin includes bias + LoRA.
  if name in gLoraGpu and not gDisableLora:
    # Reuse the existing down-projection's token/rank addressing and the
    # Codestral fused row epilogue; base packed matrices stay compressed.
    let ad=gLoraGpu[name]
    let kd=pr.loraDown[name]
    encodeVmKernel(kd.name,kd.threads,kd.tg,kd.bytes,st.arena,st.a.lora,
      gDummyFor(st.be),ad.aBuf,st.arena,input)
    let ku=pr.loraUp[name]
    # loraRows reads scratch at an absolute arena offset, output is based here.
    encodeVmKernel(ku.name,ku.threads,ku.tg,ku.bytes,st.arena,output,
      gDummyFor(st.be),ad.bBuf,st.arena,0)
  if bias and lin.hasBias:
    let k=pr.bias[name]
    encodeVmKernel(k.name,k.threads,k.tg,k.bytes,st.arena,output,gDummyFor(st.be),lin.biasH,st.arena,0)

proc runM3Resident*(m:var NimformerModelJamba;ids:seq[int];ctx:Backend;produceLogits=true;trace=false):Tensor =
  require(gM3Runtime.ready and not gM3Runtime.poisoned,"state unavailable; reset after a failed command buffer")
  require(ids.len>0 and ids.len<=gM3Runtime.a.capacity,"invalid chunk length")
  for id in ids:require(id>=0 and id<m.vocab,"token outside vocabulary")
  let T=ids.len;let a=gM3Runtime.a;let be=gM3Runtime.be
  let setupStart=getMonoTime().ticks
  preparePrograms(m,T) # compile before opening the command buffer
  let pr=gM3Runtime.programs[T]
  let odMode=getEnv("BYBY_ONE_DISPATCH","0")
  require(odMode in ["0","1","auto"],"BYBY_ONE_DISPATCH must be 0, 1 or auto")
  var useOd=T==1 and produceLogits and odMode!="0" and not gDisableLora and not gM3Runtime.skipAttn and not trace
  require(not useOd or gM3ProgramProvider.isNil,"program provider requires the resident micro-kernel path (BYBY_ONE_DISPATCH=0)")
  if useOd and not gM3Runtime.od.ready:
    let reason=eligible(m,ctx)
    if reason.len>0:
      if odMode=="1":raise newException(ValueError,"Mamba-3 one-dispatch: " & reason)
      useOd=false
    else:
      gM3Runtime.od=prepareOneDispatch(m,ctx,a,gM3Runtime.states,gM3Runtime.params,gM3Runtime.paramData)
  gM3Runtime.kernelSetupSeconds+=float64(getMonoTime().ticks-setupStart)/1e9
  # Embedding lookup is the only host activation; the entire stack and state
  # remain resident after this bounded input upload, as in Codestral.
  let input=m.embed.lookupBatch(@[ids],ctx)
  var hInput=sessionUpload(be,input.data)
  var hTrace:SessionHandle
  if trace:hTrace=sessionUpload(be,newSeq[float32](4*m.blocks.len*a.hidden))
  var opened=false
  defer:
    if trace:sessionFree(be,hTrace)
    if opened:
      discard sessionEnd(be)
      discard sessionRecoverQueue(be)
    sessionFree(be,hInput)
  try:
    require(sessionBegin(be),"sessionBegin")
    opened=true
    require(sessionCopyInto(be,hInput,gM3Runtime.arena,0,a.x,T*a.hidden),"input upload")
    if useOd:
      encodeOneDispatch(gM3Runtime.od,gM3Runtime.arena)
    else:
      for i,blk in m.blocks:
        let lp=pr.layers[i]
        let offsets=gM3Runtime.layerOffsets[i]
        let root="block" & $i
        if trace:require(sessionCopyInto(be,gM3Runtime.arena,hTrace,a.x,(i*4)*a.hidden,a.hidden),"trace input")
        encode(lp.norm,gM3Runtime,gM3Runtime.constants,offsets)
        encodeLinear(gM3Runtime,pr,root & ".mambaReal.inProj.weight",a.norm,a.proj,T)
        encode(lp.prepare,gM3Runtime,gM3Runtime.constants,offsets)
        encode(lp.scan,gM3Runtime,gM3Runtime.constants,offsets)
        encode(lp.carry,gM3Runtime,gM3Runtime.constants,offsets)
        encodeLinear(gM3Runtime,pr,root & ".mambaReal.outProj.weight",a.inner,a.mix,T)
        if trace:require(sessionCopyInto(be,gM3Runtime.arena,hTrace,a.mix,(i*4+1)*a.hidden,a.hidden),"trace mixer")
        if blk.kind!=bkMambaReal and not gM3Runtime.skipAttn:
          encodeLinear(gM3Runtime,pr,root & ".attn.qkvProj.weight",a.norm,a.attnProj,T)
          encode(lp.lp,gM3Runtime,gM3Runtime.constants,offsets)
          encode(lp.lpCarry,gM3Runtime,gM3Runtime.constants,offsets)
          encodeLinear(gM3Runtime,pr,root & ".attn.outProj.weight",a.attnInner,a.attnOut,T)
          if trace:require(sessionCopyInto(be,gM3Runtime.arena,hTrace,a.attnOut,(i*4+2)*a.hidden,a.hidden),"trace sidecar")
          if blk.kind==bkMambaRealGated:
            encode(lp.gateIn,gM3Runtime,gM3Runtime.constants,offsets)
            encodeLinear(gM3Runtime,pr,root & ".fuseGate.weight",a.gateIn,a.gate,T)
            encode(lp.gate,gM3Runtime,gM3Runtime.constants,offsets)
          else:encode(lp.mix,gM3Runtime,gM3Runtime.constants,offsets)
        encode(lp.residual,gM3Runtime,gM3Runtime.constants,offsets)
        encode(lp.norm2,gM3Runtime,gM3Runtime.constants,offsets)
        encodeLinear(gM3Runtime,pr,root & ".mambaReal.mlpFc1.weight",a.norm,a.mlp,T)
        encode(lp.mlp,gM3Runtime,gM3Runtime.constants,offsets)
        encodeLinear(gM3Runtime,pr,root & ".mambaReal.mlpFc2.weight",a.gated,a.mix,T)
        encode(lp.residual,gM3Runtime,gM3Runtime.constants,offsets)
        if trace:require(sessionCopyInto(be,gM3Runtime.arena,hTrace,a.x,(i*4+3)*a.hidden,a.hidden),"trace output")
    if produceLogits:
      encode(pr.finalNorm,gM3Runtime,gM3Runtime.constants)
      encodeLinear(gM3Runtime,pr,"outProj.weight",a.norm+(T-1)*a.hidden,a.logits,1)
    if produceLogits:
      require(sessionCopyInto(be,gM3Runtime.arena,gM3Runtime.readback,a.logits,0,m.vocab),"logits copy")
      if useOd:require(sessionCopyInto(be,gM3Runtime.arena,gM3Runtime.readback,0,m.vocab,4),"OD status copy")
    let ok=sessionEnd(be);opened=false
    require(ok,"sessionEnd: " & sessionGetLastError(be))
    if trace:
      let values=sessionRead(be,hTrace,4*m.blocks.len*a.hidden)
      gM3Runtime.lastTrace.setLen(0)
      for i,blk in m.blocks:
        var st=M3BlockStat(idx:i,kind:($blk.kind)[0],sideRms: -1'f32)
        var sum,ss,inputSq,sideSq=0.0
        for d in 0..<a.hidden:
          let value=values[(i*4+3)*a.hidden+d]
          if value.classify in {fcNan,fcInf,fcNegInf}:inc st.nonFinite
          else:
            sum+=float64(value);ss+=float64(value)*float64(value)
            st.maxAbs=max(st.maxAbs,abs(value))
          let input=values[(i*4)*a.hidden+d]
          let side=values[(i*4+2)*a.hidden+d]
          inputSq+=float64(input)*float64(input)
          sideSq+=float64(side)*float64(side)
          st.sideMax=max(st.sideMax,abs(side))
        st.mean=float32(sum/float64(a.hidden));st.rms=float32(sqrt(ss/float64(a.hidden)))
        st.inRms=float32(sqrt(inputSq/float64(a.hidden)))
        if blk.kind!=bkMambaReal:st.sideRms=float32(sqrt(sideSq/float64(a.hidden)))
        gM3Runtime.lastTrace.add(st)
    gM3Runtime.tokens+=T
    if T>1:inc gM3Runtime.prefillCalls else:inc gM3Runtime.decodeCalls
    if produceLogits:
      let raw=sessionRead(be,gM3Runtime.readback,m.vocab+(if useOd:4 else:0))
      if useOd:require(cast[uint32](raw[m.vocab+3])==0,"persistent grid barrier failed; reset state")
      result=Tensor(shape: @[1,1,m.vocab],data:raw[0..<m.vocab])
    else:result=newTensor(@[1,1,0])
  except CatchableError:
    gM3Runtime.poisoned=true
    raise

proc prefillM3Resident*(m:var NimformerModelJamba;ids:seq[int];ctx:Backend)=
  var offset=0
  while offset<ids.len:
    let count=min(ids.len-offset,gM3Runtime.a.capacity)
    discard runM3Resident(m,ids[offset..<offset+count],ctx,produceLogits=false)
    offset+=count

proc invalidateM3OneDispatch*() =
  if gM3Runtime.ready:release(gM3Runtime.od,gM3Runtime.be)
