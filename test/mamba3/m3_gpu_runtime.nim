## GPU-resident R/P/G stack: batched FP4 prefill and one command buffer/token.
## The Mamba-3 state, prefix scratch, sidecar statistics and activations never
## round-trip through the host between layers or between prefix/decode.
import std/[tables, strutils, os, math, monotimes, bitops]
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
    bias,f32Batch,f16Batch:Table[string,Kernel]
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

type M3Graph = object
  ## Decode graph (the DeepSeek approach): one token's dispatches are recorded once into an indirect
  ## command buffer and replayed for every following token, so the CPU stops encoding them.
  ## The token comes from a one-slot ring that the host sets, a micro-VM kernel gathers its
  ## embedding row, and the logits land in their own buffer.
  tried,valid,failed:bool
  ring,table,logits:SessionHandle
  embed:Kernel
  rows:int

var gM3Graph:M3Graph

proc releaseM3Graph*() =
  if gM3Graph.ring.metalHandle!=nil or gM3Graph.valid:
    let be=gM3Runtime.be
    if gM3Graph.valid:sessionGraphReset(be)
    if gM3Graph.ring.metalHandle!=nil:sessionFree(be,gM3Graph.ring)
    if gM3Graph.table.metalHandle!=nil:sessionFree(be,gM3Graph.table)
    if gM3Graph.logits.metalHandle!=nil:sessionFree(be,gM3Graph.logits)
  gM3Graph=default(M3Graph)


proc setM3ProgramProvider*(provider:M3ProgramProvider) =
  # Cached programs must be rebuilt when the source of their bytecode changes.
  gM3Runtime.programs.clear()
  gM3ProgramProvider=provider

proc require(ok:bool;message:string)=
  if not ok:raise newException(CatchableError,"Mamba-3 GPU: " & message)

proc compile(p:VmProgram;tag:string;threads,tg:int;bytes=1024;be=gbMetal):Kernel =
  Kernel(name:vmKernelForBe(p,tag,be),threads:threads,tg:tg,bytes:bytes)

proc m3ProfileOn():bool =
  ## BYBY_M3_PROFILE=1 prints, for every decode step, the host time spent encoding the step, waiting
  ## for the GPU (commit + wait) and reading the logits back.
  getEnv("BYBY_M3_PROFILE","")=="1"

var gM3GpuSecs:float
proc m3ProfileLine(kind:string;tA,tB,tC,tD:int64) =
  proc ms(a,b:int64):string = formatFloat(float64(b-a)/1e6,ffDecimal,2)
  let g=sessionGpuSeconds(gM3Runtime.be)
  echo "[m3-profile] ",kind," encode ",ms(tA,tB)," ms, gpu wait ",ms(tB,tC)," ms, read ",ms(tC,tD)," ms, gpu ",formatFloat((g-gM3GpuSecs)*1000,ffDecimal,2)," ms"
  gM3GpuSecs=g

# Prefill scheduler (the DeepSeek prefill approach): the prompt goes through in micro-batches of rows,
# and a micro-batch goes through in command buffers sized by the measured speed where the OS limits
# how long one command buffer may run (Intel/AMD Macs, Vulkan), or in one command buffer committed
# early every few layers where it does not (Apple GPUs), so the GPU starts on the first layers while
# the host still encodes the rest.
proc m3EnvInt(name:string;dflt:int):int =
  let raw=getEnv(name,"")
  if raw.len==0:return dflt
  try:result=parseInt(raw.strip())
  except ValueError:
    stderr.writeLine "[m3] " & name & "='" & raw & "' is not an integer - using " & $dflt
    result=dflt

proc m3EnvFloat(name:string;dflt:float64):float64 =
  let raw=getEnv(name,"")
  if raw.len==0:return dflt
  try:result=parseFloat(raw.strip())
  except ValueError:
    stderr.writeLine "[m3] " & name & "='" & raw & "' is not a number - using " & $dflt
    result=dflt

proc m3UbatchRows():int =
  ## BYBY_UBATCH: rows per prefill micro-batch (llama.cpp's n_ubatch). 0 or unset keeps the arena
  ## capacity (MAMBA3_PREFILL_CHUNK, 128 rows); a larger value grows the arena to hold it.
  max(0,m3EnvInt("BYBY_UBATCH",0))

proc m3PrefillCbMs():float64 =
  ## BYBY_PREFILL_CB_MS: GPU time aimed at per prefill command buffer where command buffers are paced
  ## (default 300 ms, as DeepSeek); 0 puts the whole micro-batch into one command buffer.
  max(0.0,m3EnvFloat("BYBY_PREFILL_CB_MS",300.0))

proc m3PrefillLayersPerCb():int =
  ## BYBY_PREFILL_LAYERS_PER_CB: where command buffers are not paced, commit (without waiting) after
  ## this many layers so the GPU can start early; 0 keeps one commit per micro-batch.
  max(0,m3EnvInt("BYBY_PREFILL_LAYERS_PER_CB",4))

var gM3PrefillMsPerLayerRow:float64 ## smoothed ms of one layer on one (tile-rounded) row; 0 = not measured yet
var gM3PrefillLogged:bool

proc m3Tile(rows:int):int = (rows+7) div 8*8 ## the FP4 prefill GEMM works in 8-row tiles

proc encode(k:Kernel;st:M3Runtime;constant:SessionHandle)=
  encodeVmKernel(k.name,k.threads,k.tg,k.bytes,st.arena,0,
    gDummyFor(st.be),constant,st.arena,0)

proc encode(k:Kernel;st:M3Runtime;constant,offsets:SessionHandle)=
  encodeVmKernel(k.name,k.threads,k.tg,k.bytes,st.arena,0,
    offsets,constant,st.arena,0)

proc releaseM3Pipe*()
proc releaseM3Runtime*() =
  if gM3Runtime.ready:
    releaseM3Graph()
    releaseM3Pipe()
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
  let capacity=max(max(4,parseInt(getEnv("MAMBA3_PREFILL_CHUNK","128"))),m3UbatchRows())
  require(capacity<=2048,"MAMBA3_PREFILL_CHUNK and BYBY_UBATCH must be <=2048")
  gM3PrefillMsPerLayerRow=0
  st.a=arenaLayout(m.embedDim,mr.intermediateSize,mr.numHeads,mr.dState,m.vocab,((capacity+7) div 8)*8)
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

proc m3FuseOn():bool =
  ## BYBY_M3_FUSE=0 keeps the residual adds and the SiLU gate of a decode step as their own
  ## dispatches (the reference for the fused path).
  getEnv("BYBY_M3_FUSE","1")!="0"

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
    if lin.isF16:
      # FP16 sidecar (BYBY_LINEARPP_DTYPE=fp16): 8 rows per dispatch, scalar input loads
      # because arena row offsets are not 16-byte aligned.
      pr.f16Batch[name]=compile(buildF16GemmProgram(lin.K,lin.N,8,vecInput=false),
        "m3_f16batch8_" & $lin.K & "_" & $lin.N,lin.N*32,32,f16GemmTgBytes(8),be)
    elif not lin.isFp4 and not lin.isInt4 and lin.K mod 32==0:
      pr.f32Batch[name]=compile(buildF32GemmBatchProgram(lin.K,lin.N,32,8),
        "m3_f32batch8_" & $lin.K & "_" & $lin.N,lin.N*32,32,8*32*4,be)
  if T==1 and m3FuseOn():
    # The fused decode GEMVs of encodeStack, built here so they are compiled with the rest.
    for i,blk in m.blocks:
      let root="block" & $i & ".mambaReal."
      for (name,gated) in [(root & "outProj.weight",false),(root & "mlpFc2.weight",true)]:
        if (gated or blk.kind==bkMambaReal) and name in gLinCache and gLinCache[name].isFp4 and
           not gLinCache[name].hasBias:
          discard fp4DecodeFused(name,accumulate=true,gatedInput=gated)
  prewarmVulkanJit(be)
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
  elif lin.isF16:
    # FP16 Linear++ sidecar: 8-row batches, then the single-row FP16 GEMV of encodeLin.
    var row=0
    let k=pr.f16Batch[name]
    while row+8<=T:
      encodeVmKernel(k.name,k.threads,k.tg,k.bytes,st.arena,output+row*lin.N,
        lin.wF16,gDummyFor(st.be),st.arena,input+row*lin.K)
      row+=8
    while row<T:
      encodeLin(lin,st.arena,input+row*lin.K,st.arena,output+row*lin.N,bias=false)
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
    return # encodeLin includes the bias.
  if bias and lin.hasBias:
    let k=pr.bias[name]
    encodeVmKernel(k.name,k.threads,k.tg,k.bytes,st.arena,output,gDummyFor(st.be),lin.biasH,st.arena,0)


proc fusedLinear(st:M3Runtime;name:string;input,output:int;gatedInput:bool):bool =
  ## Decode only (one row): output += W input, or W (in * silu(in[K..])) with gatedInput, as one
  ## FP4 GEMV. False when the projection cannot take the fused form (the caller encodes the plain
  ## sequence instead).
  if name notin gLinCache:return false
  let lin=gLinCache[name]
  if not lin.isFp4 or lin.hasBias:return false
  let k=fp4DecodeFused(name,accumulate=true,gatedInput=gatedInput)
  if k.kn.len==0:return false
  let c=gFp4CacheGpu[name]
  encodeVmKernel(k.kn,k.threads,k.tg,k.bytes,st.arena,output,c.wBuf,c.lutBuf,st.arena,input)
  true

proc encodeStack(m:NimformerModelJamba;pr:RuntimePrograms;T:int;hTrace:var SessionHandle;trace:bool;
    firstLayer=0;lastLayer=high(int);commitEvery=0)=
  ## Blocks firstLayer ..< lastLayer of the stack for T rows, into the open session (or the decode graph
  ## being recorded). With commitEvery>0 the session is committed without waiting after every
  ## commitEvery blocks, so the GPU starts on the first blocks while the host still encodes the rest.
  let a=gM3Runtime.a;let be=gM3Runtime.be
  let fuse=T==1 and m3FuseOn()
  for i in firstLayer..<min(lastLayer,m.blocks.len):
    let blk=m.blocks[i]
    if commitEvery>0 and i>firstLayer and (i-firstLayer) mod commitEvery==0:
      require(sessionCommitContinue(be),"commit continue")
    let lp=pr.layers[i]
    let offsets=gM3Runtime.layerOffsets[i]
    let root="block" & $i
    if trace:require(sessionCopyInto(be,gM3Runtime.arena,hTrace,a.x,(i*4)*a.hidden,a.hidden),"trace input")
    encode(lp.norm,gM3Runtime,gM3Runtime.constants,offsets)
    encodeLinear(gM3Runtime,pr,root & ".mambaReal.inProj.weight",a.norm,a.proj,T)
    encode(lp.prepare,gM3Runtime,gM3Runtime.constants,offsets)
    encode(lp.scan,gM3Runtime,gM3Runtime.constants,offsets)
    encode(lp.carry,gM3Runtime,gM3Runtime.constants,offsets)
    # One decode row of a plain R block: the output projection adds itself into the residual.
    let fuseOut=fuse and blk.kind==bkMambaReal and not trace and
      fusedLinear(gM3Runtime,root & ".mambaReal.outProj.weight",a.inner,a.x,false)
    if not fuseOut:
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
    if not fuseOut:encode(lp.residual,gM3Runtime,gM3Runtime.constants,offsets)
    encode(lp.norm2,gM3Runtime,gM3Runtime.constants,offsets)
    encodeLinear(gM3Runtime,pr,root & ".mambaReal.mlpFc1.weight",a.norm,a.mlp,T)
    # The SiLU gate and the residual add of the MLP fold into the down projection.
    if not (fuse and fusedLinear(gM3Runtime,root & ".mambaReal.mlpFc2.weight",a.mlp,a.x,true)):
      encode(lp.mlp,gM3Runtime,gM3Runtime.constants,offsets)
      encodeLinear(gM3Runtime,pr,root & ".mambaReal.mlpFc2.weight",a.gated,a.mix,T)
      encode(lp.residual,gM3Runtime,gM3Runtime.constants,offsets)
    if trace:require(sessionCopyInto(be,gM3Runtime.arena,hTrace,a.x,(i*4+3)*a.hidden,a.hidden),"trace output")

proc m3GraphWanted(be:GpuBackend):bool =
  ## Opt-in (BYBY_GRAPH=1). The replayed graph gives the same text as the ordinary decode (checked on
  ## Intel Iris Plus 640), but there it is slower than encoding directly, so it is not the default.
  ## DeepSeek turns its graph on where command buffers are not paced (Apple GPUs); that default is
  ## left off here until a run on an Apple GPU has been checked.
  be==gbMetal and getEnv("BYBY_GRAPH","")=="1"

proc embedGatherProgram(a:M3Arena;tg:int):VmProgram =
  ## mem[a.x + g] = table[ring[0] * hidden + g] for g in 0 ..< hidden.
  var p:VmProgram
  p.emitUTgid(0);p.emitULid(1)
  p.emitUMulK(2,0,uint32(tg));p.emitUAdd(2,2,1)
  p.emitUConst(3,0);p.emitULoadU32(4,3,0)     # token id from the ring (read as bytes)
  p.emitUMulK(4,4,uint32(a.hidden));p.emitUAdd(4,4,2)
  p.emitULoadIn(1,4,0)
  p.emitUStoreF32(1,2,uint32(a.x))
  p.emitHalt();p

proc m3GraphStep(m:var NimformerModelJamba;token:int;pr:RuntimePrograms;logits:var seq[float32]):bool =
  ## One decode token through the recorded graph; false when the graph cannot be used (the caller
  ## runs the ordinary path, nothing has executed on the GPU in that case).
  if gM3Graph.failed:return false
  let be=gM3Runtime.be;let a=gM3Runtime.a
  let prof=m3ProfileOn()
  if not gM3Graph.tried:
    let initStart=getMonoTime().ticks
    gM3Graph.tried=true
    var tg=0
    for cand in [256,128,64,32]:
      if a.hidden mod cand==0:
        tg=cand;break
    let n=m.vocab*a.hidden
    if tg==0 or m.embed.weight.floatValues.len<n:
      gM3Graph.failed=true;return false
    gM3Graph.table=sessionUploadPrivate(be,m.embed.weight.floatValues.toOpenArray(0,n-1))
    gM3Graph.ring=sessionAllocScratch(be,8)
    gM3Graph.logits=sessionAllocScratch(be,m.vocab)
    gM3Graph.embed=compile(embedGatherProgram(a,tg),"m3_embed_gather_" & $a.hidden,a.hidden,tg,16,be)
    gM3Graph.rows=a.hidden
    gM3Runtime.kernelSetupSeconds+=float64(getMonoTime().ticks-initStart)/1e9
  sessionTokenRingSet(be,gM3Graph.ring,0,token)
  var opened=false
  var tA,tB,tC:int64
  var recorded=false
  try:
    tA=getMonoTime().ticks
    require(sessionBegin(be),"sessionBegin")
    opened=true
    if not gM3Graph.valid:
      if not sessionGraphBegin(be,1):
        discard sessionEnd(be);opened=false
        gM3Graph.failed=true;return false
      let k=gM3Graph.embed
      encodeVmKernel(k.name,k.threads,k.tg,k.bytes,gM3Runtime.arena,0,
        gM3Graph.ring,gDummyFor(be),gM3Graph.table,0)
      var noTrace:SessionHandle
      encodeStack(m,pr,1,noTrace,false)
      encode(pr.finalNorm,gM3Runtime,gM3Runtime.constants)
      let lin=gLinCache["outProj.weight"]
      encodeLin(lin,gM3Runtime.arena,a.norm,gM3Graph.logits,0)
      if not sessionGraphEnd(be):
        # Nothing was executed (the commands were only recorded): close the session untouched.
        discard sessionEnd(be);opened=false
        gM3Graph.failed=true;return false
      gM3Graph.valid=true
      recorded=true
    require(sessionGraphRun(be,0,1,0,0),"graph replay")
    tB=getMonoTime().ticks
    let ok=sessionEnd(be);opened=false
    tC=getMonoTime().ticks
    # The call that records the graph is setup, not decode speed: it encodes the whole stack, builds
    # the indirect-command pipelines, and its first replay waits for the driver to finish them
    # (about 2.3 s on Intel Iris Plus 640). The token it computes costs ~20 ms by comparison.
    if recorded:gM3Runtime.kernelSetupSeconds+=float64(tC-tA)/1e9
    require(ok,"sessionEnd: " & sessionGetLastError(be))
    logits=sessionRead(be,gM3Graph.logits,m.vocab)
    result=true
    if prof:m3ProfileLine(if recorded:"graph+record" else:"graph",tA,tB,tC,getMonoTime().ticks)
  except CatchableError:
    if opened:
      discard sessionEnd(be)
      discard sessionRecoverQueue(be)
    gM3Runtime.poisoned=true
    raise

proc m3PrefillPaced(be:GpuBackend):bool =
  ## Command buffers are bounded by measured time where the OS limits how long one may run.
  sessionPacesCommandBuffers(be) and m3PrefillCbMs()>0

proc m3PrefillSplit(be:GpuBackend;T:int;produceLogits,trace,useOd:bool):bool =
  ## A prefill micro-batch (several rows, no logits) goes through the scheduler unless both of its
  ## controls are off, which keeps the plain one-command-buffer path.
  T>1 and not produceLogits and not trace and not useOd and
    (m3PrefillPaced(be) or m3PrefillLayersPerCb()>0)

proc runPrefillStack(m:var NimformerModelJamba;pr:RuntimePrograms;T:int;hInput:SessionHandle) =
  ## All blocks for the T rows of one micro-batch, starting from the embeddings in hInput.
  let be=gM3Runtime.be;let a=gM3Runtime.a;let layers=m.blocks.len
  let paced=m3PrefillPaced(be);let cbMs=m3PrefillCbMs()
  let commitEvery=if paced:0 else:m3PrefillLayersPerCb()
  if not gM3PrefillLogged:
    gM3PrefillLogged=true
    stderr.writeLine "[prefill] micro-batches of up to " & $a.capacity & " rows, " &
      (if paced:"command buffers of ~" & $int(cbMs) & " ms" else:"commit every " & $commitEvery & " layers")
  var noTrace:SessionHandle
  var opened=false
  var layer=0
  try:
    while layer<layers:
      var count=layers-layer
      if paced:
        # Unknown speed: one layer, then as many layers as fit the time budget.
        count=if gM3PrefillMsPerLayerRow<=0:1
              else:clamp(int(min(cbMs/(gM3PrefillMsPerLayerRow*float64(m3Tile(T))),float64(layers))),1,layers-layer)
      let t0=getMonoTime().ticks
      require(sessionBegin(be),"sessionBegin")
      opened=true
      if layer==0:require(sessionCopyInto(be,hInput,gM3Runtime.arena,0,a.x,T*a.hidden),"input upload")
      encodeStack(m,pr,T,noTrace,false,layer,layer+count,commitEvery)
      let ok=sessionEnd(be);opened=false
      require(ok,"sessionEnd: " & sessionGetLastError(be))
      if existsEnv("BYBY_PREFILL_LOG") or existsEnv("BYBY_SESSION_LOG"):
        stderr.writeLine "[prefill] rows " & $T & ", layers " & $count & ", encode+gpu " &
          formatFloat(float64(getMonoTime().ticks-t0)/1e6,ffDecimal,2) & " ms"
      if paced:
        let per=float64(getMonoTime().ticks-t0)/1e6/float64(count*m3Tile(T))
        # Follow a faster measurement at once (the first command buffer also pays warm-up costs),
        # a slower one gradually.
        gM3PrefillMsPerLayerRow=if gM3PrefillMsPerLayerRow<=0 or per<gM3PrefillMsPerLayerRow:per
                                else:0.5*gM3PrefillMsPerLayerRow+0.5*per
      layer+=count
  except CatchableError:
    if opened:
      discard sessionEnd(be)
      discard sessionRecoverQueue(be)
    gM3Runtime.poisoned=true
    raise

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
  var useOd=T==1 and produceLogits and odMode!="0" and not gM3Runtime.skipAttn and not trace
  require(not useOd or gM3ProgramProvider.isNil,"program provider requires the resident micro-kernel path (BYBY_ONE_DISPATCH=0)")
  if useOd and not gM3Runtime.od.ready:
    let reason=eligible(m,ctx)
    if reason.len>0:
      if odMode=="1":raise newException(ValueError,"Mamba-3 one-dispatch: " & reason)
      useOd=false
    else:
      gM3Runtime.od=prepareOneDispatch(m,ctx,a,gM3Runtime.states,gM3Runtime.params,gM3Runtime.paramData)
  gM3Runtime.kernelSetupSeconds+=float64(getMonoTime().ticks-setupStart)/1e9
  # Greedy-friendly decode through the recorded graph (DeepSeek-style): the embedding gather and the
  # whole stack replay without the CPU encoding them again.
  if T==1 and produceLogits and not trace and not useOd and gM3ProgramProvider.isNil and
     not gM3Runtime.skipAttn and m3GraphWanted(be):
    var gl:seq[float32]
    if m3GraphStep(m,ids[0],pr,gl):
      gM3Runtime.tokens+=1;inc gM3Runtime.decodeCalls
      return Tensor(shape: @[1,1,m.vocab],data:gl)
  # Embedding lookup is the only host activation; the entire stack and state
  # remain resident after this bounded input upload, as in Codestral.
  let input=m.embed.lookupBatch(@[ids],ctx)
  var hInput=sessionUpload(be,input.data)
  var hTrace:SessionHandle
  if trace:hTrace=sessionUpload(be,newSeq[float32](4*m.blocks.len*a.hidden))
  var opened=false
  let prof=T==1 and produceLogits and not trace and m3ProfileOn()
  var tA,tB,tC:int64
  defer:
    if trace:sessionFree(be,hTrace)
    if opened:
      discard sessionEnd(be)
      discard sessionRecoverQueue(be)
    sessionFree(be,hInput)
  try:
    if m3PrefillSplit(be,T,produceLogits,trace,useOd):
      runPrefillStack(m,pr,T,hInput)
      gM3Runtime.tokens+=T;inc gM3Runtime.prefillCalls
      return newTensor(@[1,1,0])
    tA=getMonoTime().ticks
    require(sessionBegin(be),"sessionBegin")
    opened=true
    require(sessionCopyInto(be,hInput,gM3Runtime.arena,0,a.x,T*a.hidden),"input upload")
    if useOd:
      encodeOneDispatch(gM3Runtime.od,gM3Runtime.arena)
    else:
      encodeStack(m,pr,T,hTrace,trace)
    if produceLogits:
      encode(pr.finalNorm,gM3Runtime,gM3Runtime.constants)
      encodeLinear(gM3Runtime,pr,"outProj.weight",a.norm+(T-1)*a.hidden,a.logits,1)
    if produceLogits:
      require(sessionCopyInto(be,gM3Runtime.arena,gM3Runtime.readback,a.logits,0,m.vocab),"logits copy")
      if useOd:require(sessionCopyInto(be,gM3Runtime.arena,gM3Runtime.readback,0,m.vocab,4),"OD status copy")
    tB=getMonoTime().ticks
    let ok=sessionEnd(be);opened=false
    tC=getMonoTime().ticks
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
      if prof:m3ProfileLine("eager",tA,tB,tC,getMonoTime().ticks)
    else:result=newTensor(@[1,1,0])
  except CatchableError:
    gM3Runtime.poisoned=true
    raise

# Greedy decode on the GPU (the DeepSeek decode pipeline, as micro kernels for Metal and Vulkan):
# a step gathers its embedding row from a token ring on the GPU, runs the stack, the final norm and
# lm_head, masks the banned tokens and writes the argmax into the next ring slot. Several steps are
# encoded back to back (committed without waiting in between), so the GPU never waits for the host
# between tokens and only the chosen ids are read back, never the logits.
const M3PipeRing = 128
const M3BannedLogit = -3.0e38'f32   ## below every finite logit; the argmax never prefers it
type M3Pipe = object
  ready:bool
  table,ring,banned,seed:SessionHandle
  nBanned,groups,tg:int
  tableF16:bool
  gather,ban,pick1,pick2:Kernel

var gM3Pipe:M3Pipe

proc releaseM3Pipe*() =
  if gM3Pipe.ready:
    let be=gM3Runtime.be
    sessionFree(be,gM3Pipe.table);sessionFree(be,gM3Pipe.ring)
    sessionFree(be,gM3Pipe.banned);sessionFree(be,gM3Pipe.seed)
  gM3Pipe=default(M3Pipe)

proc pipeGatherProgram(a:M3Arena;vocab,tg:int;f16:bool):VmProgram =
  ## x[g] = table[ring[slot] * hidden + g]; the ring slot is the input binding's offset.
  var p:VmProgram
  p.emitUTgid(0);p.emitULid(1);p.emitUMulK(2,0,uint32(tg));p.emitUAdd(2,2,1)
  p.emitUConst(3,0);p.emitULoadIn(1,3,0)
  p.emitRToU(4,1);p.emitUMulK(4,4,uint32(a.hidden));p.emitUAdd(4,4,2)
  if f16:p.emitULoadF16(2,4,0) else:p.emitULoadConstF(2,4,0)
  p.emitUStoreF32(2,2,uint32(a.x));p.emitHalt();p

proc pipeBanProgram(a:M3Arena;n,tg:int):VmProgram =
  ## logits[banned[i]] = M3BannedLogit for i < n.
  var p:VmProgram
  p.emitUTgid(0);p.emitULid(1);p.emitUMulK(2,0,uint32(tg));p.emitUAdd(2,2,1)
  p.emitUToR(1,2);p.emitLoadConstF32(2,float32(n));p.emitCmpLt(3,1,2)
  sessionVmIfGpu(p,3):
    p.emitULoadConstF(4,2,0);p.emitRToU(4,4)
    p.emitLoadConstF32(5,M3BannedLogit);p.emitUStoreF32(5,4,uint32(a.logits))
  p.emitHalt();p

proc emitPickReduce(p:var VmProgram;tg:int) =
  ## Threadgroup argmax of (R3 value, R4 index) pairs, first index on ties; R1 = float(lid).
  ## Leaves the winner in tg[0] (value) and tg[tg] (index).
  p.emitTgStoreF32(3,1,0);p.emitTgStoreF32(4,1,uint32(tg));p.emitBarrierTg()
  p.emitLoadConstF32(5,float32(fastLog2(tg)));p.emitLoadConstF32(6,float32(tg div 2))
  sessionVmForGpu(p,5):
    p.emitCmpLt(7,1,6)
    sessionVmIfGpu(p,7):
      p.emitAdd(8,1,6)
      p.emitTgLoadF32(9,1,0);p.emitTgLoadF32(10,1,uint32(tg))
      p.emitTgLoadF32(11,8,0);p.emitTgLoadF32(12,8,uint32(tg))
      p.emitCmpLt(13,9,11)                       # mine < other
      p.emitCmpLt(14,11,9)                       # other < mine
      p.emitCmpLt(15,12,10)                      # other index < mine
      p.emitLoadConstF32(16,1);p.emitSub(16,16,13);p.emitSub(16,16,14)   # equal values
      p.emitMul(16,16,15);p.emitAdd(16,16,13)    # take the other one
      sessionVmIfGpu(p,16):
        p.emitTgStoreF32(11,1,0);p.emitTgStoreF32(12,1,uint32(tg))
    p.emitBarrierTg();p.emitMulConst(6,6,0.5'f32)

proc pipePick1Program(a:M3Arena;vocab,groups,tg:int):VmProgram =
  ## Pass 1: threadgroup g keeps the largest finite logit over indices g*tg+lid + k*groups*tg and
  ## writes (value, index) to arena[a.mlp + 2g]. NaN and infinite logits are never picked.
  var p:VmProgram
  let stride=groups*tg
  p.emitUTgid(0);p.emitULid(1);p.emitUMulK(2,0,uint32(tg));p.emitUAdd(2,2,1)
  p.emitUToR(1,1);p.emitUToR(2,2)
  p.emitLoadConstF32(3,-3.3e38'f32);p.emitLoadConstF32(4,-1)
  p.emitLoadConstF32(17,float32((vocab+stride-1) div stride))
  p.emitLoadConstF32(18,float32(vocab));p.emitLoadConstF32(19,3.0e38'f32)
  sessionVmForGpu(p,17):
    p.emitCmpLt(8,2,18)
    sessionVmIfGpu(p,8):
      p.emitRToU(3,2);p.emitULoadF32(9,3,uint32(a.logits))
      p.emitCmpLt(10,9,19);p.emitCmpLt(11,3,9);p.emitMul(10,10,11)
      sessionVmIfGpu(p,10):
        p.emitAddConst(3,9,0);p.emitAddConst(4,2,0)
    p.emitAddConst(2,2,float32(stride))
  p.emitPickReduce(tg)
  p.emitLoadConstF32(7,1);p.emitCmpLt(8,1,7)
  sessionVmIfGpu(p,8):
    p.emitLoadConstF32(9,0);p.emitTgLoadF32(10,9,0);p.emitTgLoadF32(11,9,uint32(tg))
    p.emitUMulK(5,0,2);p.emitUStoreF32(10,5,uint32(a.mlp));p.emitUStoreF32(11,5,uint32(a.mlp+1))
  p.emitHalt();p

proc pipePick2Program(groups:int):VmProgram =
  ## Pass 2 (one threadgroup of `groups` lanes): reduces the pass-1 pairs read through the input
  ## binding and stores the winning index at the output binding (the next ring slot).
  var p:VmProgram
  p.emitULid(1);p.emitUMulK(2,1,2);p.emitUToR(1,1)
  p.emitULoadIn(3,2,0);p.emitULoadIn(4,2,1)
  p.emitPickReduce(groups)
  p.emitLoadConstF32(7,1);p.emitCmpLt(8,1,7)
  sessionVmIfGpu(p,8):
    p.emitLoadConstF32(9,0);p.emitTgLoadF32(11,9,uint32(groups))
    p.emitUConst(3,0);p.emitUStoreF32(11,3,0)
  p.emitHalt();p

proc m3GpuSampleOn*():bool =
  ## BYBY_M3_GPU_SAMPLE=0 keeps the host sampler for greedy decoding (the reference path).
  getEnv("BYBY_M3_GPU_SAMPLE","1")!="0"

proc m3DecodeAhead():int =
  ## BYBY_DECODE_AHEAD: greedy decode steps encoded per command-buffer batch (default 16). The host
  ## reads the chosen ids once per batch; a stop token inside a batch wastes the steps after it.
  clamp(m3EnvInt("BYBY_DECODE_AHEAD",16),1,M3PipeRing div 2-2)

proc m3DecodeAsync(be:GpuBackend):bool =
  ## Metal keeps two decode batches in flight (the next one is encoded and committed while the
  ## GPU runs the previous one); BYBY_M3_DECODE_ASYNC=0 waits for each batch instead.
  be==gbMetal and getEnv("BYBY_M3_DECODE_ASYNC","1")!="0"

proc ringToken(be:GpuBackend;slot:int):int =
  ## The id the sampler stored (as a float) in a ring slot, read straight from the shared buffer.
  int(cast[float32](int32(sessionTokenRingGet(be,gM3Pipe.ring,slot))))

proc prepareM3Pipe(m:var NimformerModelJamba;banned:seq[int]) =
  if gM3Pipe.ready and gM3Pipe.nBanned==banned.len:return
  releaseM3Pipe()
  let be=gM3Runtime.be;let a=gM3Runtime.a
  let t0=getMonoTime().ticks
  var pipe:M3Pipe
  let n=m.vocab*a.hidden
  require(m.embed.weight.floatValues.len>=n,"embedding table missing on the host")
  # An FP32 table up to 512 MB (exactly the host embedding); larger tables go up as FP16.
  pipe.tableF16=n*4>512*1024*1024
  if pipe.tableF16:
    let bits=toF16Bits(m.embed.weight.floatValues.toOpenArray(0,n-1))
    pipe.table=sessionUploadBytesShared(be,cast[ptr UncheckedArray[uint8]](unsafeAddr bits[0]).toOpenArray(0,bits.len*2-1))
  else:
    pipe.table=sessionUploadFloatsShared(be,m.embed.weight.floatValues.toOpenArray(0,n-1))
  pipe.ring=sessionAllocScratch(be,M3PipeRing)
  pipe.seed=sessionAllocScratch(be,1)
  var ids=newSeq[float32](max(1,banned.len))
  for i,b in banned:ids[i]=float32(b)
  pipe.banned=sessionUpload(be,ids)
  pipe.nBanned=banned.len
  var tg=0
  for cand in [256,128,64,32]:
    if a.hidden mod cand==0:
      tg=cand;break
  require(tg>0,"hidden size has no power-of-two threadgroup")
  pipe.gather=compile(pipeGatherProgram(a,m.vocab,tg,pipe.tableF16),
    "m3_pipe_gather_" & $a.hidden & (if pipe.tableF16:"_f16" else:""),a.hidden,tg,16,be)
  let banThreads=max(64,(banned.len+63) div 64*64)
  pipe.ban=compile(pipeBanProgram(a,banned.len,64),"m3_pipe_ban_" & $banned.len,banThreads,64,16,be)
  pipe.groups=64;pipe.tg=256
  pipe.pick1=compile(pipePick1Program(a,m.vocab,pipe.groups,pipe.tg),"m3_pipe_pick1_" & $m.vocab,
    pipe.groups*pipe.tg,pipe.tg,2*pipe.tg*4,be)
  pipe.pick2=compile(pipePick2Program(pipe.groups),"m3_pipe_pick2_" & $pipe.groups,
    pipe.groups,pipe.groups,2*pipe.groups*4,be)
  prewarmVulkanJit(be)
  pipe.ready=true
  gM3Pipe=pipe
  gM3Runtime.kernelSetupSeconds+=float64(getMonoTime().ticks-t0)/1e9

proc encodePipeStep(m:NimformerModelJamba;pr:RuntimePrograms;slotIn,slotOut:int;seeded:bool) =
  let be=gM3Runtime.be;let a=gM3Runtime.a
  if seeded:require(sessionCopyInto(be,gM3Pipe.seed,gM3Pipe.ring,0,slotIn,1),"seed token")
  let g=gM3Pipe.gather
  if gM3Pipe.tableF16:
    encodeVmKernel(g.name,g.threads,g.tg,g.bytes,gM3Runtime.arena,0,gM3Pipe.table,gDummyFor(be),gM3Pipe.ring,slotIn)
  else:
    encodeVmKernel(g.name,g.threads,g.tg,g.bytes,gM3Runtime.arena,0,gDummyFor(be),gM3Pipe.table,gM3Pipe.ring,slotIn)
  var noTrace:SessionHandle
  encodeStack(m,pr,1,noTrace,false)
  encode(pr.finalNorm,gM3Runtime,gM3Runtime.constants)
  encodeLinear(gM3Runtime,pr,"outProj.weight",a.norm,a.logits,1)
  if gM3Pipe.nBanned>0:
    let k=gM3Pipe.ban
    encodeVmKernel(k.name,k.threads,k.tg,k.bytes,gM3Runtime.arena,0,gDummyFor(be),gM3Pipe.banned,gDummyFor(be),0)
  let k1=gM3Pipe.pick1
  encodeVmKernel(k1.name,k1.threads,k1.tg,k1.bytes,gM3Runtime.arena,0,gDummyFor(be),gDummyFor(be),gDummyFor(be),0)
  let k2=gM3Pipe.pick2
  encodeVmKernel(k2.name,k2.threads,k2.tg,k2.bytes,gM3Pipe.ring,slotOut,gDummyFor(be),gDummyFor(be),gM3Runtime.arena,a.mlp)

proc m3GreedyDecodeAsync(m:var NimformerModelJamba;pr:RuntimePrograms;maxNew,ahead:int;
    onToken:proc(id:int):bool):int =
  ## m3GreedyDecode with two command-buffer batches in flight (Metal): batch b+1 is encoded and
  ## committed before the host waits for batch b, so the GPU never idles while the host reads
  ## ids and encodes. The ring already holds the seed token in slot 0.
  let be=gM3Runtime.be
  var step=0          # steps encoded
  var queued=0        # steps committed (encoded, may still run)
  var batches:seq[int]
  var stopped=false
  template enqueue(kArg:int) =
    let k=kArg
    var opened=false
    try:
      require(sessionBegin(be),"sessionBegin");opened=true
      for j in 0..<k:
        if j>0:require(sessionCommitContinue(be),"commit continue")
        encodePipeStep(m,pr,step mod M3PipeRing,(step+1) mod M3PipeRing,seeded=step==0)
        inc step
      let ok=sessionEndAsync(be);opened=false
      require(ok,"sessionEndAsync")
    except CatchableError:
      if opened:discard sessionEnd(be)
      raise
    batches.add k;queued+=k
  try:
    var next=1
    while batches.len<2 and queued<maxNew:
      enqueue(min(next,maxNew-queued));next=ahead
    while batches.len>0:
      let k=batches[0]
      batches.delete(0)
      require(sessionWaitOldest(be),"decode batch failed: " & sessionGetLastError(be))
      gM3Runtime.tokens+=k;gM3Runtime.decodeCalls+=k
      if m3ProfileOn():
        let g=sessionGpuSeconds(be)
        echo "[m3-profile] batch ",k," gpu ",formatFloat((g-gM3GpuSecs)*1000/float(k),ffDecimal,2)," ms/token"
        gM3GpuSecs=g
      let first=result
      for j in 0..<k:
        let id=ringToken(be,(first+j+1) mod M3PipeRing)
        require(id>=0 and id<m.vocab,"GPU sampler returned token " & $id)
        if not onToken(id):
          stopped=true;break
        inc result
      if stopped:break
      if queued<maxNew:enqueue(min(ahead,maxNew-queued))
  except CatchableError:
    while batches.len>0:
      discard sessionWaitOldest(be);batches.delete(0)
    discard sessionRecoverQueue(be)
    gM3Runtime.poisoned=true
    raise
  # Steps encoded past a stop still run: let them finish before anything else uses the queue.
  while batches.len>0:
    discard sessionWaitOldest(be);batches.delete(0)

proc m3GreedyDecode*(m:var NimformerModelJamba;firstToken,maxNew:int;banned:seq[int];
    onToken:proc(id:int):bool):int =
  ## Greedy decode of up to maxNew tokens after firstToken (the last prompt token, not fed yet),
  ## chosen on the GPU. onToken gets each id in order and returns false to stop (the id it
  ## returned false for is not counted). The first batch is one step, so the first token comes
  ## back as soon as it exists; later batches are BYBY_DECODE_AHEAD steps. Steps encoded past a
  ## stop are computed and discarded, so the recurrent state is then ahead of the delivered text:
  ## this is for one-shot generation, not for a conversation that continues from the state.
  require(gM3Runtime.ready and not gM3Runtime.poisoned,"state unavailable; reset after a failed command buffer")
  require(firstToken>=0 and firstToken<m.vocab,"token outside vocabulary")
  let setupStart=getMonoTime().ticks
  preparePrograms(m,1)
  gM3Runtime.kernelSetupSeconds+=float64(getMonoTime().ticks-setupStart)/1e9
  prepareM3Pipe(m,banned)
  let be=gM3Runtime.be
  let pr=gM3Runtime.programs[1]
  let ahead=m3DecodeAhead()
  block seed:
    var v=sessionUpload(be,@[float32(firstToken)])
    var opened=false
    try:
      require(sessionBegin(be),"sessionBegin");opened=true
      require(sessionCopyInto(be,v,gM3Pipe.seed,0,0,1),"seed upload")
      let ok=sessionEnd(be);opened=false;require(ok,"seed sessionEnd")
    finally:
      if opened:discard sessionEnd(be)
      sessionFree(be,v)
  if m3DecodeAsync(be):
    return m3GreedyDecodeAsync(m,pr,maxNew,ahead,onToken)
  var step=0
  var batch=1
  var stopped=false
  while result<maxNew and not stopped:
    let k=min(batch,maxNew-result)
    var opened=false
    try:
      require(sessionBegin(be),"sessionBegin");opened=true
      for j in 0..<k:
        if j>0:require(sessionCommitContinue(be),"commit continue")
        encodePipeStep(m,pr,step mod M3PipeRing,(step+1) mod M3PipeRing,seeded=step==0)
        inc step
      let ok=sessionEnd(be);opened=false
      require(ok,"sessionEnd: " & sessionGetLastError(be))
    except CatchableError:
      if opened:
        discard sessionEnd(be)
        discard sessionRecoverQueue(be)
      gM3Runtime.poisoned=true
      raise
    gM3Runtime.tokens+=k;gM3Runtime.decodeCalls+=k
    if m3ProfileOn():
      let g=sessionGpuSeconds(be)
      echo "[m3-profile] batch ",k," gpu ",formatFloat((g-gM3GpuSecs)*1000/float(k),ffDecimal,2)," ms/token"
      gM3GpuSecs=g
    let ring=sessionRead(be,gM3Pipe.ring,M3PipeRing)
    for j in 0..<k:
      let id=int(ring[(step-k+j+1) mod M3PipeRing])
      require(id>=0 and id<m.vocab,"GPU sampler returned token " & $id)
      if not onToken(id):
        stopped=true;break
      inc result
    batch=ahead

proc m3PrefillTop():int =
  ## Rows of the largest prefill micro-batch: BYBY_UBATCH, at most the arena capacity.
  let cap=gM3Runtime.a.capacity
  let ubatch=m3UbatchRows()
  if ubatch>0:min(ubatch,cap) else:cap

proc m3PrefillBuckets*(top:int):seq[int] =
  ## Chunk lengths of a bucketed prefill, largest first: the micro-batch size, then a quarter of it, and so
  ## on down to 8 rows. The kernels that scan a chunk are specialized per chunk length, so a conversation
  ## that only ever feeds these lengths compiles each set once, which a warm-up can do before the first prompt.
  var s=max(top,1)
  result.add s
  while s>=32:
    s=(s div 4+7) div 8*8
    result.add s
  if result[^1]>8:result.add 8

proc prefillM3Resident*(m:var NimformerModelJamba;ids:seq[int];ctx:Backend)=
  ## The prompt in micro-batches (BYBY_UBATCH rows, at most the arena capacity). Where command buffers
  ## are paced, a micro-batch shrinks (halving, never below 8 rows) until one layer of it fits the time
  ## budget, so a slow device or a big model never needs a command buffer longer than that.
  let top=m3PrefillTop()
  let paced=m3PrefillPaced(gM3Runtime.be)
  let cbMs=m3PrefillCbMs()
  var offset=0
  while offset<ids.len:
    var rows=top
    if paced and gM3PrefillMsPerLayerRow>0:
      while rows>8 and gM3PrefillMsPerLayerRow*float64(m3Tile(rows))>cbMs:rows=max(8,rows div 2)
    let count=min(ids.len-offset,rows)
    discard runM3Resident(m,ids[offset..<offset+count],ctx,produceLogits=false)
    offset+=count

proc prefillM3Bucketed*(m:var NimformerModelJamba;ids:seq[int];ctx:Backend):int =
  ## Like prefillM3Resident, but every chunk has a length from m3PrefillBuckets, so no kernel set is
  ## compiled once those lengths are warm (warmM3Resident). Returns how many leading ids it fed; fewer
  ## than the smallest bucket are left for the caller to feed as decode steps.
  let buckets=m3PrefillBuckets(m3PrefillTop())
  let paced=m3PrefillPaced(gM3Runtime.be)
  let cbMs=m3PrefillCbMs()
  while ids.len-result>=buckets[^1]:
    var rows=buckets[^1]
    for b in buckets:
      if b>ids.len-result:continue
      # Paced devices skip a bucket whose single layer would not fit the command buffer budget.
      if paced and b>buckets[^1] and gM3PrefillMsPerLayerRow>0 and
         gM3PrefillMsPerLayerRow*float64(m3Tile(b))>cbMs:continue
      rows=b
      break
    discard runM3Resident(m,ids[result..<result+rows],ctx,produceLogits=false)
    result+=rows

proc warmM3Resident*(m:var NimformerModelJamba;ctx:Backend;prefill=true):seq[int] =
  ## Before the first prompt of a conversation: runs every bucketed prefill length (unless prefill is
  ## false) and the decode step once, so their kernel sets are compiled and the prefill speed is
  ## measured. Returns the chunk lengths it ran. The recurrent state is dirty afterwards, so the caller
  ## resets it.
  if prefill:
    for rows in m3PrefillBuckets(m3PrefillTop()):
      discard runM3Resident(m,newSeq[int](rows),ctx,produceLogits=false)
      result.add rows
  for _ in 0..<3:
    discard runM3Resident(m,@[0],ctx)

proc invalidateM3OneDispatch*() =
  if gM3Runtime.ready:release(gM3Runtime.od,gM3Runtime.be)
