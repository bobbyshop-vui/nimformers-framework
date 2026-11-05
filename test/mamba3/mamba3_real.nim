## Mamba-3 SISO. Official projection order: z,x,B,C,dt,A,trap,angles.
## No causal convolution, A_log, or output RMSNorm in the 187M checkpoint.
## Compatibility names below preserve the copied Codestral LoRA/checkpoint ABI.
import std/[math, tables]
import nimformer, backend, mamba2_ssm_gpu
import session_vm, gpu_layers, m3_inference_kernels, m3_training_kernels

type
  Mamba3Real* = object
    hiddenSize*, intermediateSize*, numHeads*, headDim*, nGroups*, dState*, convKernel*: int
    inProj*, outProj*, mlpFc1*, mlpFc2*: Linear
    mlpNorm*: Tensor
    bBias*, cBias*, bNorm*, cNorm*, dSkip*, dtBias*: Tensor
    normEps*: float32
  Mamba3State* = object
    ssmState*: seq[float32]
    aux*: seq[float32] # angles[H,R], previous K[H,N], previous V[C]
  Mamba3Cache* = object
    B*, T*: int
    proj*, inner*: Tensor
    k*, q*, angles*, coeff*, states*: seq[float32]
    terminal*: Mamba3State

proc sigmoid(x: float32): float32 =
  if x >= 0: 1'f32 / (1'f32 + exp(-x))
  else:
    let e = exp(x)
    e / (1'f32 + e)
proc softplus(x: float32): float32 = max(x, 0'f32) + ln(1'f32 + exp(-abs(x)))
proc ropeCount(m: Mamba3Real): int = m.dState div 4
proc projSize(m: Mamba3Real): int = 2*m.intermediateSize + 2*m.nGroups*m.dState + 3*m.numHeads + m.ropeCount

proc newMamba3Real*(hiddenSize, intermediateSize, numHeads, dState, nGroups: int;
    convKernel = 0; normEps = 1e-5'f32; initializeWeights = true): Mamba3Real =
  if nGroups != 1 or dState mod 4 != 0 or intermediateSize mod numHeads != 0:
    raise newException(ValueError, "Mamba-3 SISO requires ngroups=1, d_state divisible by 4 and integral head dimension")
  result.hiddenSize=hiddenSize; result.intermediateSize=intermediateSize
  result.numHeads=numHeads; result.headDim=intermediateSize div numHeads
  result.dState=dState; result.nGroups=nGroups; result.normEps=normEps
  result.inProj=newLinear(hiddenSize, result.projSize, initializeWeights)
  result.outProj=newLinear(intermediateSize, hiddenSize, initializeWeights)
  result.mlpFc1=newLinear(hiddenSize,4*hiddenSize,initializeWeights)
  result.mlpFc2=newLinear(2*hiddenSize,hiddenSize,initializeWeights)
  result.mlpNorm=newTensor(@[hiddenSize],1'f32)
  result.bBias=newTensor(@[numHeads, 1, dState], 1'f32)
  result.cBias=newTensor(@[numHeads, 1, dState], 1'f32)
  result.bNorm=newTensor(@[dState], 1'f32); result.cNorm=newTensor(@[dState], 1'f32)
  result.dtBias=newTensor(@[numHeads], -4'f32); result.dSkip=newTensor(@[numHeads], 1'f32)

proc newMamba3State*(m: Mamba3Real): Mamba3State =
  result.ssmState=newSeq[float32](m.intermediateSize*m.dState)
  result.aux=newSeq[float32](m.numHeads*(m.ropeCount+m.dState)+m.intermediateSize)

proc inProjSsmRowRange*(intermediateSize, nGroups, dState, numHeads: int): (int,int) =
  (intermediateSize, 2*intermediateSize+2*nGroups*dState+3*numHeads+dState div 4)
proc inProjSsmRowRange*(m: Mamba3Real): (int,int) =
  inProjSsmRowRange(m.intermediateSize,m.nGroups,m.dState,m.numHeads)

proc paramListMamba3Real*(m: var Mamba3Real; prefix: string): seq[(string, ptr Tensor)] =
  result = @[(prefix & ".inProj.weight", addr m.inProj.weight),
    (prefix & ".inProj.bias", addr m.inProj.bias),
    (prefix & ".outProj.weight", addr m.outProj.weight),
    (prefix & ".outProj.bias", addr m.outProj.bias),
    (prefix & ".bBias", addr m.bBias), (prefix & ".cBias", addr m.cBias),
    (prefix & ".bNorm", addr m.bNorm), (prefix & ".cNorm", addr m.cNorm),
    (prefix & ".dSkip", addr m.dSkip), (prefix & ".dtBias", addr m.dtBias),
    (prefix & ".mlpFc1.weight", addr m.mlpFc1.weight),
    (prefix & ".mlpFc1.bias", addr m.mlpFc1.bias),
    (prefix & ".mlpFc2.weight", addr m.mlpFc2.weight),
    (prefix & ".mlpFc2.bias", addr m.mlpFc2.bias),
    (prefix & ".mlpNorm", addr m.mlpNorm)]

proc hfNameMapMamba3*(layerIdx: int): Table[string,string] =
  let p="backbone.layers." & $layerIdx & "."
  for (local, hf) in [("inProj.weight","in_proj.weight"),("outProj.weight","out_proj.weight"),
      ("bBias","B_bias"),("cBias","C_bias"),("bNorm","B_norm.weight"),
      ("cNorm","C_norm.weight"),("dtBias","dt_bias"),("dSkip","D")]:
    result[local]=p & "mixer." & hf
  result["ln1Weight"]=p & "norm.weight"
  result["mlpNorm"]=p & "norm2.weight"
  result["mlpFc1.weight"]=p & "mlp.fc1.weight"
  result["mlpFc2.weight"]=p & "mlp.fc2.weight"

proc prepare(m: Mamba3Real; proj: Tensor; cache: var Mamba3Cache; initial: Mamba3State) =
  let C=m.intermediateSize; let N=m.dState; let H=m.numHeads; let R=m.ropeCount
  let P=m.projSize; let BT=cache.B*cache.T
  cache.k=newSeq[float32](BT*H*N); cache.q=newSeq[float32](BT*H*N)
  cache.angles=newSeq[float32](BT*H*R); cache.coeff=newSeq[float32](BT*H*3)
  var phase=newSeq[float32](cache.B*H*R)
  if initial.aux.len>0:
    for i in 0..<H*R: phase[i]=initial.aux[i]
  for row in 0..<BT:
    let b=row div cache.T
    let off=row*P
    var bn,cn: seq[float32]
    bn=newSeq[float32](N); cn=newSeq[float32](N)
    var bs,cs=0'f32
    for n in 0..<N:
      bs+=proj.data[off+2*C+n]*proj.data[off+2*C+n]
      cs+=proj.data[off+2*C+N+n]*proj.data[off+2*C+N+n]
    let bi=1'f32/sqrt(bs/float32(N)+m.normEps)
    let ci=1'f32/sqrt(cs/float32(N)+m.normEps)
    for n in 0..<N:
      bn[n]=proj.data[off+2*C+n]*bi*m.bNorm.data[n]
      cn[n]=proj.data[off+2*C+N+n]*ci*m.cNorm.data[n]
    for h in 0..<H:
      let dt=softplus(proj.data[off+2*C+2*N+h]+m.dtBias.data[h])
      let ap=proj.data[off+2*C+2*N+H+h]
      let a= -max(1e-4'f32, (if ap>=0: 1'f32+ap else: 1'f32/(1'f32-ap)))
      let trap=sigmoid(proj.data[off+2*C+2*N+2*H+h])
      let co=(row*H+h)*3
      let decay=exp(a*dt)
      cache.coeff[co]=decay; cache.coeff[co+1]=(1'f32-trap)*dt*decay
      cache.coeff[co+2]=trap*dt
      let base=(row*H+h)*N
      for n in 0..<N:
        cache.k[base+n]=bn[n]+m.bBias.data[h*N+n]
        cache.q[base+n]=cn[n]+m.cBias.data[h*N+n]
      for r in 0..<R:
        let pi=(b*H+h)*R+r
        phase[pi]+=tanh(proj.data[off+2*C+2*N+3*H+r])*dt*float32(PI)
        cache.angles[(row*H+h)*R+r]=phase[pi]
        let co=cos(phase[pi]); let si=sin(phase[pi])
        let j=base+2*r
        let k0=cache.k[j]; let k1=cache.k[j+1]
        let q0=cache.q[j]; let q1=cache.q[j+1]
        cache.k[j]=k0*co-k1*si; cache.k[j+1]=k0*si+k1*co
        cache.q[j]=q0*co-q1*si; cache.q[j+1]=q0*si+q1*co

proc scanHost(m: Mamba3Real; c: var Mamba3Cache; initial: Mamba3State) =
  let C=m.intermediateSize; let N=m.dState; let H=m.numHeads; let P=m.projSize
  for b in 0..<c.B:
    for t in 0..<c.T:
      let row=b*c.T+t
      for d in 0..<C:
        let h=d div m.headDim; let co=(row*H+h)*3
        let v=c.proj.data[row*P+C+d]
        let pv=if t>0: c.proj.data[(row-1)*P+C+d] else: initial.aux[H*(m.ropeCount+N)+d]
        var y=m.dSkip.data[h]*v
        for n in 0..<N:
          let i=((b*(c.T+1)+t)*C+d)*N+n
          let k=c.k[(row*H+h)*N+n]
          let pk=if t>0: c.k[((row-1)*H+h)*N+n] else: initial.aux[H*m.ropeCount+h*N+n]
          let s=c.coeff[co]*c.states[i]+c.coeff[co+1]*pk*pv+c.coeff[co+2]*k*v
          c.states[i+C*N]=s
          y+=c.q[(row*H+h)*N+n]*s
        let z=c.proj.data[row*P+d]
        c.inner.data[row*C+d]=y*z*sigmoid(z)

proc scanGpu(m: Mamba3Real; c: var Mamba3Cache; initial: Mamba3State; ctx: Backend) =
  ## One lane owns a value channel and scans time; state addresses use uint32.
  ## Projections reuse the resident FP4-KM/GEMM path from Codestral.
  let C=m.intermediateSize; let H=m.numHeads; let N=m.dState; let P=m.projSize
  var mem: seq[float32]
  template pack(a: seq[float32]): int =
    block:
      let offset=mem.len
      mem.add(a)
      offset
  let op=pack(c.proj.data); let ok=pack(c.k); let oq=pack(c.q)
  let oc=pack(c.coeff); let os=pack(c.states); let od=pack(m.dSkip.data)
  let oy=pack(c.inner.data)
  var prevK=newSeq[float32](c.k.len); var prevV=newSeq[float32](c.B*c.T*C)
  for b in 0..<c.B:
    for t in 0..<c.T:
      let row=b*c.T+t
      for j in 0..<H*N:
        prevK[row*H*N+j]=if t>0: c.k[(row-1)*H*N+j] else: initial.aux[H*m.ropeCount+j]
      for d in 0..<C:
        prevV[row*C+d]=if t>0: c.proj.data[(row-1)*P+C+d] else: initial.aux[H*(m.ropeCount+N)+d]
  let opk=pack(prevK); let opv=pack(prevV)
  var p: VmProgram
  p.emitUTgid(0) # gid = b*C+d, dispatch tg=1
  p.emitUToR(0,0); p.emitMulConst(1,0,1'f32/float32(C)); p.emitRToU(1,1)
  p.emitUMulK(2,1,uint32(C)); p.emitUToR(2,2); p.emitSub(2,0,2); p.emitRToU(2,2)
  p.emitMulConst(3,2,1'f32/float32(m.headDim)); p.emitRToU(3,3)
  p.emitUMulK(4,1,uint32(c.T*P)); p.emitUAdd(4,4,2) # z address
  p.emitUMulK(5,1,uint32(c.T*H*N)); p.emitUMulK(6,3,uint32(N)); p.emitUAdd(5,5,6)
  p.emitUMulK(6,1,uint32(c.T*H*3)); p.emitUMulK(7,3,3); p.emitUAdd(6,6,7)
  p.emitUMulK(7,1,uint32((c.T+1)*C*N)); p.emitUMulK(8,2,uint32(N)); p.emitUAdd(7,7,8)
  p.emitUMulK(13,1,uint32(c.T*C)); p.emitUAdd(13,13,2)
  p.emitLoadConstF32(0,float32(c.T))
  sessionVmForGpu(p,0):
    p.emitULoadF32(2,6,uint32(oc)); p.emitULoadF32(3,6,uint32(oc+1)); p.emitULoadF32(4,6,uint32(oc+2))
    p.emitULoadF32(5,4,uint32(op+C)); p.emitULoadF32(6,13,uint32(opv))
    p.emitULoadF32(7,3,uint32(od)); p.emitMul(7,7,5)
    p.emitUAddK(8,7,0); p.emitUAddK(9,5,0)
    p.emitLoadConstF32(1,float32(N))
    sessionVmForGpu(p,1):
      p.emitULoadF32(8,8,uint32(os)); p.emitULoadF32(9,9,uint32(ok))
      p.emitULoadF32(10,9,uint32(opk))
      p.emitMul(8,8,2); p.emitMul(10,10,6); p.emitFma(8,10,3,8)
      p.emitMul(9,9,5); p.emitFma(8,9,4,8)
      p.emitUStoreF32(8,8,uint32(os+C*N))
      p.emitULoadF32(12,9,uint32(oq)); p.emitFma(7,8,12,7)
      p.emitUAddK(8,8,1); p.emitUAddK(9,9,1)
    p.emitULoadF32(13,4,uint32(op)); p.emitSigmoid(14,13)
    p.emitMul(7,7,13); p.emitMul(7,7,14); p.emitUStoreF32(7,13,uint32(oy))
    p.emitUAddK(4,4,uint32(P)); p.emitUAddK(5,5,uint32(H*N))
    p.emitUAddK(6,6,uint32(H*3)); p.emitUAddK(7,7,uint32(C*N)); p.emitUAddK(13,13,uint32(C))
  p.emitHalt()
  let tag="@m3_siso_" & $c.B & "_" & $c.T & "_" & $C & "_" & $H & "_" & $N
  if not runMicroVmChain(ctx,[(p,c.B*C,4,1,tag)],mem):
    raise newException(IOError,"Mamba-3 SISO scan GPU dispatch failed")
  c.states=mem[os..<os+c.states.len]
  c.inner.data=mem[oy..<oy+c.inner.data.len]

proc executeTraining(ctx:Backend;programs:seq[tuple[p:VmProgram,threads,tg,bytes:int]];
    mem:var seq[float32];params:seq[float32];tag:string)=
  let be=ctx.kind.toByby()
  ensureDummy(be)
  var names:seq[string]
  for i,entry in programs:names.add(vmKernelForBe(entry.p,tag & "_" & $i,be))
  var h=sessionUpload(be,mem)
  var hc=sessionUpload(be,params)
  var opened=false
  defer:
    if opened:discard sessionEnd(be)
    sessionFree(be,h);sessionFree(be,hc)
  if not sessionBegin(be):raise newException(IOError,"Mamba-3 training sessionBegin")
  opened=true
  for i,entry in programs:
    encodeVmKernel(names[i],entry.threads,entry.tg,entry.bytes,h,0,gDummyFor(be),hc,h,0)
  let ok=sessionEnd(be);opened=false
  if not ok:raise newException(IOError,"Mamba-3 training kernel sessionEnd failed")
  mem=sessionRead(be,h,mem.len)

proc prepareGpu(m:Mamba3Real;proj:Tensor;c:var Mamba3Cache;initial:Mamba3State;ctx:Backend)=
  let a=arenaLayout(m.hiddenSize,m.intermediateSize,m.numHeads,m.dState,1,c.T,1)
  let s=stateLayout(a,a.total,0)
  var params:seq[float32]
  template pack(v:seq[float32]):int =
    block:
      let off=params.len
      params.add(v)
      off
  var w:M3Params
  w.bNorm=pack(m.bNorm.data);w.cNorm=pack(m.cNorm.data)
  w.bBias=pack(m.bBias.data);w.cBias=pack(m.cBias.data)
  w.dt=pack(m.dtBias.data);w.skip=pack(m.dSkip.data)
  c.k=newSeq[float32](c.B*c.T*a.H*a.N);c.q=newSeq[float32](c.k.len)
  c.angles=newSeq[float32](c.B*c.T*a.H*a.R);c.coeff=newSeq[float32](c.B*c.T*a.H*3)
  for b in 0..<c.B:
    var mem=newSeq[float32](s.total)
    mem.blit(a.proj,proj.data,b*c.T*a.P,c.T*a.P)
    mem.blit(s.phase,initial.aux,0,a.H*a.R)
    executeTraining(ctx,@[(prepareSiso(a,s,w,c.T,m.normEps),a.H*256,256,1024)],mem,params,
      "m3_train_prepare_" & $c.T & "_" & $a.C & "_" & $a.H & "_" & $a.N)
    c.k.blit(b*c.T*a.H*a.N,mem,a.k,c.T*a.H*a.N)
    c.q.blit(b*c.T*a.H*a.N,mem,a.q,c.T*a.H*a.N)
    c.angles.blit(b*c.T*a.H*a.R,mem,a.angles,c.T*a.H*a.R)
    c.coeff.blit(b*c.T*a.H*3,mem,a.coeff,c.T*a.H*3)

proc reversePrepareGpu(m:Mamba3Real;c:Mamba3Cache;dp:Tensor;dk,dq,dc:seq[float32];ctx:Backend):Tensor =
  let a=arenaLayout(m.hiddenSize,m.intermediateSize,m.numHeads,m.dState,1,c.T,1)
  var params=m.bNorm.data
  let cNorm=params.len;params.add(m.cNorm.data)
  let dtBias=params.len;params.add(m.dtBias.data)
  result=dp
  for b in 0..<c.B:
    var mem:seq[float32]
    template pack(v:seq[float32];off,count:int):int =
      block:
        let start=mem.len
        mem.add(v[off..<off+count])
        start
    var o:M3ReverseOffsets
    o.proj=pack(c.proj.data,b*c.T*a.P,c.T*a.P)
    o.k=pack(c.k,b*c.T*a.H*a.N,c.T*a.H*a.N)
    o.q=pack(c.q,b*c.T*a.H*a.N,c.T*a.H*a.N)
    o.coeff=pack(c.coeff,b*c.T*a.H*3,c.T*a.H*3)
    o.angles=pack(c.angles,b*c.T*a.H*a.R,c.T*a.H*a.R)
    o.dk=pack(dk,b*c.T*a.H*a.N,c.T*a.H*a.N)
    o.dq=pack(dq,b*c.T*a.H*a.N,c.T*a.H*a.N)
    o.dc=pack(dc,b*c.T*a.H*3,c.T*a.H*3)
    o.dp=pack(dp.data,b*c.T*a.P,c.T*a.P)
    o.ddt=mem.len;mem.setLen(mem.len+c.T*a.H)
    executeTraining(ctx,@[
      (rotateBackward(a,o,dtBias,c.T),a.H*256,256,1024),
      (finishDt(a,o,dtBias),c.T*a.H,256,16),
      (normBackward(a,o,0,0,m.normEps),c.T*256,256,1024),
      (normBackward(a,o,cNorm,1,m.normEps),c.T*256,256,1024)],
      mem,params,"m3_train_reverse_prepare_" & $c.T & "_" & $a.C & "_" & $a.H & "_" & $a.N)
    result.data.blit(b*c.T*a.P,mem,o.dp,c.T*a.P)

proc forwardCore(m: Mamba3Real; proj: Tensor; B,T: int; initial: Mamba3State; ctx: Backend): Mamba3Cache =
  result.B=B; result.T=T; result.proj=proj
  result.inner=newTensor(@[B,T,m.intermediateSize])
  result.states=newSeq[float32](B*(T+1)*m.intermediateSize*m.dState)
  for b in 0..<B:
    for i,v in initial.ssmState: result.states[b*(T+1)*initial.ssmState.len+i]=v
  if isGpuCtx(ctx):
    prepareGpu(m,proj,result,initial,ctx)
    scanGpu(m,result,initial,ctx)
  else:
    prepare(m,proj,result,initial)
    scanHost(m,result,initial)
  result.terminal=newMamba3State(m)
  if B==1:
    let C=m.intermediateSize; let N=m.dState; let H=m.numHeads; let R=m.ropeCount
    result.terminal.ssmState=result.states[T*C*N..<(T+1)*C*N]
    for i in 0..<H*R: result.terminal.aux[i]=result.angles[(T-1)*H*R+i]
    for i in 0..<H*N: result.terminal.aux[H*R+i]=result.k[(T-1)*H*N+i]
    for i in 0..<C: result.terminal.aux[H*(R+N)+i]=proj.data[(T-1)*m.projSize+C+i]

proc forwardMamba3Real*(m: Mamba3Real; x: Tensor; ctx: Backend): tuple[y:Tensor,cache:Mamba3Cache] =
  let proj=m.inProj.forward(x,ctx)
  result.cache=forwardCore(m,proj,x.shape[0],x.shape[1],newMamba3State(m),ctx)
  result.y=m.outProj.forward(result.cache.inner,ctx)

proc forwardStepMamba3Real*(m: Mamba3Real; x: Tensor; state: var Mamba3State;
    ctx: Backend; fp4InName=""; fp4OutName=""; useGpuSsm=true): Tensor =
  doAssert x.shape == @[1,1,m.hiddenSize]
  let proj=m.inProj.forward(x,ctx)
  let cache=forwardCore(m,proj,1,1,state,ctx)
  state=cache.terminal
  result=m.outProj.forward(cache.inner,ctx)

proc reverseScanGpu(m: Mamba3Real; c: Mamba3Cache; dy: Tensor; ctx: Backend):
    tuple[dp:Tensor, dk,dq,dc:seq[float32]] =
  let C=m.intermediateSize; let H=m.numHeads; let N=m.dState; let P=m.projSize
  var mem:seq[float32]
  template pack(a:seq[float32]):int =
    block:
      let o=mem.len
      mem.add(a)
      o
  let op=pack(c.proj.data); let ok=pack(c.k); let oq=pack(c.q)
  let oc=pack(c.coeff); let os=pack(c.states); let od=pack(m.dSkip.data)
  let og=pack(dy.data)
  let odp=pack(newSeq[float32](c.proj.data.len))
  let odk=pack(newSeq[float32](c.k.len)); let odq=pack(newSeq[float32](c.q.len))
  let odc=pack(newSeq[float32](c.coeff.len))
  let ods=pack(newSeq[float32](c.B*C*N))
  var p:VmProgram
  p.emitUTgid(0);p.emitUToR(0,0)
  p.emitMulConst(1,0,1'f32/float32(C));p.emitRToU(1,1)
  p.emitUMulK(2,1,uint32(C));p.emitUToR(2,2);p.emitSub(2,0,2);p.emitRToU(2,2)
  p.emitMulConst(3,2,1'f32/float32(m.headDim));p.emitRToU(3,3)
  p.emitUMulK(4,1,uint32(c.T*P));p.emitUAdd(4,4,2);p.emitUAddK(4,4,uint32((c.T-1)*P))
  p.emitUMulK(5,1,uint32(c.T*H*N));p.emitUMulK(6,3,uint32(N));p.emitUAdd(5,5,6);p.emitUAddK(5,5,uint32((c.T-1)*H*N))
  p.emitUMulK(6,1,uint32(c.T*H*3));p.emitUMulK(7,3,3);p.emitUAdd(6,6,7);p.emitUAddK(6,6,uint32((c.T-1)*H*3))
  p.emitUMulK(7,1,uint32((c.T+1)*C*N));p.emitUMulK(8,2,uint32(N));p.emitUAdd(7,7,8);p.emitUAddK(7,7,uint32((c.T-1)*C*N))
  p.emitUMulK(13,1,uint32(c.T*C));p.emitUAdd(13,13,2);p.emitUAddK(13,13,uint32((c.T-1)*C))
  p.emitUMulK(14,0,uint32(N))
  p.emitLoadConstF32(0,float32(c.T));p.emitLoadConstF32(20,float32(c.T-1))
  sessionVmForGpu(p,0):
    p.emitULoadF32(2,6,uint32(oc));p.emitULoadF32(3,6,uint32(oc+1));p.emitULoadF32(4,6,uint32(oc+2))
    p.emitULoadF32(5,4,uint32(op+C));p.emitLoadConstF32(6,0)
    p.emitLoadConstF32(21,0);p.emitCmpLt(21,21,20)
    sessionVmIfGpu(p,21):
      p.emitUAddK(15,4,uint32(0x100000000'u64-uint64(P)))
      p.emitULoadF32(6,15,uint32(op+C))
    p.emitULoadF32(7,3,uint32(od));p.emitMul(7,7,5)
    p.emitUAddK(8,7,0);p.emitUAddK(9,5,0)
    p.emitLoadConstF32(1,float32(N))
    sessionVmForGpu(p,1):
      p.emitULoadF32(8,8,uint32(os+C*N));p.emitULoadF32(12,9,uint32(oq));p.emitFma(7,8,12,7)
      p.emitUAddK(8,8,1);p.emitUAddK(9,9,1)
    p.emitULoadF32(13,13,uint32(og));p.emitULoadF32(14,4,uint32(op));p.emitSigmoid(15,14)
    p.emitMul(16,13,14);p.emitMul(16,16,15) # dY before gate
    p.emitLoadConstF32(17,1);p.emitSub(17,17,15);p.emitFma(17,14,17,17) # overwritten below
    p.emitLoadConstF32(17,1);p.emitSub(17,17,15);p.emitMul(17,17,14);p.emitAddConst(17,17,1)
    p.emitMul(17,17,15);p.emitMul(17,17,13);p.emitMul(17,17,7);p.emitUStoreF32(17,4,uint32(odp))
    p.emitULoadF32(17,3,uint32(od));p.emitMul(17,17,16)
    p.emitUAddK(15,4,uint32(C));p.emitUAtomicAdd(17,15,uint32(odp))
    p.emitUAddK(8,7,0);p.emitUAddK(9,5,0);p.emitUAddK(10,14,0)
    sessionVmForGpu(p,1):
      p.emitULoadF32(8,8,uint32(os+C*N));p.emitMul(17,16,8);p.emitUAtomicAdd(17,9,uint32(odq))
      p.emitULoadF32(12,9,uint32(oq));p.emitULoadF32(11,10,uint32(ods));p.emitFma(11,16,12,11)
      p.emitULoadF32(8,8,uint32(os));p.emitMul(17,11,8);p.emitUAtomicAdd(17,6,uint32(odc))
      p.emitLoadConstF32(10,0)
      sessionVmIfGpu(p,21):
        p.emitUAddK(11,9,uint32(0x100000000'u64-uint64(H*N)))
        p.emitULoadF32(10,11,uint32(ok))
      p.emitMul(17,11,10);p.emitMul(17,17,6);p.emitUAtomicAdd(17,6,uint32(odc+1))
      p.emitULoadF32(9,9,uint32(ok));p.emitMul(17,11,9);p.emitMul(17,17,5);p.emitUAtomicAdd(17,6,uint32(odc+2))
      p.emitMul(17,11,4);p.emitMul(18,17,5);p.emitUAtomicAdd(18,9,uint32(odk))
      p.emitMul(18,17,9);p.emitUAddK(15,4,uint32(C));p.emitUAtomicAdd(18,15,uint32(odp))
      sessionVmIfGpu(p,21):
        p.emitMul(17,11,3);p.emitMul(18,17,6);p.emitUAtomicAdd(18,11,uint32(odk))
        p.emitMul(18,17,10);p.emitUAddK(15,4,uint32(0x100000000'u64+uint64(C)-uint64(P)));p.emitUAtomicAdd(18,15,uint32(odp))
      p.emitMul(11,11,2);p.emitUStoreF32(11,10,uint32(ods))
      p.emitUAddK(8,8,1);p.emitUAddK(9,9,1);p.emitUAddK(10,10,1)
    p.emitUAddK(4,4,uint32(0x100000000'u64-uint64(P)))
    p.emitUAddK(5,5,uint32(0x100000000'u64-uint64(H*N)))
    p.emitUAddK(6,6,uint32(0x100000000'u64-uint64(H*3)))
    p.emitUAddK(7,7,uint32(0x100000000'u64-uint64(C*N)))
    p.emitUAddK(13,13,uint32(0x100000000'u64-uint64(C)))
    p.emitAddConst(20,20,-1)
  p.emitHalt()
  let tag="@m3_siso_bwd_" & $c.B & "_" & $c.T & "_" & $C & "_" & $H & "_" & $N
  if not runMicroVmChain(ctx,[(p,c.B*C,4,1,tag)],mem):
    raise newException(IOError,"Mamba-3 reverse scan dispatch failed")
  result.dp=Tensor(shape:c.proj.shape,data:mem[odp..<odp+c.proj.data.len])
  result.dk=mem[odk..<odk+c.k.len];result.dq=mem[odq..<odq+c.q.len];result.dc=mem[odc..<odc+c.coeff.len]

proc backwardMamba3Core*(m: Mamba3Real; c: Mamba3Cache; dInner: Tensor; ctx: Backend): Tensor =
  ## Reverse recurrence through trapezoidal interpolation, cumulative RoPE,
  ## grouped B/C RMSNorm, heavy-tail A, softplus dt, and SiLU gate.
  ## Base SSM tensors are frozen; only the projected input needs gradients for LoRA.
  let C=m.intermediateSize; let H=m.numHeads; let N=m.dState
  let P=m.projSize; let R=m.ropeCount; let BT=c.B*c.T
  result=newTensor(c.proj.shape)
  var dk=newSeq[float32](c.k.len); var dq=newSeq[float32](c.q.len)
  var dc=newSeq[float32](c.coeff.len)
  if isGpuCtx(ctx):
    let g=reverseScanGpu(m,c,dInner,ctx)
    result=g.dp;dk=g.dk;dq=g.dq;dc=g.dc
    return reversePrepareGpu(m,c,result,dk,dq,dc,ctx)
  else:
    var ds=newSeq[float32](c.B*C*N)
    for b in 0..<c.B:
      for t in countdown(c.T-1,0):
        let row=b*c.T+t
        for d in 0..<C:
          let h=d div m.headDim; let co=(row*H+h)*3
          let v=c.proj.data[row*P+C+d]
          let pv=if t>0: c.proj.data[(row-1)*P+C+d] else: 0'f32
          var y=m.dSkip.data[h]*v
          for n in 0..<N:
            y+=c.q[(row*H+h)*N+n]*c.states[((b*(c.T+1)+t+1)*C+d)*N+n]
          let z=c.proj.data[row*P+d]; let sig=sigmoid(z)
          let dy=dInner.data[row*C+d]*z*sig
          result.data[row*P+d]=dInner.data[row*C+d]*y*sig*(1'f32+z*(1'f32-sig))
          result.data[row*P+C+d]+=dy*m.dSkip.data[h]
          for n in 0..<N:
            let j=(row*H+h)*N+n
            let i=((b*(c.T+1)+t)*C+d)*N+n
            let si=(b*C+d)*N+n
            let pk=if t>0: c.k[j-H*N] else: 0'f32
            dq[j]+=dy*c.states[i+C*N]
            ds[si]+=dy*c.q[j]
            let g=ds[si]
            dc[co]+=g*c.states[i]
            dc[co+1]+=g*pk*pv
            dc[co+2]+=g*c.k[j]*v
            dk[j]+=g*c.coeff[co+2]*v
            result.data[row*P+C+d]+=g*c.coeff[co+2]*c.k[j]
            if t>0:
              dk[j-H*N]+=g*c.coeff[co+1]*pv
              result.data[(row-1)*P+C+d]+=g*c.coeff[co+1]*pk
            ds[si]=g*c.coeff[co]
  for b in 0..<c.B:
    var dphase=newSeq[float32](H*R)
    for t in countdown(c.T-1,0):
      let row=b*c.T+t; let off=row*P
      var db=newSeq[float32](N); var dcv=newSeq[float32](N)
      for h in 0..<H:
        let base=(row*H+h)*N; let co=(row*H+h)*3
        let dt=softplus(c.proj.data[off+2*C+2*N+h]+m.dtBias.data[h])
        let ap=c.proj.data[off+2*C+2*N+H+h]
        let heavy=if ap>=0: 1'f32+ap else: 1'f32/(1'f32-ap)
        let a= -max(1e-4'f32,heavy)
        let trap=sigmoid(c.proj.data[off+2*C+2*N+2*H+h])
        let alpha=c.coeff[co]
        var ddt=dc[co]*alpha*a+dc[co+1]*(1'f32-trap)*alpha*(1'f32+dt*a)+dc[co+2]*trap
        let da=dc[co]*alpha*dt+dc[co+1]*(1'f32-trap)*dt*alpha*dt
        result.data[off+2*C+2*N+H+h]=if heavy>1e-4'f32: -da*(if ap>=0: 1'f32 else: heavy*heavy) else: 0'f32
        result.data[off+2*C+2*N+2*H+h]=(-dc[co+1]*dt*alpha+dc[co+2]*dt)*trap*(1'f32-trap)
        for r in 0..<R:
          let j=base+2*r
          dphase[h*R+r]+= -dk[j]*c.k[j+1]+dk[j+1]*c.k[j]-dq[j]*c.q[j+1]+dq[j+1]*c.q[j]
          let angle=c.angles[(row*H+h)*R+r]
          let cs=cos(angle); let sn=sin(angle)
          let k0=dk[j]; let k1=dk[j+1]; let q0=dq[j]; let q1=dq[j+1]
          dk[j]=k0*cs+k1*sn; dk[j+1]= -k0*sn+k1*cs
          dq[j]=q0*cs+q1*sn; dq[j+1]= -q0*sn+q1*cs
          let th=tanh(c.proj.data[off+2*C+2*N+3*H+r])
          result.data[off+2*C+2*N+3*H+r]+=dphase[h*R+r]*dt*float32(PI)*(1'f32-th*th)
          ddt+=dphase[h*R+r]*th*float32(PI)
        result.data[off+2*C+2*N+h]=ddt*sigmoid(c.proj.data[off+2*C+2*N+h]+m.dtBias.data[h])
        for n in 0..<N:
          db[n]+=dk[base+n]; dcv[n]+=dq[base+n]
      for which in 0..1:
        let g=if which==0: db else: dcv
        let w=if which==0: m.bNorm.data else: m.cNorm.data
        let start=off+2*C+which*N
        var sq,dot=0'f32
        for n in 0..<N:
          sq+=c.proj.data[start+n]*c.proj.data[start+n]
          dot+=g[n]*w[n]*c.proj.data[start+n]
        let inv=1'f32/sqrt(sq/float32(N)+m.normEps)
        for n in 0..<N:
          result.data[start+n]=g[n]*w[n]*inv-c.proj.data[start+n]*dot*inv*inv*inv/float32(N)

proc backwardMamba3Real*(m: Mamba3Real; x: Tensor; cache: Mamba3Cache;
    dOut: Tensor; ctx: Backend; prefix: string; grads: var Table[string,Tensor]): Tensor =
  let g=m.outProj.backward(cache.inner,dOut,ctx)
  grads[prefix & ".outProj.weight"]=g.dW; grads[prefix & ".outProj.bias"]=g.dB
  let dp=backwardMamba3Core(m,cache,g.dX,ctx)
  let gi=m.inProj.backward(x,dp,ctx)
  grads[prefix & ".inProj.weight"]=gi.dW; grads[prefix & ".inProj.bias"]=gi.dB
  result=gi.dX
