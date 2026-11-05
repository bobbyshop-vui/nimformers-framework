## Resident Mamba-3 kernels. All activation/state addresses stay on the device.
import std/math
import session_vm, mamba2_vm_kernels

type
  M3Arena* = object
    hidden*, C*, H*, D*, N*, R*, P*, F*, capacity*: int
    x*, norm*, proj*, k*, q*, coeff*, inner*, mix*, mlp*, gated*: int
    angles*:int
    attnProj*, attnInner*, attnOut*, gateIn*, gate*, lora*, logits*, total*: int
  M3DeviceState* = object
    phase*, prevK*, prevV*, ssm*, lpS*, lpZ*, lpZNext*, total*: int
  M3Params* = object
    norm*, norm2*, bNorm*, cNorm*, bBias*, cBias*, dt*, skip*: int
  M3OffsetSlot* = enum
    moNorm, moNorm2, moPhase, moPrevK, moPrevV, moSsm,
    moBNorm, moCNorm, moBBias, moCBias, moDt, moSkip,
    moLpS, moLpZ, moLpZNext

proc layerOffset(p:var VmProgram; reg:uint32; slot:M3OffsetSlot) =
  # Offset descriptors use raw uint32 bits in the otherwise unused memU8
  # binding. Integer loads preserve exact addresses even above 2^24 floats.
  p.emitUConst(reg,0)
  p.emitULoadU32(reg,reg,uint32(ord(slot)*4))

proc stateLoad(p:var VmProgram; rd,ub:uint32; offset,dynamicReg:int;delta=0) =
  if dynamicReg<0:p.emitULoadF32(rd,ub,uint32(offset))
  else:
    p.emitUAdd(31,ub,uint32(dynamicReg))
    p.emitULoadF32(rd,31,uint32(delta))

proc stateStore(p:var VmProgram; rs,ub:uint32; offset,dynamicReg:int;delta=0) =
  if dynamicReg<0:p.emitUStoreF32(rs,ub,uint32(offset))
  else:
    p.emitUAdd(31,ub,uint32(dynamicReg))
    p.emitUStoreF32(rs,31,uint32(delta))

proc paramLoad(p:var VmProgram; rd,ub:uint32; offset,dynamicReg:int;delta=0) =
  if dynamicReg<0:p.emitULoadConstF(rd,ub,uint32(offset))
  else:
    p.emitUAdd(31,ub,uint32(dynamicReg))
    p.emitULoadConstF(rd,31,uint32(delta))

proc arenaLayout*(hidden,C,H,N,vocab,capacity,maxRank:int):M3Arena =
  result.hidden=hidden;result.C=C;result.H=H;result.N=N
  result.D=C div H;result.R=N div 4
  result.P=2*C+2*N+3*H+result.R;result.F=2*hidden;result.capacity=capacity
  var o=64
  template take(f:untyped; n:int)=
    f=o;o+=n
  take(result.x,capacity*hidden);take(result.norm,capacity*hidden)
  take(result.proj,capacity*result.P);take(result.k,capacity*H*N)
  take(result.q,capacity*H*N);take(result.coeff,capacity*H*3)
  take(result.angles,capacity*H*result.R)
  take(result.inner,capacity*C);take(result.mix,capacity*hidden)
  take(result.mlp,capacity*4*hidden);take(result.gated,capacity*2*hidden)
  take(result.attnProj,capacity*3*hidden);take(result.attnInner,capacity*hidden)
  take(result.attnOut,capacity*hidden);take(result.gateIn,capacity*2*hidden)
  take(result.gate,capacity*hidden);take(result.lora,capacity*max(32,maxRank))
  take(result.logits,vocab)
  result.total=o

proc stateLayout*(a:M3Arena; base,attnHeads:int):M3DeviceState =
  var o=base
  template take(f:untyped;n:int)=
    f=o;o+=n
  take(result.phase,a.H*a.R);take(result.prevK,a.H*a.N)
  take(result.prevV,a.C);take(result.ssm,a.C*a.N)
  if attnHeads>0:
    let d=a.hidden div attnHeads
    take(result.lpS,attnHeads*d*d);take(result.lpZ,a.hidden);take(result.lpZNext,a.hidden)
  result.total=o

proc rowIndex*(p:var VmProgram; cols:int)=
  p.emitRToU(0,0);p.emitMulConst(1,0,1'f32/float32(cols));p.emitRToU(1,1)
  p.emitUMulK(2,1,uint32(cols));p.emitUToR(2,2);p.emitSub(2,0,2);p.emitRToU(2,2)

proc rmsRows*(C,input,output,gamma:int; eps:float32;
    dynamicOffsets=false; gammaSlot:M3OffsetSlot=moNorm):VmProgram =
  var p:VmProgram
  let gammaReg=if dynamicOffsets:30 else: -1
  if dynamicOffsets:p.layerOffset(30,gammaSlot)
  p.emitUTgid(0);p.emitULid(1);p.emitUToR(1,1)
  p.emitUMulK(2,0,uint32(C));p.emitUAdd(3,2,1)
  p.emitLoadConstF32(3,0);p.emitLoadConstF32(8,float32((C+255) div 256))
  p.emitUToR(2,1)
  sessionVmForGpu(p,8):
    p.emitLoadConstF32(4,float32(C));p.emitCmpLt(4,2,4)
    sessionVmIfGpu(p,4):
      p.emitULoadF32(5,3,uint32(input));p.emitFma(3,5,5,3)
    p.emitUAddK(3,3,256);p.emitAddConst(2,2,256)
  p.emitTgStoreF32(3,1,0);p.emitBarrierTg()
  p.emitTgTreeReduce(1,256,4,5,6,10,11)
  p.emitLoadConstF32(4,0);p.emitTgLoadF32(15,4,0)
  p.emitMulConst(15,15,1'f32/float32(C));p.emitAddConst(15,15,eps);p.emitRsqrt(15,15)
  p.emitUAdd(3,2,1);p.emitUAddK(4,1,0);p.emitUToR(2,1)
  sessionVmForGpu(p,8):
    p.emitLoadConstF32(5,float32(C));p.emitCmpLt(5,2,5)
    sessionVmIfGpu(p,5):
      p.emitULoadF32(6,3,uint32(input));p.paramLoad(7,4,gamma,gammaReg)
      p.emitMul(6,6,15);p.emitMul(6,6,7);p.emitUStoreF32(6,3,uint32(output))
    p.emitUAddK(3,3,256);p.emitUAddK(4,4,256);p.emitAddConst(2,2,256)
  p.emitHalt();p

proc sincos*(p:var VmProgram; angle:uint32)=
  ## Range-reduced degree 19/18 polynomials; no host trig or vendor changes.
  ## Returns R12=sin, R13=cos, wraps the phase into [-pi,pi]. Scratch R8..15,U15.
  p.emitLoadConstF32(8,0);p.emitCmpLt(8,angle,8)
  p.emitMulConst(9,8,-2);p.emitAddConst(9,9,1)
  p.emitMul(10,angle,9);p.emitMulConst(11,10,1'f32/float32(2*PI))
  p.emitRToU(15,11);p.emitUToR(11,15);p.emitMulConst(11,11,float32(2*PI));p.emitSub(10,10,11)
  p.emitLoadConstF32(11,float32(PI));p.emitCmpLt(11,11,10)
  sessionVmIfGpu(p,11): p.emitAddConst(10,10,-float32(2*PI))
  p.emitMul(angle,10,9);p.emitMul(8,angle,angle)
  p.emitLoadConstF32(12,-1'f32/121645100408832000'f32)
  for v in [1'f32/355687428096000'f32,-1'f32/1307674368000'f32,1'f32/6227020800'f32,
      -1'f32/39916800'f32,1'f32/362880'f32,-1'f32/5040'f32,1'f32/120'f32,-1'f32/6'f32,1'f32]:
    p.emitMul(12,12,8);p.emitAddConst(12,12,v)
  p.emitMul(12,12,angle)
  p.emitLoadConstF32(13,-1'f32/6402373705728000'f32)
  for v in [1'f32/20922789888000'f32,-1'f32/87178291200'f32,1'f32/479001600'f32,
      -1'f32/3628800'f32,1'f32/40320'f32,-1'f32/720'f32,1'f32/24'f32,-1'f32/2'f32,1'f32]:
    p.emitMul(13,13,8);p.emitAddConst(13,13,v)

proc prepareSiso*(a:M3Arena; s:M3DeviceState; w:M3Params; T:int;eps:float32;
    dynamicOffsets=false):VmProgram =
  var p:VmProgram
  let phaseReg=if dynamicOffsets:16 else: -1
  let dtReg=if dynamicOffsets:17 else: -1
  let bNormReg=if dynamicOffsets:18 else: -1
  let cNormReg=if dynamicOffsets:19 else: -1
  let bBiasReg=if dynamicOffsets:20 else: -1
  let cBiasReg=if dynamicOffsets:21 else: -1
  if dynamicOffsets:
    p.layerOffset(16,moPhase);p.layerOffset(17,moDt)
    p.layerOffset(18,moBNorm);p.layerOffset(19,moCNorm)
    p.layerOffset(20,moBBias);p.layerOffset(21,moCBias)
  p.emitUTgid(0);p.emitULid(1);p.emitUToR(1,1)
  p.emitUMulK(2,0,uint32(a.N));p.emitUAdd(3,2,1) # head*N+lane
  p.emitUMulK(4,0,uint32(a.R));p.emitUAdd(4,4,1)
  p.emitLoadConstF32(26,0);p.emitLoadConstF32(28,float32(a.R));p.emitCmpLt(28,1,28)
  sessionVmIfGpu(p,28):p.stateLoad(26,4,s.phase,phaseReg)
  p.emitUConst(5,0);p.emitUMulK(6,0,uint32(a.N))
  p.emitUMulK(7,0,3);p.emitLoadConstF32(27,float32(T))
  sessionVmForGpu(p,27):
    p.emitUAdd(8,5,1)
    p.emitLoadConstF32(2,0);p.emitLoadConstF32(3,0)
    p.emitLoadConstF32(4,float32(a.N));p.emitCmpLt(4,1,4)
    sessionVmIfGpu(p,4):
      p.emitULoadF32(2,8,uint32(a.proj+2*a.C))
      p.emitULoadF32(3,8,uint32(a.proj+2*a.C+a.N))
    p.emitMul(5,2,2);p.emitTgStoreF32(5,1,0);p.emitBarrierTg()
    p.emitTgTreeReduce(1,256,4,5,6,7,8)
    p.emitLoadConstF32(9,0);p.emitTgLoadF32(10,9,0)
    p.emitMulConst(10,10,1'f32/float32(a.N));p.emitAddConst(10,10,eps);p.emitRsqrt(22,10)
    p.emitBarrierTg();p.emitMul(5,3,3);p.emitTgStoreF32(5,1,0);p.emitBarrierTg()
    p.emitTgTreeReduce(1,256,4,5,6,7,8)
    p.emitLoadConstF32(9,0);p.emitTgLoadF32(10,9,0)
    p.emitMulConst(10,10,1'f32/float32(a.N));p.emitAddConst(10,10,eps);p.emitRsqrt(23,10)
    p.emitUAdd(8,5,0)
    p.emitULoadF32(16,8,uint32(a.proj+2*a.C+2*a.N))
    p.paramLoad(17,0,w.dt,dtReg);p.emitAdd(16,16,17);p.emitSoftplus(16,16)
    p.emitULoadF32(17,8,uint32(a.proj+2*a.C+2*a.N+a.H))
    p.emitLoadConstF32(18,0);p.emitCmpLt(18,17,18)
    sessionVmIfGpu(p,18):
      p.emitNeg(17,17);p.emitAddConst(17,17,1);p.emitLoadConstF32(19,1);p.emitDiv(17,19,17)
    p.emitLoadConstF32(19,1);p.emitSub(18,19,18)
    sessionVmIfGpu(p,18):p.emitAddConst(17,17,1)
    p.emitLoadConstF32(18,1e-4'f32);p.emitCmpLt(19,17,18)
    sessionVmIfGpu(p,19):p.emitLoadConstF32(17,1e-4'f32)
    p.emitNeg(17,17);p.emitMul(17,17,16);p.emitExp(17,17)
    p.emitULoadF32(20,8,uint32(a.proj+2*a.C+2*a.N+2*a.H));p.emitSigmoid(20,20)
    p.emitLoadConstF32(18,1);p.emitSub(18,18,20);p.emitMul(18,18,16);p.emitMul(18,18,17)
    p.emitMul(19,20,16)
    p.emitLoadConstF32(4,1);p.emitCmpLt(4,1,4)
    sessionVmIfGpu(p,4):
      p.emitUStoreF32(17,7,uint32(a.coeff));p.emitUStoreF32(18,7,uint32(a.coeff+1));p.emitUStoreF32(19,7,uint32(a.coeff+2))
    p.emitLoadConstF32(4,float32(a.N div 2));p.emitCmpLt(4,1,4)
    sessionVmIfGpu(p,4):
      p.emitUMulK(9,1,2);p.emitUAdd(10,5,9);p.emitUAdd(11,2,9)
      p.emitULoadF32(2,10,uint32(a.proj+2*a.C));p.emitULoadF32(3,10,uint32(a.proj+2*a.C+1))
      p.emitULoadF32(4,10,uint32(a.proj+2*a.C+a.N));p.emitULoadF32(5,10,uint32(a.proj+2*a.C+a.N+1))
      p.paramLoad(6,9,w.bNorm,bNormReg);p.paramLoad(7,9,w.bNorm+1,bNormReg,1)
      p.emitMul(2,2,22);p.emitMul(3,3,22);p.emitMul(2,2,6);p.emitMul(3,3,7)
      p.paramLoad(6,9,w.cNorm,cNormReg);p.paramLoad(7,9,w.cNorm+1,cNormReg,1)
      p.emitMul(4,4,23);p.emitMul(5,5,23);p.emitMul(4,4,6);p.emitMul(5,5,7)
      p.paramLoad(6,11,w.bBias,bBiasReg);p.paramLoad(7,11,w.bBias+1,bBiasReg,1);p.emitAdd(2,2,6);p.emitAdd(3,3,7)
      p.paramLoad(6,11,w.cBias,cBiasReg);p.paramLoad(7,11,w.cBias+1,cBiasReg,1);p.emitAdd(4,4,6);p.emitAdd(5,5,7)
      p.emitLoadConstF32(6,float32(a.R));p.emitCmpLt(6,1,6)
      sessionVmIfGpu(p,6):
        p.emitUAdd(12,5,1);p.emitULoadF32(6,12,uint32(a.proj+2*a.C+2*a.N+3*a.H))
        p.emitMulConst(6,6,2);p.emitSigmoid(6,6);p.emitMulConst(6,6,2);p.emitAddConst(6,6,-1)
        p.emitMul(6,6,16);p.emitMulConst(6,6,float32(PI));p.emitAdd(26,26,6)
        p.sincos(26)
        p.emitUToR(8,5);p.emitMulConst(8,8,1'f32/float32(a.P));p.emitRToU(13,8)
        p.emitUMulK(13,13,uint32(a.H*a.R));p.emitUAdd(13,13,4)
        p.emitUStoreF32(26,13,uint32(a.angles))
        p.emitMul(6,2,13);p.emitMul(7,3,12);p.emitSub(6,6,7)
        p.emitMul(7,2,12);p.emitFma(3,3,13,7);p.emitAddConst(2,6,0)
        p.emitMul(6,4,13);p.emitMul(7,5,12);p.emitSub(6,6,7)
        p.emitMul(7,4,12);p.emitFma(5,5,13,7);p.emitAddConst(4,6,0)
      p.emitUAdd(12,6,9)
      p.emitUStoreF32(2,12,uint32(a.k));p.emitUStoreF32(3,12,uint32(a.k+1))
      p.emitUStoreF32(4,12,uint32(a.q));p.emitUStoreF32(5,12,uint32(a.q+1))
    p.emitBarrierTg()
    p.emitUAddK(5,5,uint32(a.P));p.emitUAddK(6,6,uint32(a.H*a.N));p.emitUAddK(7,7,uint32(a.H*3))
  p.emitLoadConstF32(4,float32(a.R));p.emitCmpLt(4,1,4)
  sessionVmIfGpu(p,4):p.stateStore(26,4,s.phase,phaseReg)
  p.emitHalt();p

proc scanSiso*(a:M3Arena;s:M3DeviceState;w:M3Params;T:int;dynamicOffsets=false):VmProgram =
  var p:VmProgram
  let ssmReg=if dynamicOffsets:16 else: -1
  let prevVReg=if dynamicOffsets:17 else: -1
  let prevKReg=if dynamicOffsets:18 else: -1
  let skipReg=if dynamicOffsets:19 else: -1
  if dynamicOffsets:
    p.layerOffset(16,moSsm);p.layerOffset(17,moPrevV)
    p.layerOffset(18,moPrevK);p.layerOffset(19,moSkip)
  p.emitUTgid(0);p.emitULid(1);p.emitUToR(1,1)
  p.emitUToR(0,0);p.emitMulConst(2,0,1'f32/float32(a.D));p.emitRToU(2,2)
  p.emitUMulK(3,0,uint32(a.N));p.emitUAdd(3,3,1) # S[d,n]
  p.emitUMulK(4,2,uint32(a.N));p.emitUAdd(4,4,1) # K[h,n]
  p.emitUMulK(5,2,3);p.emitUAddK(6,0,0) # coefficient, value row
  p.emitUConst(7,0);p.emitLoadConstF32(24,float32(T));p.emitLoadConstF32(25,0)
  let slices=(a.N+31) div 32
  for j in 0..<slices:
    let r=uint32(16+j)
    p.emitLoadConstF32(r,0);p.emitLoadConstF32(2,float32(a.N-j*32));p.emitCmpLt(2,1,2)
    sessionVmIfGpu(p,2):p.stateLoad(r,3,s.ssm+j*32,ssmReg,j*32)
  sessionVmForGpu(p,24):
    p.emitULoadF32(8,5,uint32(a.coeff));p.emitULoadF32(9,5,uint32(a.coeff+1));p.emitULoadF32(10,5,uint32(a.coeff+2))
    p.emitUAdd(8,7,0);p.emitULoadF32(3,8,uint32(a.proj+a.C))
    p.stateLoad(4,0,s.prevV,prevVReg)
    p.emitLoadConstF32(2,0);p.emitCmpLt(2,2,25)
    sessionVmIfGpu(p,2):
      p.emitUAddK(9,8,cast[uint32](-a.P));p.emitULoadF32(4,9,uint32(a.proj+a.C))
    p.emitLoadConstF32(11,0)
    for j in 0..<slices:
      let r=uint32(16+j)
      p.emitLoadConstF32(2,float32(a.N-j*32));p.emitCmpLt(2,1,2)
      sessionVmIfGpu(p,2):
        p.emitULoadF32(5,4,uint32(a.k+j*32));p.emitULoadF32(6,4,uint32(a.q+j*32))
        p.emitUAddK(9,4,0)
        p.emitLoadConstF32(2,0);p.emitCmpLt(2,2,25)
        # Initial K is addressed by head*N+lane; later K uses the preceding row.
        p.emitUMulK(10,2,uint32(a.N));p.emitUAdd(10,10,1)
        p.stateLoad(7,10,s.prevK+j*32,prevKReg,j*32)
        sessionVmIfGpu(p,2):
          p.emitUAddK(9,4,cast[uint32](-a.H*a.N));p.emitULoadF32(7,9,uint32(a.k+j*32))
        p.emitMul(r,r,8);p.emitMul(7,7,4);p.emitFma(r,7,9,r)
        p.emitMul(5,5,3);p.emitFma(r,5,10,r)
        p.emitFma(11,r,6,11)
    p.emitTgStoreF32(11,1,0);p.emitBarrierTg()
    p.emitTgTreeReduce(1,32,2,5,6,7,12)
    p.emitLoadConstF32(2,1);p.emitCmpLt(2,1,2)
    sessionVmIfGpu(p,2):
      p.emitLoadConstF32(2,0);p.emitTgLoadF32(11,2,0)
      p.paramLoad(5,2,w.skip,skipReg)
      p.emitFma(11,5,3,11);p.emitULoadF32(5,8,uint32(a.proj));p.emitSigmoid(6,5)
      p.emitMul(11,11,5);p.emitMul(11,11,6);p.emitUStoreF32(11,6,uint32(a.inner))
    p.emitBarrierTg()
    p.emitUAddK(4,4,uint32(a.H*a.N));p.emitUAddK(5,5,uint32(a.H*3))
    p.emitUAddK(6,6,uint32(a.C));p.emitUAddK(7,7,uint32(a.P));p.emitAddConst(25,25,1)
  for j in 0..<slices:
    p.emitLoadConstF32(2,float32(a.N-j*32));p.emitCmpLt(2,1,2)
    sessionVmIfGpu(p,2):p.stateStore(uint32(16+j),3,s.ssm+j*32,ssmReg,j*32)
  p.emitHalt();p

proc carrySiso*(a:M3Arena;s:M3DeviceState;T:int;dynamicOffsets=false):VmProgram =
  var p:VmProgram
  let prevKReg=if dynamicOffsets:16 else: -1
  let prevVReg=if dynamicOffsets:17 else: -1
  if dynamicOffsets:p.layerOffset(16,moPrevK);p.layerOffset(17,moPrevV)
  p.emitRToU(0,0)
  p.emitLoadConstF32(1,float32(a.H*a.N));p.emitCmpLt(1,0,1)
  sessionVmIfGpu(p,1):
    p.emitULoadF32(2,0,uint32(a.k+(T-1)*a.H*a.N));p.stateStore(2,0,s.prevK,prevKReg)
  p.emitLoadConstF32(1,float32(a.C));p.emitCmpLt(1,0,1)
  sessionVmIfGpu(p,1):
    p.emitULoadF32(2,0,uint32(a.proj+(T-1)*a.P+a.C));p.stateStore(2,0,s.prevV,prevVReg)
  p.emitHalt();p

proc siluMlp*(a:M3Arena):VmProgram =
  var p:VmProgram
  p.rowIndex(a.F);p.emitUMulK(3,1,uint32(a.F*2));p.emitUAdd(3,3,2)
  p.emitULoadF32(4,3,uint32(a.mlp));p.emitULoadF32(5,3,uint32(a.mlp+a.F))
  p.emitSigmoid(6,5);p.emitMul(4,4,5);p.emitMul(4,4,6)
  p.emitUStoreF32(4,0,uint32(a.gated));p.emitHalt();p

proc residual*(input,branch,output:int):VmProgram =
  var p:VmProgram
  p.emitRToU(0,0);p.emitULoadF32(1,0,uint32(input));p.emitULoadF32(2,0,uint32(branch))
  p.emitAdd(1,1,2);p.emitUStoreF32(1,0,uint32(output));p.emitHalt();p

proc gateInput*(a:M3Arena):VmProgram =
  var p:VmProgram
  p.rowIndex(a.hidden);p.emitUMulK(3,1,uint32(2*a.hidden));p.emitUAdd(3,3,2)
  p.emitULoadF32(4,0,uint32(a.mix));p.emitULoadF32(5,0,uint32(a.attnOut))
  p.emitUStoreF32(4,3,uint32(a.gateIn));p.emitUStoreF32(5,3,uint32(a.gateIn+a.hidden))
  p.emitHalt();p

proc mixGate*(a:M3Arena):VmProgram =
  var p:VmProgram
  p.emitRToU(0,0);p.emitULoadF32(1,0,uint32(a.mix));p.emitULoadF32(2,0,uint32(a.attnOut))
  p.emitULoadF32(3,0,uint32(a.gate));p.emitSigmoid(3,3)
  p.emitLoadConstF32(4,1);p.emitSub(4,4,3);p.emitMul(1,1,3);p.emitFma(1,2,4,1)
  p.emitUStoreF32(1,0,uint32(a.mix));p.emitHalt();p

proc biasRows*(cols,output:int):VmProgram =
  var p:VmProgram
  p.rowIndex(cols);p.emitULoadF32(4,0,uint32(output));p.emitULoadConstF(5,2)
  p.emitAdd(4,4,5);p.emitUStoreF32(4,0,uint32(output));p.emitHalt();p

proc loraRows*(cols,rank,output,scratch:int;scale:float32):VmProgram =
  var p:VmProgram
  p.rowIndex(cols);p.emitUMulK(3,1,uint32(rank));p.emitUMulK(4,2,uint32(rank))
  p.emitLoadConstF32(4,0);p.emitLoadConstF32(8,float32(rank))
  sessionVmForGpu(p,8):
    p.emitULoadIn(5,3,uint32(scratch));p.emitULoadConstF(6,4)
    p.emitFma(4,5,6,4);p.emitUAddK(3,3,1);p.emitUAddK(4,4,1)
  p.emitMulConst(4,4,scale);p.emitULoadF32(5,0,uint32(output));p.emitAdd(4,4,5)
  p.emitUStoreF32(4,0,uint32(output));p.emitHalt();p

proc linearPlusScan*(a:M3Arena;s:M3DeviceState;heads,T:int;dynamicOffsets=false):VmProgram =
  let D=a.hidden div heads
  let S=(D+31) div 32
  doAssert S<=8
  var p:VmProgram
  let sReg=if dynamicOffsets:16 else: -1
  let zReg=if dynamicOffsets:17 else: -1
  let zNextReg=if dynamicOffsets:18 else: -1
  if dynamicOffsets:
    p.layerOffset(16,moLpS);p.layerOffset(17,moLpZ);p.layerOffset(18,moLpZNext)
  p.emitUTgid(0);p.emitULid(1);p.emitUToR(1,1);p.emitUToR(0,0)
  p.emitMulConst(2,0,1'f32/float32(D));p.emitRToU(2,2)
  p.emitUMulK(3,2,uint32(D));p.emitUAdd(3,3,1) # key global
  p.emitUMulK(4,3,uint32(D))
  p.emitUMulK(5,2,uint32(D));p.emitUMulK(5,5,0xffffffff'u32);p.emitUAdd(5,0,5)
  p.emitUAdd(4,4,5) # state[head,key,value]
  p.emitUConst(6,0);p.emitUAddK(7,0,0)
  for j in 0..<S:
    p.emitLoadConstF32(uint32(16+j),0);p.emitLoadConstF32(uint32(24+j),0)
    p.emitLoadConstF32(2,float32(D-j*32));p.emitCmpLt(2,1,2)
    sessionVmIfGpu(p,2):
      p.stateLoad(uint32(16+j),4,s.lpS+j*32*D,sReg,j*32*D)
      p.stateLoad(uint32(24+j),3,s.lpZ+j*32,zReg,j*32)
  p.emitLoadConstF32(15,float32(T))
  sessionVmForGpu(p,15):
    p.emitUAdd(8,6,0);p.emitULoadF32(3,8,uint32(a.attnProj+2*a.hidden))
    p.emitLoadConstF32(11,0);p.emitLoadConstF32(12,0)
    for j in 0..<S:
      p.emitLoadConstF32(2,float32(D-j*32));p.emitCmpLt(2,1,2)
      sessionVmIfGpu(p,2):
        p.emitUAdd(9,6,3);p.emitULoadF32(4,9,uint32(a.attnProj+a.hidden+j*32))
        p.emitULoadF32(5,9,uint32(a.attnProj+j*32))
        for r in [4'u32,5'u32]:
          p.emitLoadConstF32(6,0);p.emitCmpLt(6,r,6)
          sessionVmIfGpu(p,6):p.emitExp(r,r)
          p.emitLoadConstF32(7,1);p.emitSub(6,7,6)
          sessionVmIfGpu(p,6):p.emitAddConst(r,r,1)
        p.emitFma(uint32(16+j),4,3,uint32(16+j));p.emitAdd(uint32(24+j),uint32(24+j),4)
        p.emitFma(11,5,uint32(16+j),11);p.emitFma(12,5,uint32(24+j),12)
    p.emitTgStoreF32(11,1,0);p.emitBarrierTg();p.emitTgTreeReduce(1,32,2,4,5,6,7)
    p.emitLoadConstF32(2,0);p.emitTgLoadF32(11,2,0);p.emitBarrierTg()
    p.emitTgStoreF32(12,1,0);p.emitBarrierTg();p.emitTgTreeReduce(1,32,2,4,5,6,7)
    p.emitLoadConstF32(2,1);p.emitCmpLt(2,1,2)
    sessionVmIfGpu(p,2):
      p.emitLoadConstF32(2,0);p.emitTgLoadF32(12,2,0);p.emitAddConst(12,12,1e-6'f32)
      p.emitDiv(11,11,12);p.emitUStoreF32(11,7,uint32(a.attnInner))
    p.emitBarrierTg();p.emitUAddK(6,6,uint32(3*a.hidden));p.emitUAddK(7,7,uint32(a.hidden))
  for j in 0..<S:
    p.emitLoadConstF32(2,float32(D-j*32));p.emitCmpLt(2,1,2)
    sessionVmIfGpu(p,2):
      p.stateStore(uint32(16+j),4,s.lpS+j*32*D,sReg,j*32*D)
      p.emitUToR(6,5);p.emitLoadConstF32(7,1);p.emitCmpLt(6,6,7)
      sessionVmIfGpu(p,6):p.stateStore(uint32(24+j),3,s.lpZNext+j*32,zNextReg,j*32)
  p.emitHalt();p

proc loraDown*(K, rank: int): VmProgram =
  ## mem[t*rank + j] = sum_k in[t*K + k] * A[j*K + k] (A in the const buffer).
  ## One 256-lane group per (t, j) with a shared tree reduction; the generic
  ## 16x16 NT matmul ran this rank-4 product on ceil(T/16) workgroups.
  doAssert K mod 32 == 0 and rank > 0
  var p: VmProgram
  p.emitUTgid(0); p.emitULid(1)
  p.emitUToR(0, 0); p.emitLoadConstF32(1, float32(rank)); p.emitDiv(2, 0, 1)
  p.emitRToU(2, 2)                                           # U2 = t
  p.emitUMulK(3, 2, uint32(rank)); p.emitUToR(3, 3); p.emitSub(3, 0, 3)
  p.emitRToU(3, 3)                                           # U3 = j
  p.emitUMulK(5, 2, uint32(K)); p.emitUAdd(5, 5, 1)          # x[t, lane]
  p.emitUMulK(6, 3, uint32(K)); p.emitUAdd(6, 6, 1)          # A[j, lane]
  p.emitLoadConstF32(4, 0'f32)
  p.emitLoadConstF32(7, float32(K div 32))
  sessionVmForGpu(p, 7):
    p.emitULoadIn(5, 5); p.emitULoadConstF(6, 6); p.emitFma(4, 5, 6, 4)
    p.emitUAddK(5, 5, uint32(32)); p.emitUAddK(6, 6, uint32(32))
  p.emitUToR(10, 1)                                          # R10 = lane
  p.emitTgStoreF32(4, 10, 0)
  p.emitBarrierTg()
  var stride = 32 div 2
  while stride > 0:
    p.emitLoadConstF32(8, float32(stride)); p.emitCmpLt(11, 10, 8)
    sessionVmIfGpu(p, 11):
      p.emitTgLoadF32(5, 10, uint32(stride)); p.emitTgLoadF32(6, 10, 0)
      p.emitAdd(6, 6, 5); p.emitTgStoreF32(6, 10, 0)
    p.emitBarrierTg()
    stride = stride div 2
  p.emitLoadConstF32(8, 1'f32); p.emitCmpLt(11, 10, 8)
  sessionVmIfGpu(p, 11):
    p.emitTgLoadF32(5, 10, 0); p.emitUStoreF32(5, 0)
  p.emitHalt()
  result = p


proc carryLinearPlus*(a:M3Arena;s:M3DeviceState;dynamicOffsets=false):VmProgram =
  var p:VmProgram
  let zReg=if dynamicOffsets:16 else: -1
  let zNextReg=if dynamicOffsets:17 else: -1
  if dynamicOffsets:p.layerOffset(16,moLpZ);p.layerOffset(17,moLpZNext)
  p.emitRToU(0,0);p.stateLoad(1,0,s.lpZNext,zNextReg)
  p.stateStore(1,0,s.lpZ,zReg);p.emitHalt();p
