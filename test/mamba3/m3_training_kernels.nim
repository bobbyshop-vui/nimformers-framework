## GPU reverse of data-dependent A/dt, cumulative RoPE and grouped B/C norm.
import std/math
import session_vm, mamba2_vm_kernels, m3_inference_kernels

type M3ReverseOffsets* = object
  proj*,k*,q*,coeff*,angles*,dk*,dq*,dc*,dp*,ddt*,total*:int

proc rotateBackward*(a:M3Arena;o:M3ReverseOffsets;dtBias:int;T:int):VmProgram =
  var p:VmProgram
  p.emitUTgid(0);p.emitULid(1);p.emitUToR(1,1)
  p.emitUMulK(2,0,uint32(a.N));p.emitUMulK(3,1,2);p.emitUAdd(2,2,3)
  p.emitUAddK(2,2,uint32((T-1)*a.H*a.N))
  p.emitUConst(4,uint32((T-1)*a.P));p.emitUMulK(5,0,3);p.emitUAddK(5,5,uint32((T-1)*a.H*3))
  p.emitUMulK(6,0,uint32(a.R));p.emitUAdd(6,6,1);p.emitUAddK(6,6,uint32((T-1)*a.H*a.R))
  p.emitUAddK(7,0,uint32((T-1)*a.H))
  p.emitLoadConstF32(27,0);p.emitLoadConstF32(28,float32(T))
  sessionVmForGpu(p,28):
    p.emitUAdd(8,4,0)
    p.emitULoadF32(16,8,uint32(o.proj+2*a.C+2*a.N));p.emitULoadConstF(17,0,uint32(dtBias))
    p.emitAdd(16,16,17);p.emitSoftplus(16,16)
    p.emitLoadConstF32(2,1);p.emitCmpLt(2,1,2)
    sessionVmIfGpu(p,2):
      p.emitULoadF32(17,8,uint32(o.proj+2*a.C+2*a.N+a.H))
      p.emitLoadConstF32(2,0);p.emitCmpLt(2,17,2)
      p.emitLoadConstF32(18,1);p.emitAdd(18,18,17) # positive heavy tail
      sessionVmIfGpu(p,2):
        p.emitNeg(18,17);p.emitAddConst(18,18,1);p.emitLoadConstF32(3,1);p.emitDiv(18,3,18)
      p.emitLoadConstF32(3,1e-4'f32);p.emitCmpLt(3,18,3)
      sessionVmIfGpu(p,3):p.emitLoadConstF32(18,1e-4'f32)
      p.emitNeg(19,18) # A
      p.emitULoadF32(20,5,uint32(o.coeff)) # alpha
      p.emitULoadF32(21,8,uint32(o.proj+2*a.C+2*a.N+2*a.H));p.emitSigmoid(21,21)
      p.emitULoadF32(4,5,uint32(o.dc));p.emitULoadF32(5,5,uint32(o.dc+1));p.emitULoadF32(6,5,uint32(o.dc+2))
      p.emitMul(7,4,20);p.emitMul(22,7,19)
      p.emitLoadConstF32(23,1);p.emitSub(23,23,21);p.emitMul(23,23,20);p.emitMul(23,23,5)
      p.emitMul(24,16,19);p.emitAddConst(24,24,1);p.emitFma(22,23,24,22);p.emitFma(22,6,21,22)
      p.emitUStoreF32(22,7,uint32(o.ddt))
      p.emitMul(7,7,16);p.emitMul(23,23,16);p.emitFma(7,23,16,7)
      p.emitLoadConstF32(24,1)
      sessionVmIfGpu(p,2):p.emitMul(24,18,18)
      p.emitMul(7,7,24);p.emitNeg(7,7)
      sessionVmIfGpu(p,3):p.emitLoadConstF32(7,0)
      p.emitUStoreF32(7,8,uint32(o.dp+2*a.C+2*a.N+a.H))
      p.emitMul(7,5,16);p.emitMul(7,7,20);p.emitNeg(7,7);p.emitFma(7,6,16,7)
      p.emitLoadConstF32(24,1);p.emitSub(24,24,21);p.emitMul(24,24,21);p.emitMul(7,7,24)
      p.emitUStoreF32(7,8,uint32(o.dp+2*a.C+2*a.N+2*a.H))
    p.emitBarrierTg()
    p.emitLoadConstF32(2,float32(a.N div 2));p.emitCmpLt(2,1,2)
    sessionVmIfGpu(p,2):
      p.emitULoadF32(2,2,uint32(o.dk));p.emitULoadF32(3,2,uint32(o.dk+1))
      p.emitULoadF32(4,2,uint32(o.dq));p.emitULoadF32(5,2,uint32(o.dq+1))
      p.emitLoadConstF32(6,float32(a.R));p.emitCmpLt(6,1,6)
      sessionVmIfGpu(p,6):
        p.emitULoadF32(6,2,uint32(o.k));p.emitULoadF32(7,2,uint32(o.k+1))
        p.emitFma(27,3,6,27);p.emitMul(7,2,7);p.emitSub(27,27,7)
        p.emitULoadF32(6,2,uint32(o.q));p.emitULoadF32(7,2,uint32(o.q+1))
        p.emitFma(27,5,6,27);p.emitMul(7,4,7);p.emitSub(27,27,7)
        p.emitULoadF32(26,6,uint32(o.angles));p.sincos(26)
        p.emitMul(6,2,13);p.emitFma(6,3,12,6)
        p.emitMul(7,3,13);p.emitMul(2,2,12);p.emitSub(3,7,2);p.emitAddConst(2,6,0)
        p.emitMul(6,4,13);p.emitFma(6,5,12,6)
        p.emitMul(7,5,13);p.emitMul(4,4,12);p.emitSub(5,7,4);p.emitAddConst(4,6,0)
        p.emitUAdd(9,4,1);p.emitULoadF32(6,9,uint32(o.proj+2*a.C+2*a.N+3*a.H))
        p.emitMulConst(6,6,2);p.emitSigmoid(6,6);p.emitMulConst(6,6,2);p.emitAddConst(6,6,-1)
        p.emitMulConst(7,27,float32(PI));p.emitMul(8,7,6);p.emitUAtomicAdd(8,7,uint32(o.ddt))
        p.emitMul(8,6,6);p.emitLoadConstF32(9,1);p.emitSub(8,9,8)
        p.emitMul(7,7,16);p.emitMul(7,7,8);p.emitUAtomicAdd(7,9,uint32(o.dp+2*a.C+2*a.N+3*a.H))
      p.emitUStoreF32(2,2,uint32(o.dk));p.emitUStoreF32(3,2,uint32(o.dk+1))
      p.emitUStoreF32(4,2,uint32(o.dq));p.emitUStoreF32(5,2,uint32(o.dq+1))
    p.emitBarrierTg()
    p.emitUAddK(2,2,cast[uint32](-a.H*a.N));p.emitUAddK(4,4,cast[uint32](-a.P))
    p.emitUAddK(5,5,cast[uint32](-a.H*3));p.emitUAddK(6,6,cast[uint32](-a.H*a.R));p.emitUAddK(7,7,cast[uint32](-a.H))
  p.emitHalt();p

proc finishDt*(a:M3Arena;o:M3ReverseOffsets;dtBias:int):VmProgram =
  var p:VmProgram
  p.rowIndex(a.H);p.emitUMulK(3,1,uint32(a.P));p.emitUAdd(3,3,2)
  p.emitULoadF32(4,3,uint32(o.proj+2*a.C+2*a.N));p.emitULoadConstF(5,2,uint32(dtBias))
  p.emitAdd(4,4,5);p.emitSigmoid(4,4);p.emitULoadF32(5,0,uint32(o.ddt));p.emitMul(4,4,5)
  p.emitUStoreF32(4,3,uint32(o.dp+2*a.C+2*a.N));p.emitHalt();p

proc normBackward*(a:M3Arena;o:M3ReverseOffsets;weight,which:int;eps:float32):VmProgram =
  var p:VmProgram
  p.emitUTgid(0);p.emitULid(1);p.emitUToR(1,1)
  p.emitUMulK(2,0,uint32(a.P));p.emitUAdd(2,2,1)
  p.emitUMulK(3,0,uint32(a.H*a.N));p.emitUAdd(3,3,1)
  p.emitLoadConstF32(2,0);p.emitLoadConstF32(3,0);p.emitLoadConstF32(4,0)
  p.emitLoadConstF32(5,float32(a.N));p.emitCmpLt(5,1,5)
  sessionVmIfGpu(p,5):
    p.emitULoadF32(2,2,uint32(o.proj+2*a.C+which*a.N))
    p.emitULoadConstF(3,1,uint32(weight))
    p.emitUAddK(4,3,0);p.emitLoadConstF32(8,float32(a.H))
    sessionVmForGpu(p,8):
      p.emitULoadF32(6,4,uint32(if which==0:o.dk else:o.dq));p.emitAdd(4,4,6)
      p.emitUAddK(4,4,uint32(a.N))
  p.emitMul(6,2,2);p.emitTgStoreF32(6,1,0);p.emitBarrierTg();p.emitTgTreeReduce(1,256,7,8,9,10,11)
  p.emitLoadConstF32(7,0);p.emitTgLoadF32(12,7,0);p.emitMulConst(12,12,1'f32/float32(a.N))
  p.emitAddConst(12,12,eps);p.emitRsqrt(12,12);p.emitBarrierTg()
  p.emitMul(6,4,3);p.emitMul(6,6,2);p.emitTgStoreF32(6,1,0);p.emitBarrierTg();p.emitTgTreeReduce(1,256,7,8,9,10,11)
  p.emitLoadConstF32(7,0);p.emitTgLoadF32(13,7,0)
  sessionVmIfGpu(p,5):
    p.emitMul(6,4,3);p.emitMul(6,6,12)
    p.emitMul(13,13,12);p.emitMul(13,13,12);p.emitMul(13,13,12);p.emitMulConst(13,13,1'f32/float32(a.N))
    p.emitMul(13,13,2);p.emitSub(6,6,13);p.emitUStoreF32(6,2,uint32(o.dp+2*a.C+which*a.N))
  p.emitHalt();p
