## Small math check: full sequence vs cached decode and projected finite differences.
import std/[math, random, os, tables]
import nimformer, backend, mamba3_real, m3_block_kernel
import test_nimformer
randomize(17)
let cpu=newBackend("cpu")
var m=newMamba3Real(8,16,2,8,1)
var x=newTensor(@[2,4,8])
for i in 0..<x.data.len: x.data[i]=sin(float32(i)*0.31'f32)*0.4'f32
let fw=forwardMamba3Real(m,x,cpu)
for b in 0..<2:
  var st=newMamba3State(m)
  for t in 0..<4:
    let step=forwardStepMamba3Real(m,Tensor(shape: @[1,1,8],data:x.data[(b*4+t)*8..<(b*4+t+1)*8]),st,cpu)
    for d in 0..<8: doAssert abs(step.data[d]-fw.y.data[(b*4+t)*8+d])<1e-5'f32
let di=newTensor(fw.cache.inner.shape,1'f32)
let grad=backwardMamba3Core(m,fw.cache,di,cpu)
# Differentiate an input-projection row by perturbing its bias. This covers
# every split, all timesteps, grouped norm and accumulated RoPE.
for row in 0..<m.inProj.outF:
  let saved=m.inProj.bias.data[row]
  m.inProj.bias.data[row]=saved+0.001'f32
  let plus=forwardMamba3Real(m,x,cpu).cache.inner.data
  m.inProj.bias.data[row]=saved-0.001'f32
  let minus=forwardMamba3Real(m,x,cpu).cache.inner.data
  m.inProj.bias.data[row]=saved
  var numerical,analytic=0'f32
  for i in 0..<plus.len: numerical+=(plus[i]-minus[i])/0.002'f32
  for i in 0..<8: analytic+=grad.data[i*m.inProj.outF+row]
  doAssert abs(numerical-analytic)<0.02'f32+0.01'f32*abs(numerical),"gradient row " & $row & " numerical=" & $numerical & " analytic=" & $analytic
# Whole residual block, including pretrained gated MLP and optional Linear++.
let gamma=newTensor(@[8],1'f32)
let attn=newLinearPlusAttention(8,2,false)
for sidecar in [false,true]:
  let blockFw=if sidecar: forwardPBlockKernel(m,attn,x,gamma,1e-5'f32,cpu)
              else: forwardRBlockKernel(m,x,gamma,1e-5'f32,cpu)
  var grads=initTable[string,Tensor]()
  let ones=newTensor(x.shape,1'f32)
  let dx=if sidecar: backwardPBlockKernel(m,attn,x,gamma,ones,blockFw.cache,1e-5'f32,cpu,"block0",grads)
         else: backwardRBlockKernel(m,x,gamma,ones,blockFw.cache,1e-5'f32,cpu,"block0",grads)
  for idx in [0,13,40,63]:
    let orig=x.data[idx]
    x.data[idx]=orig+0.001'f32
    let plus=if sidecar: forwardPBlockKernel(m,attn,x,gamma,1e-5'f32,cpu).y
             else: forwardRBlockKernel(m,x,gamma,1e-5'f32,cpu).y
    x.data[idx]=orig-0.001'f32
    let minus=if sidecar: forwardPBlockKernel(m,attn,x,gamma,1e-5'f32,cpu).y
              else: forwardRBlockKernel(m,x,gamma,1e-5'f32,cpu).y
    x.data[idx]=orig
    var num=0'f32
    for i in 0..<plus.data.len: num+=(plus.data[i]-minus.data[i])/0.002'f32
    doAssert abs(num-dx.data[idx])<0.03'f32+0.01'f32*abs(num),"MLP/sidecar gradient"
if getEnv("MAMBA3_SMOKE_GPU", "0")=="1":
  let gpu=newBackend("metal")
  let actual=forwardMamba3Real(m,x,gpu)
  for i in 0..<actual.cache.inner.data.len:
    doAssert abs(actual.cache.inner.data[i]-fw.cache.inner.data[i])<2e-4'f32
  let gpuGrad=backwardMamba3Core(m,actual.cache,di,gpu)
  for i in 0..<grad.data.len:
    doAssert abs(gpuGrad.data[i]-grad.data[i])<5e-4'f32,"GPU gradient index " & $i & " actual=" & $gpuGrad.data[i] & " reference=" & $grad.data[i]
  echo "GPU forward/reverse recurrence agrees with CPU"
echo "Mamba-3 SISO: sequence/decode and gradients passed"

# Integration of the copied LoRA path with the FP4-KM projection cache.
when defined(mamba3SmokeLora):
  import m3_hybrid, m3_lora, quant, fp4_gemv_gpu, m3_gpu_runtime
  when defined(mamba3SmokeMerge):
    import m3_inplace_lora_merge
  var model=newNimformerModelJamba(32,32,2,4,"RPG",mambaRealDState=8)
  var weights=initTable[string,QuantTensor]()
  var targets:seq[string]
  for name,lin in paramLinears(model):
    if lin[].inF>0 and lin[].outF>0:
      weights[name]=quantizeTensor(lin[].weight.data,lin[].weight.shape,qkFp4KM)
      if name!="outProj.weight": targets.add(name)
  var adapters=attachLoraQuantResident(model,weights,r=2,alpha=4, targetNames=targets)
  for _,ad in adapters.adapters.mpairs:
    for i in 0..<ad.B.data.len: ad.B.data[i]=0.002'f32*sin(float32(i))
  bindQuantBaseToLinears(model,adapters)
  mergeInUploadStreamingJamba(model,adapters,cpu)
  prepareGpuModel(model,adapters,cpu)
  let ids = @[1,3,7,2,4,1,5,6,8,2,7,10,9]
  var expected:seq[Tensor]
  for i,tid in ids: expected.add(forwardStepJambaGpuRetry(model,tid,ids,i,cpu))
  if getEnv("MAMBA3_SMOKE_GPU", "0")=="1":
    let gpu=newBackend("metal")
    prepareGpuModel(model,adapters,gpu)
    for name in targets:
      doAssert name in gFp4CacheGpu and gFp4CacheGpu[name].residentReady
    for i,tid in ids:
      let actual=forwardStepJambaGpuRetry(model,tid,ids,i,gpu)
      for j in 0..<actual.data.len:
        doAssert abs(actual.data[j]-expected[i].data[j])<0.002'f32,"FP4-KM/LoRA decode mismatch"
    echo "Resident FP4-KM + nonzero LoRA decode agrees with CPU"

    resetGpuGenState(model,gpu)
    prefillResident(model,ids[0..<3],gpu)
    prefillResident(model,ids[3..<11],gpu)
    prefillResident(model,ids[11..<12],gpu)
    let after=forwardStepJambaGpuRetry(model,ids[^1],ids,ids.high,gpu)
    for j in 0..<after.data.len:
      doAssert abs(after.data[j]-expected[^1].data[j])<0.003'f32,"resident prefix/continuation/decode mismatch at " & $j & " got=" & $after.data[j] & " expected=" & $expected[^1].data[j]
    echo "Resident RPG prefix and continued prefix agree with CPU"
    echo lpSelfCheck(gpu,1)
    resetGpuGenState(model,gpu)
    prefillResident(model,ids[0..<ids.high],gpu)
    var stats:seq[BlockStat]
    let traced=forwardStepJambaGpuTraceRetry(model,ids[^1],ids,ids.high,gpu,stats)
    doAssert stats.len==3
    for j in 0..<traced.data.len:doAssert abs(traced.data[j]-expected[^1].data[j])<0.003'f32
    echo "GPU block trace preserves decode output"
    when defined(mamba3SmokeMerge):
      let merged=mergeResidentLoraInplace(model,adapters,gpu)
      doAssert merged>0 and adapters.adapters.len==0
      putEnv("BYBY_ONE_DISPATCH","0")
      resetGpuGenState(model,gpu)
      let multi=forwardStepJambaGpuRetry(model,ids[0],ids,0,gpu)
      putEnv("BYBY_ONE_DISPATCH","1");putEnv("BYBY_OD_TG","1")
      resetGpuGenState(model,gpu)
      let persistent=forwardStepJambaGpuRetry(model,ids[0],ids,0,gpu)
      for j in 0..<multi.data.len:doAssert abs(multi.data[j]-persistent.data[j])<0.002'f32
      echo "Bounded GPU merge + persistent stack agree with resident decode"
