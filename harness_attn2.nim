import std/[random, math, sequtils]
import nimformer
import backend

proc runCase(B, T, C, nHeads, seed: int) =
  randomize(seed)
  proc randTensor(shape: seq[int]): Tensor =
    result = newTensor(shape)
    for i in 0 ..< result.data.len:
      result.data[i] = (rand(1.0) - 0.5).float32
  let ctx = newBackend("cpu")
  var attn = newCausalSelfAttention(C, nHeads)
  for i in 0 ..< attn.qkv.weight.data.len: attn.qkv.weight.data[i] = (rand(1.0)-0.5).float32
  for i in 0 ..< attn.qkv.bias.data.len: attn.qkv.bias.data[i] = (rand(1.0)-0.5).float32
  for i in 0 ..< attn.proj.weight.data.len: attn.proj.weight.data[i] = (rand(1.0)-0.5).float32
  for i in 0 ..< attn.proj.bias.data.len: attn.proj.bias.data[i] = (rand(1.0)-0.5).float32
  let x = randTensor(@[B, T, C])
  let dOut = randTensor(@[B, T, C])
  let outFwd = attn.forward(x, ctx)
  let bw = attn.backward(x, dOut, ctx)
  echo "B=", B, " T=", T, " C=", C, " nHeads=", nHeads,
       " | fwdSum=", outFwd.data.foldl(a+b, 0.0'f32),
       " dXsum=", bw.dX.data.foldl(a+b, 0.0'f32),
       " dQkvWsum=", bw.dQkvW.data.foldl(a+b, 0.0'f32),
       " dProjWsum=", bw.dProjW.data.foldl(a+b, 0.0'f32)

runCase(1, 1, 4, 2, 1)
runCase(3, 5, 12, 3, 2)
runCase(2, 16, 32, 4, 3)
runCase(1, 7, 8, 1, 4)
