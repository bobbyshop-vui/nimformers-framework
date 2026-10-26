import std/[random, math, sequtils]
import nimformer
import backend

randomize(42)

proc randTensor(shape: seq[int]): Tensor =
  result = newTensor(shape)
  for i in 0 ..< result.data.len:
    result.data[i] = (rand(1.0) - 0.5).float32

let ctx = newBackend("cpu")

let B = 2
let T = 6
let C = 8
let nHeads = 2

var attn = newCausalSelfAttention(C, nHeads)
# deterministic weights
for i in 0 ..< attn.qkv.weight.data.len: attn.qkv.weight.data[i] = (rand(1.0)-0.5).float32
for i in 0 ..< attn.qkv.bias.data.len: attn.qkv.bias.data[i] = (rand(1.0)-0.5).float32
for i in 0 ..< attn.proj.weight.data.len: attn.proj.weight.data[i] = (rand(1.0)-0.5).float32
for i in 0 ..< attn.proj.bias.data.len: attn.proj.bias.data[i] = (rand(1.0)-0.5).float32

let x = randTensor(@[B, T, C])
let dOut = randTensor(@[B, T, C])

let outFwd = attn.forward(x, ctx)
echo "forward sum: ", outFwd.data.foldl(a+b, 0.0'f32)
echo "forward[0..5]: ", outFwd.data[0..5]

let bw = attn.backward(x, dOut, ctx)
echo "dX sum: ", bw.dX.data.foldl(a+b, 0.0'f32)
echo "dQkvW sum: ", bw.dQkvW.data.foldl(a+b, 0.0'f32)
echo "dQkvB sum: ", bw.dQkvB.data.foldl(a+b, 0.0'f32)
echo "dProjW sum: ", bw.dProjW.data.foldl(a+b, 0.0'f32)
echo "dProjB sum: ", bw.dProjB.data.foldl(a+b, 0.0'f32)
echo "dX[0..5]: ", bw.dX.data[0..5]
