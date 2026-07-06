# Nimformer Framework

A pure **Nim** port of a small transformer (char-level LM) with real
forward/backward passes running on the **Metal GPU** (macOS), plus a custom
quantization system (int8 / int4 / fp8 / "APF" auto-adaptive per tensor) for
compressing checkpoints.

No dependency on Python, PyTorch, or tinygrad — every tensor op, matmul,
attention, LayerNorm, Adam optimizer, and quantization routine is
hand-written in Nim + Metal Shading Language.

---

## 1. Table of contents

- [2. Project layout](#2-project-layout)
- [3. Requirements & setup](#3-requirements--setup)
- [4. Build](#4-build)
- [5. Using each library module](#5-using-each-library-module)
  - [5.1. `customfloat.nim` — CustomFloat & APF](#51-customfloatnim--customfloat--apf)
  - [5.2. `quant.nim` — Tensor quantization](#52-quantnim--tensor-quantization)
  - [5.3. `metal_ai.nim` — MetalContext & GPU kernels](#53-metal_ainim--metalcontext--gpu-kernels)
  - [5.4. `nimformer.nim` — Model, forward/backward, optimizer](#54-nimformernim--model-forwardbackward-optimizer)
- [6. Full end-to-end example](#6-full-end-to-end-example)
- [7. Using `main.nim` — the CLI training driver](#7-using-mainnim--the-cli-training-driver)
- [8. The `.nimq` file format](#8-the-nimq-file-format)
- [9. Important notes / known issues](#9-important-notes--known-issues)

---

## 2. Project layout

```
customfloat.nim     -- CustomFloat + APF, pure Nim, RUNS ON ANY OS (no GPU needed)
quant.nim            -- Quantization: int8/int4/fp8_e4m3/fp8_e5m2/custom/auto(APF)
                        + reading/writing a compressed state dict to a .nimq binary file
nimformer.nim        -- Tensor + Linear/LayerNorm/Attention/FeedForward/Embedding
                        + TransformerBlock + NimformerModel + ApfAdam optimizer
                        (REAL forward/backward, running on GPU via metal_ai)
metal_bridge.h       -- Generic C header so Nim can call into Metal via {.importc.}
metal_bridge.m       -- Objective-C implementation of the bridge (compiled alongside Nim)
metal_kernels.metal  -- Real Metal kernels: add, matmul, relu/sigmoid/tanh,
                        softmax, layernorm, embedding_lookup
metal_ai.nim         -- MetalContext (device/queue/pipeline cache/buffer pool)
                        + Nim wrappers for ALL the kernels above
test_nimformer.nim            -- CLI driver: tokenizer, data loading, training loop,
                        checkpoint + resume, quantization on save
tokenizer.json       -- Pre-saved byte-level tokenizer (vocab_size=198)
databricks-dolly-15k.jsonl -- Sample dataset for test training (instruction/response format)
Makefile             -- make / make build / make run / make metal / make clean
```

`customfloat.nim`, `quant.nim`, `nimformer.nim`, `metal_ai.nim` are **PURE
LIBRARIES** — no `when isMainModule`, meant only to be `import`ed. `main.nim`
is the actual CLI entry point used to run training.

> ⚠️ **Note on the current contents of the repo:** at the time this README
> was written, `test_nimformer.nim` and `metal_bridge.h` in the uploaded
> folder both **mistakenly contain the same content** (the actual content of
> `main.nim`) instead of their own correct content (`test_nimformer.nim`
> should be a small, CLI-free test file; `metal_bridge.h` should be a plain C
> header). Before pushing/building, make sure these two files are replaced
> with their correct content, otherwise `make build` will fail to compile
> with clang. The Git repo should contain: `main.nim` (correct content),
> `metal_bridge.h` (the C header), and (recommended) a lightweight
> `test_nimformer.nim` that just builds+forwards+backwards+quantizes as a
> quick demo, with no CLI/external data required.

---

## 3. Requirements & setup

- **macOS** with a Metal-capable GPU — required for the GPU part. Only
  `customfloat.nim` is CPU-only and works on Linux/Windows.
- Xcode Command Line Tools (needed for `clang` to compile the Objective-C
  code and link the `Metal`/`Foundation` frameworks):
  ```bash
  xcode-select --install
  ```
- Nim >= 2.0:
  ```bash
  brew install nim
  # or via choosenim: https://github.com/dom96/choosenim
  ```

---

## 4. Build

Using the provided `Makefile`:

```bash
make            # build test_nimformer + metal_ai, then run test_nimformer
make build      # build only, don't run
make cuda-lib "your cuda compiler path" #compile cuda shader
make run        # build + run test_nimformer + the backend is auto
make run-cuda   # run on nvdia gpu
make run-cpu    # run on cpu
make run-metal  # run on metal gpu
make clean      # remove binaries, nimcache/, and any model_*.nimq checkpoints
```
---
## 4.1 Setting the backend

Build flag:

```bash
nim c -d:release -d:backend=cpu test_nimformer.nim
nim c -d:release -d:backend=metal test_nimformer.nim
nim c -d:release -d:withCuda -d:backend=cuda --passL:"-L. -lcudakernels -lcudart -lcublas -lstdc++" test_nimformer.nim
```

In code:

```nim
import backend
let ctx = newBackend("cpu")
let ctx = newBackend("metal")
let ctx = newBackend("cuda")
let ctx = newBackend("auto")
```
## 5. Using each library module

### 5.1. `customfloat.nim` — CustomFloat & APF

`CustomFloat` describes a custom floating-point type with a configurable
number of exponent/mantissa bits (like fp8, fp16, bfloat16... except you
define the bit widths yourself). "APF" (Adaptive Precision Float)
automatically figures out how many bits are actually needed based on the
values in a tensor.

```nim
import customfloat

# Define a custom dtype: 5 exponent bits, 5 mantissa bits (fp11)
let cf = newCustomFloat(exponentBits = 5, mantissaBits = 5, name = "fp11")
echo cf.totalBits    # 11 (1 sign + 5 exp + 5 mant)
echo cf.itemSize     # 2 (bytes needed to store 1 element, rounded up)

# Encode/decode a float32 array
let data = @[1.5'f32, -0.001, 3.14159, 100000.0]
let packed: seq[uint8] = encodeArray(data, cf)
let restored: seq[float32] = decodeArray(packed, cf)

# Ready-made presets (equivalent to FP8_E4M3 ... FP64 in the Python version)
echo FP8_E4M3.totalBits   # 8
echo FP8_E5M2.totalBits   # 8
echo FP16C.totalBits      # 16
echo FP32C.totalBits      # 32
```

**APF — auto-build a dtype for a tensor** (no need to pick exponent/mantissa
bits yourself):

```nim
# Automatically compute the exponent/mantissa bits needed for THIS tensor
let cf2 = buildCustomDtypeForTensor(data)
echo cf2.name   # e.g. "auto_e4m6"

# Or pass both (weight, gradient) — the gradient helps APF figure out how
# many extra mantissa bits are needed so Adam updates don't underflow
let grad = @[0.0001'f32, -0.0002, 0.0003, 0.0001]
let cf3 = buildCustomDtypeForTensor(data, grad)

# Convenience helper combining encode + dtype-build in one call (used in the
# training loop)
let (encoded, cfUsed) = apfCastForTraining(data, grad)
let decoded = apfDecodeForTraining(encoded, cfUsed)
```

APF configuration constants you can tune when calling (all have sensible
defaults):
```nim
const
  APF_DEFAULT_REL_ERROR_TOL = 1e-3   # max acceptable relative error
  APF_MIN_MANTISSA_BITS = 2
  APF_MAX_MANTISSA_BITS = 23
  APF_EXP_MARGIN_BITS = 1
```

### 5.2. `quant.nim` — Tensor quantization

Wraps `customfloat.nim` into a single unified quantization API, supporting
7 kinds via the `QuantKind` enum:

```nim
type QuantKind* = enum
  qkFp32Raw   # no compression — used for bias/LayerNorm (error-sensitive)
  qkInt8      # symmetric int8, scale = max(|x|)/127
  qkInt4      # symmetric int4, 2 values/byte, scale = max(|x|)/7
  qkFp8E4M3   # CustomFloat(4,3)
  qkFp8E5M2   # CustomFloat(5,2)
  qkCustom    # any CustomFloat you declare yourself (any bit width)
  qkAuto      # APF — auto-builds a CustomFloat for EACH tensor
```

Use the general dispatcher API (recommended, instead of calling
`quantizeInt8`/`quantizeInt4`/... individually):

```nim
import quant

let w = @[0.5'f32, -1.2, 3.7, -0.001, 2.2]
let shape = @[5]

# --- int8 ---
let q8 = quantizeTensor(w, shape, qkInt8)
let back8 = dequantizeTensor(q8)

# --- int4 ---
let q4 = quantizeTensor(w, shape, qkInt4)

# --- fp8 (2 variants) ---
let qf8a = quantizeTensor(w, shape, qkFp8E4M3)
let qf8b = quantizeTensor(w, shape, qkFp8E5M2)

# --- custom dtype of your choosing (e.g. fp6: 3 exponent bits + 2 mantissa bits) ---
let myDtype = newCustomFloat(3, 2, "fp6")
let qCustom = quantizeTensor(w, shape, qkCustom, customCf = myDtype)

# --- auto (APF), pass the gradient too if you have it for a better estimate ---
let grad = @[0.001'f32, -0.002, 0.0007, 0.0001, 0.0003]
let qAuto = quantizeTensor(w, shape, qkAuto, gradArr = grad)

# --- no compression (raw fp32), used for bias/gamma/beta ---
let qRaw = quantizeTensor(w, shape, qkFp32Raw)
```

Each result is a `QuantTensor` object; decompress any of them back with a
single `dequantizeTensor(qt)` call for EVERY kind (it dispatches internally
based on `qt.kind`).

**Saving / loading a compressed state dict to a `.nimq` binary file:**

```nim
# arch: 5 integers describing the architecture — a convention you choose,
# e.g. [vocab, embedDim, nHeads, nLayers, ffMult]
let arch = [1000, 64, 4, 2, 4]
var sd: seq[(string, QuantTensor)] = @[
  ("outProj.weight", q8),
  ("outProj.bias", qRaw)
]
saveQuantStateDict("checkpoint.nimq", arch, sd)

let (loadedArch, loadedSd) = loadQuantStateDict("checkpoint.nimq")
```

### 5.3. `metal_ai.nim` — MetalContext & GPU kernels

Create the GPU context once and reuse it for the entire lifetime of the
program:

```nim
import metal_ai

let ctx = newMetalContext()   # raises IOError if the machine has no Metal GPU

# Available kernels — every function takes flat seq[float32] arrays
let a = @[1.0'f32, 2, 3, 4]
let b = @[10.0'f32, 20, 30, 40]
let c = ctx.metalAdd(a, b)                      # elementwise add

let m = ctx.metalMatmul(a, 2, 2, b, 2, 2)       # [M,K] x [K,N] -> [M,N]

let r1 = ctx.metalRelu(a)
let r2 = ctx.metalSigmoid(a)
let r3 = ctx.metalTanh(a)

let sm = ctx.metalSoftmax(a, rows = 1, cols = 4)          # row-wise softmax
let ln = ctx.metalLayernorm(a, gamma = @[1'f32,1,1,1],
                             beta = @[0'f32,0,0,0], rows = 1, cols = 4)

let table = @[0.1'f32, 0.2, 0.3, 0.4, 0.5, 0.6]  # [vocab=3, dim=2]
let ids = @[int32(0), 2]
let emb = ctx.metalEmbeddingLookup(table, vocab = 3, dim = 2, indices = ids)

# Encode/decode CustomFloat directly on the GPU (instead of on the CPU as in customfloat.nim)
import customfloat
let cf = buildCustomDtypeForTensor(a)
let packed = ctx.customfloatEncodeGpu(a, cf)
let restored = ctx.customfloatDecodeGpu(packed, cf)

# Clean up the buffer pool before the program exits (not mandatory, process
# exit reclaims memory too, but tidier if the context lives for a long time)
closeMetalContext(ctx)
```

Internally, `MetalContext` automatically:
- **Caches pipelines** by kernel name (compiles each kernel exactly once).
- **Pools GPU buffers** by size (bytes) to reuse them instead of repeatedly
  allocating/deallocating on every dispatch — this prevents the process's
  RAM footprint from growing gradually with every training step.

You'll rarely need to touch the lower-level C functions
(`mtl_dispatch`, `mtl_command_buffer_*`, `mtl_encoder_*`) directly — they
live inside `metal_ai.nim` for adding new kernels, without needing to touch
`metal_bridge.h/.m`.

### 5.4. `nimformer.nim` — Model, forward/backward, optimizer

**Basic Tensor:**
```nim
import nimformer

var t = newTensor(@[2, 3])              # all-zero tensor, shape [2,3]
var t2 = randnTensor(@[2, 3], scale=0.02'f32)  # random gaussian * scale
```

**Building a full transformer model:**
```nim
import nimformer, metal_ai

let ctx = newMetalContext()

var model = newNimformerModel(
  vocab = 64,      # tokenizer vocabulary size
  embedDim = 32,   # embedding dimension
  nHeads = 4,      # number of attention heads
  nLayers = 2,     # number of transformer blocks
  ffMult = 4       # feed-forward expansion factor (hidden = embedDim * ffMult)
)
```

**Forward — a single sequence or a whole batch:**
```nim
# A single sequence (ids: seq[int])
let logits = model.forward(@[1, 5, 9, 2], ctx)     # shape [T, vocab]

# A batch of B sequences of the same length T (recommended — much faster
# because the GPU gets a single matmul with M=B*T rows instead of B
# sequential calls)
let idsBatch = @[@[1, 5, 9, 2], @[3, 3, 1, 0]]
let logitsBatch = model.forwardBatch(idsBatch, ctx) # shape [B, T, vocab]
```

**Backward — needs the gradient of the loss w.r.t. logits (`dLoss`):**
```nim
# Single sequence:
let grads = model.backward(ids, dLoss, ctx)          # seq[Tensor], one gradient per parameter

# Batch (recommended):
let gradsBatch = model.backwardBatch(idsBatch, dLossBatch, ctx)
```
Order of the returned `grads`: `outProj.dW, outProj.dB`, then each block
(from the LAST block back to the FIRST) in the order
`attn.qkv.{W,B}, attn.proj.{W,B}, ff.fc1.{W,B}, ff.fc2.{W,B},
ln1.{gamma,beta}, ln2.{gamma,beta}`, and finally `embed.weight`.

**ApfAdam optimizer** — regular Adam plus re-quantizing the parameter with
APF every `requantizeEvery` steps:
```nim
var state = newApfAdamState(paramLen = model.outProj.weight.data.len)

# Every training step, for EACH parameter and its corresponding gradient:
let cfUsed = apfAdamStep(model.outProj.weight, gradOutProjW, state,
                          lr = 3e-3'f32, requantizeEvery = 50)
echo cfUsed.name   # the APF dtype just rebuilt for this parameter (if the requantize milestone was hit)
```
`requantizeEvery = 1` means APF re-quantizes the parameter after EVERY Adam
step; set it higher (e.g. 50) to reduce overhead, since building a dtype is
computationally expensive.

---

## 6. Full end-to-end example

```nim
import metal_ai, nimformer, quant, customfloat, math

let ctx = newMetalContext()

# 1. Build a small model
var model = newNimformerModel(vocab = 64, embedDim = 32, nHeads = 4,
                                nLayers = 2, ffMult = 4)

# 2. Forward a fake batch
let idsBatch = @[@[1, 5, 9, 2], @[3, 3, 1, 0]]
let logits = model.forwardBatch(idsBatch, ctx)      # [B=2, T=4, vocab=64]

# 3. Fake a random loss gradient and backward (in practice: cross-entropy)
var dLoss = randnTensor(logits.shape, scale = 0.01'f32)
let grads = model.backwardBatch(idsBatch, dLoss, ctx)

# 4. One Adam step on the first parameter (outProj.weight) as an example
var state = newApfAdamState(model.outProj.weight.data.len)
discard apfAdamStep(model.outProj.weight, grads[0], state, lr = 3e-3'f32)

# 5. Quantize + save (int8, int4, fp8, auto/APF) and reload
let q8 = quantizeTensor(model.outProj.weight.data, model.outProj.weight.shape, qkInt8)
let restored = dequantizeTensor(q8)

var maxErr = 0'f32
for i in 0 ..< restored.len:
  maxErr = max(maxErr, abs(restored[i] - model.outProj.weight.data[i]))
echo "Max error after int8 compression: ", maxErr

closeMetalContext(ctx)
```

---

## 8. The `.nimq` file format

A self-defined binary format, header `"NIMQ1"`, layout:

```
"NIMQ1" (5 bytes)
arch: 5 x int32           -- e.g. [vocab, embedDim, nHeads, nLayers, ffMult]
n_tensors: int32
repeated n_tensors times:
  name: (int32 len) + bytes
  QuantTensor:
    kind: 1 byte (QuantKind)
    ndims: int32 + shape[i]: int32 x ndims
    scale: float32          -- used for int8/int4
    exponentBits: int32, mantissaBits: int32   -- used for fp8/custom/auto
    data_len: int32 + data bytes
```

Convention when used through `main.nim`: **weights** (Linear/Embedding
weight) use whatever dtype you chose via `--quant`; **bias and LayerNorm
(gamma/beta)** always stay `qkFp32Raw` (uncompressed) because they are small
and far more sensitive to error than large weight matrices.

---

## 9. Important notes / known issues

- **`metal_bridge.h`/`test_nimformer.nim` content mix-up** — see the warning
  in [section 2](#2-project-layout). Double-check these two files before
  building.
- **Buffer pool in `metal_ai.nim`**: if you write new kernels/wrappers
  yourself, remember to get buffers via `ctx.poolGet(length)` and return
  them via `releaseBufs(ctx, bufs)` (or `ctx.poolPut(buf)` for a single
  buffer) instead of calling `mtlNewBuffer`/`mtlRelease` directly — otherwise
  the process's RAM will grow gradually with every step as Metal keeps
  allocating new memory regions.
- **`@autoreleasepool`**: every function in `metal_bridge.m` that touches an
  Objective-C object (even ones we don't retain ourselves) should be wrapped
  in `@autoreleasepool { ... }` — a Nim program has no run loop to drain the
  autorelease pool the way a real Cocoa app does.
- **Pipeline cache**: do NOT call `mtl_get_pipeline` directly on every
  dispatch — always go through `ctx.pipeline(name)` (which caches it) to
  avoid recompiling the pipeline over and over.
- **Bias/LayerNorm precision**: always keep these as `qkFp32Raw`, don't
  compress them — error in these small parameters has a disproportionate
  effect on training/inference stability compared to large weight matrices.
- Some parts of the original Python transformer/Adam/attention code still
  had unfinished `PLACEHOLDER_*` sections — the ported version here uses the
  standard, correct math (multi-head causal self-attention, Post-LN,
  standard Adam bias-correction) rather than a line-by-line port of the
  original's unfinished placeholders.
