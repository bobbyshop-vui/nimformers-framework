# Python and C ABI

`lib_c_api.nim` exposes the Nimformers core as a stable, C-only shared
library (`libnimformers.dylib` / `libnimformers.so` / `nimformers.dll`).
The Python package in `nimformers/` is a thin `ctypes`/NumPy layer over that
ABI; the same symbols are usable from C, C++, Rust, Go, or any FFI that can
call C.

The ABI version is **2** and is reported by `nimf_abi_version()`.

## Build

```sh
make build-for-python      # checks and builds the shared library
make python-test           # builds, then runs the pytest suite
make python-example        # runs examples/python/basic.py
make c-example             # builds and runs examples/c/smoke.c
```

`pip install .` only installs the Python layer; it does not build the Nim
library.  Run `make build-for-python` first and either copy the resulting
shared library into `nimformers/`, set `NIMFORMERS_LIB=/path/to/libnimformers`,
or keep the repository root on the loader search path.

The output name is platform-specific:

| Platform | Library |
| --- | --- |
| macOS | `libnimformers.dylib` |
| Linux | `libnimformers.so` |
| Windows | `nimformers.dll` |

The Python loader searches, in order:

1. an explicit `library=` argument,
2. `NIMFORMERS_LIB`,
3. the `nimformers/` package directory,
4. the repository root,
5. the current directory and `build/`,
6. the platform dynamic-loader path (`find_library("nimformers")`).

## Quick start

```python
import numpy as np
from nimformers import Context

with Context() as ctx:                     # auto -> Metal or Vulkan
    a = np.random.randn(4, 8).astype(np.float32)
    b = np.random.randn(8, 16).astype(np.float32)

    y = ctx.matmul(a, b)                   # [4, 16]
    p = ctx.softmax(a, axis=-1)            # rows sum to 1
    ln = ctx.layernorm(a, np.ones(8), np.zeros(8))
    q = ctx.quantize(a, "int4_asym")
    w = q.dequantize()
    q.close()
```

`Context` is aliased as `Nimformers` and `nimformers.capi.Nimformers` for
backwards compatibility with the original wrapper.

## Context

| Member | Meaning |
| --- | --- |
| `Context(backend="auto", library=None, forbid_cpu_fallback=None)` | Create a backend context. Backends: `auto`, `metal`, `vulkan`, `cuda`/`opencl` (legacy Vulkan aliases). |
| `ctx.backend` / `ctx.backend_id` | Backend name / numeric id. |
| `ctx.abi_version`, `ctx.version` | ABI and library version. |
| `ctx.sync()` | Best-effort device synchronisation. |
| `ctx.set_cpu_fallback(bool)` | Whether missing GPU kernels may silently fall back to CPU. |
| `ctx.close()` | Release the context (also via `with`). |

All math methods accept array-like values, convert to contiguous `float32`,
and return NumPy arrays. Broadcasting follows NumPy rules for elementwise
operations.

## Math operations

### Elementwise and activations

```python
ctx.add(a, b)            ctx.sub(a, b)           ctx.mul(a, b)        ctx.div(a, b)
ctx.add_scalar(x, 1.0)   ctx.mul_scalar(x, 2.0)  ctx.neg(x)           ctx.pow(x, 2.0)
ctx.relu(x)              ctx.sqrt(x)             ctx.exp(x)           ctx.log(x)
ctx.sigmoid(x)           ctx.tanh(x)
ctx.apflu(x, alpha=0.1, beta=0.1)
ctx.apflu_backward(x, dy, alpha=0.1, beta=0.1)
```

### Matrix multiplication

```python
ctx.matmul(a, b)                    # NumPy batch broadcasting
ctx.matmul_nt(a, b)                 # a @ b.T, b is [..., N, K]
ctx.matmul2(a1, b1, a2, b2)         # one GPU sync, returns (a1@b1, a2@b2)
ctx.matmul_q4(a, packed, scales, zeros,
              k=..., n=..., group_size=0, groups_per_row=1)
ctx.linear_q4(x, quant_tensor)      # kind-aware helper
```

`linear_q4` uses the int4 kernel for `int4_asym` weights and transparently
dequantizes other kinds before multiplying (symmetric int4 uses
two's-complement nibbles and FP4-KM interleaves scale bytes, neither of which
matches the Q4 kernel's `(nibble - zero) * scale` decoding).  For repeated
calls, upload the weight once with `ctx.resident_q4(q)` and use
`resident.linear(x)`.

### Normalisation and attention

```python
ctx.softmax(x, axis=-1)
ctx.layernorm(x, gamma, beta, eps=1e-5, axis=-1)
ctx.add_layernorm(a, b, gamma, beta, eps=1e-5, axis=-1)   # -> (sum, normalized)
ctx.layernorm_backward(dy, x, gamma, beta, eps=1e-5)      # -> (dx, dgamma, dbeta)

ctx.attention(q, k, v, mask=None, scale=None)             # -> (out, scores)
ctx.attention_backward(q, k, v, scores, dy, scale=None)   # -> (dq, dk, dv)
```

`q`, `k`, `v` are `[batch, heads, seq, head_dim]`; `scores` is
`[batch, heads, seq, seq]`. The current Metal/Vulkan fused kernel is
**causal**; `mask` is accepted for ABI compatibility but is not applied by
those kernels.

### Data movement and reductions

```python
ctx.embedding_lookup(table, indices)     # table [vocab, dim]
ctx.gather(flat_source, indices)
ctx.compare(a, b, op="gt")               # gt/lt/eq/ge/le -> bool array
ctx.dropout(x, p=0.5, training=True, seed=0)
ctx.dropout_mask(n, p=0.5, seed=0)
ctx.argmax(x, axis=-1)                   # int64 indices
ctx.argmax_with_values(x, axis=-1)       # -> (values, indices)
ctx.masked_fill(x, mask, value)
ctx.where(cond, a, b)

ctx.sum(x, axis=-1)       ctx.mean(x, axis=-1)      ctx.max(x, axis=-1)
ctx.cumsum(x, axis=-1)    ctx.cumprod(x, axis=-1)
ctx.affine_scan(a, b, axis=-1)           # h[t] = a[t]h[t-1] + b[t], h[-1]=0

ctx.permute(x, dims)                     # dims is the new axis order
ctx.split(x, split_size, axis=0)         # list of chunks
ctx.split_slice(x, axis, offset, size)
ctx.concat([a, b, ...], axis=0)
ctx.stack([a, b, ...], axis=0)
```

## Tensors

`ctx.tensor(value)`, `ctx.zeros(shape, fill=0)`, and
`ctx.randn(shape, scale=1)` create **owned** Nim tensors:

```python
t = ctx.tensor(np.arange(12, dtype=np.float32).reshape(3, 4))
t.shape, t.rank, t.numel
t.reshape(2, 6)
t.fill(0)
t.set_data(np.zeros((2, 6), np.float32))
view = t.numpy(copy=False)       # valid until t.close()
copy = t.numpy()                 # safe copy (default)
clone = t.clone()
t.close()
```

## Quantization

```python
q = ctx.quantize(weight, "int4_asym")       # or fp32/int8/int4/fp8_e4m3/
                                            # fp8_e5m2/custom/auto/fp4_km
q.kind, q.shape, q.numel, q.nbytes
q.group_size, q.cf_bits                     # (exponent_bits, mantissa_bits)
q.data(), q.scales(), q.zeros()             # payload views/copies (read-only
                                            # when the checkpoint is mmapped)
deq = q.dequantize()
deq_t = q.dequantize_transposed()      # 2-D: [cols, rows]
q.close()
```

`fp4_km` requires a 2-D tensor whose last dimension is divisible by 32.
`custom` takes `exponent_bits=` and `mantissa_bits=`.

Custom floats and APF:

```python
raw = ctx.customfloat_encode(x, exponent_bits=4, mantissa_bits=3)
x2 = ctx.customfloat_decode(raw, x.size, 4, 3)

e, m = ctx.apf_build_dtype(x, grad=None, rel_error_tol=1e-3)
raw, (e, m) = ctx.apf_cast_training(x)
x3 = ctx.apf_decode_training(raw, e, m, x.size)
```

## NIMQ state dicts

```python
sd = ctx.load_state_dict("model.nimq", mapped=True)   # mmap payloads
sd = ctx.create_state_dict(arch=(vocab, hidden, ff, layers, heads))
sd["model.layers.0.self_attn.q_proj.weight"] = q
sd.append("model.norm.weight", norm_q)
sd.names, sd.arch, len(sd)
sd.save("model.nimq")
sd.close()
```

`sd[name]` returns a `QuantTensor` handle cached by the state dict. Closing
the state dict closes those handles. `sd.to_dict()` dequantizes everything
(useful for inspection of small checkpoints).

## Micro-VM (micro-kernel dispatch)

The VM is a small register machine over a flat `float32` buffer. `R0` is the
thread index; `U0..U15` are integer address registers.

```python
from nimformers import Op

prog = ctx.vm_program()
prog.load_f32(1, 0, 0)      # R1 = mem[R0]
prog.add(1, 1, 1)           # R1 = R1 + R1
prog.store_f32(1, 0, 0)     # mem[R0] = R1
prog.halt()

data = np.arange(1, 65, dtype=np.float32)
out = ctx.vm_run(prog, data, tg_size=64)
```

| API | Meaning |
| --- | --- |
| `ctx.vm_program()` | New program. |
| `prog.emit(Op.ADD, a, b, c)` | Generic emitter; float immediates use `f32=`. |
| `prog.emit_word(w)` / `prog.set_words(list)` / `prog.words` | Raw bytecode. |
| `prog.load_f32`, `store_f32`, `load_const`, `add`, `mul`, `fma`, `exp`, `sigmoid`, `for_begin`, `if_begin`, `u_*`, ... | Named convenience emitters. |
| `prog.run(ctx, memory, n_threads=None, tg_floats=0, tg_size=0)` | Dispatch; returns the resulting array (updates a contiguous input in place). |
| `prog.save(path)` / `VMProgram.load(lib, path)` | `.bvm` persistence. |
| `VMProgram.merge(lib, *programs)` | Concatenate programs, dropping interior `HALT`s. |
| `prog.lower_msl(name)` / `prog.lower_glsl(name)` | Kernel source. |
| `prog.key(tag)` | Deterministic program key (no GPU session). |

`ctx.micro_vm_dispatch(words, memory, ...)` is the raw-bytecode shortcut.
Threadgroup memory is requested in float slots via `tg_floats`; `tg_size`
forces the threads per threadgroup (0 = backend default).

## Tokenizers

```python
tok = ctx.tokenizer("test/codestral/models/hf")  # tokenizer.json + config
ids = tok.encode("Hello world")
text = tok.decode(ids)
ids = tok.apply_chat_template(user, system)
tok.vocab_size, tok.total_vocab_size, tok.special_ids
tok.close()

sp = ctx.sentencepiece("tokenizer.model")
ids = sp.encode("Hello")
sp.decode(ids)
sp.piece_id("▁Hello")
sp.close()
```

`apply_chat_template` raises if the model has no supported chat template.

## Llama inference

```python
cfg = ctx.llama_config("model/config.json")
with ctx.load_llama("model/config.json", "model.nimq",
                    upload_resident=True) as model:
    tok = ctx.tokenizer("model")
    text = model.generate(tok, "Write a Fibonacci function",
                          max_new_tokens=128, temperature=0.7, top_p=0.9,
                          use_chat_template=False)
```

`upload_resident=True` (the default) is required by the optimized decode
path; the loader pins quantized weights on the GPU once. Loading a multi-GB checkpoint takes
time and RAM. Prompt tokens outside the model vocabulary are rejected with a
clear error.

## C usage

The public header is [`include/nimformers.h`](../include/nimformers.h) and
[`examples/c/smoke.c`](../examples/c/smoke.c) is a minimal consumer:

```sh
cc -Iinclude examples/c/smoke.c -L. -lnimformers \
   -Wl,-rpath,"$PWD" -o /tmp/nimf_smoke
/tmp/nimf_smoke
```

Ownership and error rules are documented at the top of the header. In short:
status `0` is success, `nimf_last_error()` explains failures, and strings
returned by the library (state-dict names, decoded text, lowered source,
program keys) must be released with `nimf_free_string()`.

## Coverage

| Nim module | ABI section |
| --- | --- |
| `backend.nim` (`be*`) | elementwise, matmul, softmax, layernorm, attention, embedding, gather, compare, dropout, argmax, perms/split/concat, APFLU, custom-float, resident weights |
| `tensor_ops.nim` (`be*`) | reductions, cumsum/cumprod, affine scan, dropout, comparisons, masked fill/where |
| `quant.nim` | `nimf_quantize` / `nimf_quant_dequantize`, payload accessors, NIMQ state dicts |
| `customfloat.nim` | encode/decode, APF dtype selection, APF training cast |
| `vendor/bybylang/micro_kernel` | program builder, `.bvm` I/O, lowering, dispatch |
| `test/deepseek-inference/native_hf_tokenizer.nim` | HF tokenizer |
| `test/codestral/sentencepiece_native.nim` | SentencePiece tokenizer |
| `test/deepseek-inference/nim_inference.nim` | Llama config load, weight load, generation |

The layer/model classes in `nimformer.nim` (`Linear`, `LayerNorm`,
`CausalSelfAttention`, `FeedForward`, `TransformerBlock`, `NimformerModel`)
and the Jamba/Mamba layer caches are Nim-side compositions of these kernels;
Python code composes the same primitives (`matmul_nt`, `layernorm`,
`attention`, `linear_q4`, resident weights, ...).  Model-specific
applications (`infer_lora`, Jamba/Hybrid training, Spectron training) remain
command-line programs; their building blocks are available through the APIs
above.

## Threading and lifetime

- One context per thread. GPU calls on the same context must be serialised.
- `nimf_last_error()` is thread-local.
- Tensor/quant/state-dict handles own their data; `numpy(copy=False)` views
  become invalid after `close()`.
- `Prog.run` updates a C-contiguous input in place and also returns it; pass
  a copy if you need the original.

## Tests

The suite lives in [`test/python_test/`](../test/python_test/):

```sh
make python-test          # prints every test case and PASSED/FAILED
make python-test PYTEST_FLAGS=-q   # quiet summary only
```

It covers every op family against NumPy, quantization round trips,
NIMQ save/load (including a real Spectron checkpoint when present), the
micro-VM (run/save/load/lower/merge), resident weights, tokenizers, and an
end-to-end tiny Llama generation from a synthetic checkpoint.
