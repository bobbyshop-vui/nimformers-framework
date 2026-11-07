# FIR script: PyTorch inference coverage

This document lists every PyTorch surface an inference model uses and the state of
`formers.fir` (the stdlib tracer described in [fir-script.md](fir-script.md)) for
each item. The same content is available as data in
`other/formers.n/python/formers/fir/coverage_table.py` (`COVERAGE`, `summary()`,
`items(surface, status)`); the summary and coverage tables below are generated
from it with `python3 -m formers.fir.coverage_table docs/fir-script-coverage.md`,
and a unit test checks that they match.

Reference lists: the Core ATen IR table of PyTorch 2.8 (`torch.compiler_ir`), the
`torch.nn`, `torch.nn.functional` and `Tensor` references, `torch.export`
(dynamic shapes, graph signature, functionalization, `run_decompositions`),
`torch._higher_order_ops`, and the HF transformers modeling, cache and
`modeling_rope_utils` code (`ROPE_INIT_FUNCTIONS`: `linear`, `dynamic`, `yarn`,
`longrope`, `llama3`, `proportional`, plus the implicit `default`).

The full ATen op list (2,570 overloads from PyTorch's `native_functions.yaml`) is cross-checked in [fir-script-aten-coverage.md](fir-script-aten-coverage.md).

## Status

| Status | Meaning |
| --- | --- |
| `supported` | First-class traced operation with its own FIR lowering: a generic loop kernel, a strided view descriptor, an external op (`fir.linear`, `fir.linear_t`, `fir.embedding`) or an identity. |
| `pattern` | Recognised subgraph lowered into a fused micro-kernel by a pattern pass: `attention_decode` (+ RoPE), `conv1d_step`, `state_update_reduce`, `norm` (residual + LayerNorm/RMSNorm). |
| `generic` | No named operation; the documented composition of supported operations traces and exports through the generic kernels. |
| `unsupported` | Cannot be traced or exported; the blocker column names the missing piece. |

The `from_torch` column states whether `fir.from_torch` (torch.fx adapter) accepts
the target: `yes`, `no`, `unsafe` (accepted with wrong semantics), `n/a`. The
adapter resolves FX targets by name against `torch_compat`, so a `yes` follows
the tracer status of the same item.

Statuses were checked against the code: the operation names recorded by
`tensor.py` and `nn.py`, the `torch_compat` namespace (`torch`, `torch.nn`,
`torch.nn.functional`, `transformers` pieces), the `ELEMENTWISE` set and passes
in `graph.py`, the re-fusion passes in `refusion.py`, the matchers in
`patterns.py`, the generators in `codegen.py` and `emitter.py`, the `from_torch`
name resolution in `torch_adapter.py`, the VM mnemonics in `vm.py`, and
the runner's weight loader (`fmLoadSafeF32` accepts `F32` safetensors only;
weight dtypes `f32`/`q4`, buffer dtype `f32`). Attention variants were traced on
the Llama example: QK-norm and a custom `scale` still fuse into
`attention_decode` with RoPE; a scaled position (linear RoPE) fuses attention but
leaves `rotary` as a separate kernel; logit soft-capping falls back to
`matmul`/`softmax` with copy-mode cache writes.

## Summary

| Status | Items |
| --- | --- |
| supported | 185 |
| pattern | 20 |
| generic | 248 |
| unsupported | 106 |
| **total** | **559** |

| Surface | supported | pattern | generic | unsupported | total |
| --- | --- | --- | --- | --- | --- |
| Core ATen: pointwise math (`aten.pointwise`) | 20 | 0 | 16 | 18 | 54 |
| Core ATen: activations (`aten.activation`) | 4 | 0 | 3 | 0 | 7 |
| Core ATen: comparison, logical and bitwise (`aten.comparison_logical`) | 12 | 0 | 11 | 0 | 23 |
| Core ATen: reductions (`aten.reduction`) | 5 | 0 | 11 | 2 | 18 |
| Core ATen: shape and view operations (`aten.shape_view`) | 11 | 0 | 6 | 2 | 19 |
| Core ATen: indexing, gather and scatter (`aten.indexing`) | 2 | 1 | 4 | 8 | 15 |
| Core ATen: linear algebra (`aten.linalg`) | 2 | 0 | 3 | 1 | 6 |
| Core ATen: convolution, pooling, padding and resampling (`aten.conv_pool`) | 0 | 0 | 15 | 3 | 18 |
| Core ATen: normalization (`aten.normalization`) | 0 | 1 | 4 | 0 | 5 |
| Core ATen: sorting, top-k and cumulative (`aten.sort_cumulative`) | 1 | 0 | 0 | 2 | 3 |
| Core ATen: random and dropout (`aten.random`) | 1 | 0 | 0 | 3 | 4 |
| Core ATen: creation, dtype casting and symbolic sizes (`aten.creation_cast`) | 4 | 0 | 6 | 2 | 12 |
| Core ATen: backward operators (training only) (`aten.training`) | 0 | 0 | 0 | 7 | 7 |
| Control-flow and higher-order operators (`aten.higher_order`) | 1 | 0 | 6 | 5 | 12 |
| torch.nn modules (`nn`) | 17 | 3 | 47 | 3 | 70 |
| torch.nn.functional (`functional`) | 10 | 3 | 39 | 6 | 58 |
| Tensor methods and attributes (`tensor_method`) | 41 | 0 | 15 | 6 | 62 |
| torch.* functions (`torch_function`) | 18 | 2 | 15 | 4 | 39 |
| dtype system and promotion (`dtype`) | 8 | 0 | 2 | 9 | 19 |
| Tracing semantics (`tracing`) | 10 | 0 | 5 | 7 | 22 |
| HF transformers: KV cache and state conventions (`hf.cache`) | 2 | 2 | 3 | 6 | 13 |
| HF transformers: rotary embedding variants (`hf.rope`) | 1 | 1 | 7 | 3 | 12 |
| HF transformers: attention variants (`hf.attention`) | 0 | 6 | 8 | 2 | 16 |
| HF transformers: mixture-of-experts routing (`hf.moe`) | 2 | 0 | 3 | 2 | 7 |
| HF transformers: SSM and linear-attention families (`hf.ssm`) | 1 | 1 | 11 | 0 | 13 |
| HF transformers: vision and multimodal front ends (`hf.multimodal`) | 0 | 0 | 5 | 2 | 7 |
| HF transformers: embeddings and output heads (`hf.embedding_head`) | 7 | 0 | 2 | 1 | 10 |
| Sampling heads (`hf.sampling`) | 5 | 0 | 1 | 2 | 8 |

`fir.from_torch` acceptance over all items: n/a 115, no 163, yes 281.

## Tracing semantics compared with PyTorch front ends

| Aspect | torch.jit.trace | torch.jit.script | torch.fx symbolic_trace | torch.export | fir.trace |
| --- | --- | --- | --- | --- | --- |
| Capture | runs with example tensors | compiles Python source | proxies through Python | proxies + fake tensors (strict or non-strict) | proxies through Python, metadata only |
| Shapes | specialized to the example | dynamic | proxy-level, shape-agnostic | static by default; `Dim`, `Dim.AUTO`, `Dim.DYNAMIC`, `Dim.STATIC` with ranges and relations | concrete; `Dim(name, value)` and config symbols resolved at trace time, emitted as `require sym` |
| Python control flow | frozen to the traced path | scripted `if`/`for` | fails on proxy `bool` | specializes; data-dependent branches need `torch.cond` | static branches run; proxy `bool` raises |
| Data-dependent control flow | silently frozen | supported | not captured | `torch.cond`, `while_loop` (higher-order ops) | `where`/masks; `cond` maps to `where` with both branches traced |
| Loops | unrolled | kept | unrolled | unrolled (or `scan`/`while_loop` HOPs) | unrolled; `fir.scan` is a static unroll; kernels have runtime loops over elements |
| Mutation | recorded in place | in place | in place | functionalized; buffer updates become `BUFFER_MUTATION` outputs | SSA; in-place tensor methods rebind the proxy to the out-of-place result; `State.update` writes persistent state, emitted in place when no stale version is read |
| Parameters / buffers / constants | attributes | attributes | `get_attr` | lifted to inputs (`InputSpec` PARAMETER / BUFFER / CONSTANT_TENSOR) | `Parameter` named by checkpoint path (never read); buffers are constants evaluated at construction; constants folded or written by an init node |
| Views and aliasing | aliasing tensors | aliasing | aliasing | functional views (`view`, `slice`, `as_strided`) | strided view descriptors over the base buffer; no writes through views |
| Op set | TorchScript IR | TorchScript IR | Python-level targets | ATen (training IR) → Core ATen via `run_decompositions()` | fixed traced op set (pointwise, reductions, views, matmul/linear, embedding, rotary, norms, conv1d, state read/write); torch functions are compositions; re-fusion and pattern passes produce the fused ops |

## Conventions used by HF transformers

- **KV cache.** `DynamicCache` grows by `torch.cat` and needs dynamic shapes;
  `StaticCache` (and the static layers of the layered cache API) preallocates
  `[batch, kv_heads, max_cache_len, head_dim]` and writes with `index_copy_` at
  `cache_position`. FIR models the static form: `register_state("key_cache",
  (kv_heads, positions, head_dim))` and `update(k, position, dim=1)`, which the
  `attention_decode` pattern turns into a row write at `position`;
  `torch_compat.transformers.StaticCache` has the HF `update(k, v, layer_idx,
  cache_kwargs)` interface over such states. Sliding-window
  layers keep a ring buffer (`slot = position % window`); hybrid caches mix both
  kinds per layer; `MambaCache` holds `conv_states` and `ssm_states`, matched by
  `conv1d_step` and `state_update_reduce`.
- **RoPE.** HF computes `inv_freq` once (`ROPE_INIT_FUNCTIONS[rope_type]`), then
  `cos/sin = (position_ids · inv_freq)` duplicated over both halves and
  `q·cos + rotate_half(q)·sin`, multiplied by `attention_scaling` (YaRN/LongRoPE).
  `fir.rotary_embedding` implements this for the default `inv_freq = base^(-2i/D)`;
  the HF composition itself (an `inv_freq` buffer, `cos`/`sin`, `rotate_half`) is
  re-fused into `rotary` by `refuse_rotary` when the angles equal the default
  table. Other types stay composed `sin`/`cos` over a constant `inv_freq` table.
- **Attention.** `eager`, `sdpa` and `flash_attention_2` back ends compute the same
  function; `repeat_kv` (expand + reshape) implements GQA; masks come from the mask
  utilities as additive float or boolean tensors combined with `&`. The eager
  form (matmul, additive `min * (arange > cache_position)` mask, float32 softmax)
  and the sdpa form both lower to `attention_decode`.
- **MoE.** Router logits → `softmax` (or `sigmoid` + bias for DeepSeek-V3) →
  `topk` → optional renormalization → per-expert dispatch with `index_add_` /
  `scatter`. Only a dense masked evaluation is expressible today.
- **Sampling.** HF `generate` applies logits processors (temperature, top-k, top-p,
  min-p, repetition penalty) outside the model. The formers.n host sampler covers
  greedy, temperature, min-p, top-p, repetition penalty and banned ids.

## Coverage tables

### Core ATen: pointwise math

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `aten.abs` | supported | pointwise abs (cmplt + neg) | – | yes |
| `aten.acos` | unsupported | – | no VM composition: needs an atan/asin polynomial op | no |
| `aten.acosh` | generic | log(x + sqrt(x*x - 1)) | log is accurate for positive inputs only | no |
| `aten.add.Scalar` | supported | pointwise add, fused by fuse_pointwise | – | yes |
| `aten.add.Tensor` | supported | pointwise add with broadcasting, fused by fuse_pointwise | – | yes |
| `aten.asin` | unsupported | – | no VM composition: needs an atan/asin polynomial op | no |
| `aten.asinh` | generic | log(x + sqrt(x*x + 1)) | loses accuracy for large negative x (log of a small value) | no |
| `aten.atan` | unsupported | – | no VM composition: needs an atan polynomial op | no |
| `aten.atan2` | unsupported | – | no VM composition: needs atan plus quadrant selection | no |
| `aten.atan2.out` | unsupported | – | out= variant; atan2 itself is missing | no |
| `aten.atanh` | generic | 0.5 * log((1 + x) / (1 - x)) | – | no |
| `aten.ceil` | unsupported | – | constant tensors fold at trace time; runtime values need a rounding op (raises Unsupported) | yes |
| `aten.clamp` | generic | minimum(maximum(x, lo), hi) (torch.clamp / Tensor.clamp) | – | yes |
| `aten.clamp.Tensor` | generic | minimum(maximum(x, lo), hi) with tensor bounds | – | yes |
| `aten.cos` | supported | pointwise cos (range reduction to [-pi, pi] + Taylor series) | accuracy drops for very large arguments | yes |
| `aten.cosh` | generic | (exp(x) + exp(-x)) / 2 | – | yes |
| `aten.div.Scalar` | supported | pointwise div | – | yes |
| `aten.div.Scalar_mode` | unsupported | – | rounding_mode floor/trunc: constant tensors fold at trace time; runtime values need a rounding op (raises Unsupported) | yes |
| `aten.div.Tensor` | supported | pointwise div with broadcasting | – | yes |
| `aten.div.Tensor_mode` | unsupported | – | rounding_mode floor/trunc: constant tensors fold at trace time; runtime values need a rounding op (raises Unsupported) | yes |
| `aten.erf` | unsupported | – | constant tensors fold at trace time; exact gelu carries its own erf; no standalone runtime erf op | yes |
| `aten.exp` | supported | pointwise exp (VM exp) | – | yes |
| `aten.expm1` | generic | exp(x) - 1 | cancellation near 0 | yes |
| `aten.floor` | unsupported | – | constant tensors fold at trace time; runtime values need a rounding op (raises Unsupported) | yes |
| `aten.fmod.Scalar` | unsupported | – | constant tensors fold at trace time; runtime values need a rounding op (raises Unsupported) | yes |
| `aten.fmod.Tensor` | unsupported | – | constant tensors fold at trace time; runtime values need a rounding op (raises Unsupported) | yes |
| `aten.log` | supported | pointwise log (eight square roots + atanh series) | positive inputs only | yes |
| `aten.log10` | generic | log(x) * (1 / ln 10) | positive inputs only | yes |
| `aten.log1p` | generic | log(1 + x) | cancellation near 0 | yes |
| `aten.log2` | generic | log(x) * (1 / ln 2) | positive inputs only | yes |
| `aten.maximum` | supported | pointwise maximum (cmplt + select) | – | yes |
| `aten.minimum` | supported | pointwise minimum (cmplt + select) | – | yes |
| `aten.mul.Scalar` | supported | pointwise mul | – | yes |
| `aten.mul.Tensor` | supported | pointwise mul with broadcasting | – | yes |
| `aten.neg` | supported | pointwise neg | – | yes |
| `aten.pow.Scalar` | generic | exp(x * log(base)) | positive scalar base only | no |
| `aten.pow.Tensor_Scalar` | supported | repeated multiplication (+ reciprocal); 0.5 -> sqrt, -0.5 -> rsqrt | other non-integer constant exponents fail lowering; integer exponents in [-32, 32] | yes |
| `aten.pow.Tensor_Tensor` | generic | exp(b * log(a)) | positive base only | yes |
| `aten.reciprocal` | generic | 1 / x | – | yes |
| `aten.remainder.Scalar` | unsupported | – | constant tensors fold at trace time; runtime values need a rounding op (raises Unsupported) (ring-buffer slot = position % window) | no |
| `aten.remainder.Tensor` | unsupported | – | constant tensors fold at trace time; runtime values need a rounding op (raises Unsupported) | yes |
| `aten.round` | unsupported | – | constant tensors fold at trace time; runtime values need a rounding op (raises Unsupported) | yes |
| `aten.rsqrt` | supported | pointwise rsqrt (VM rsqrt) | – | yes |
| `aten.sign` | generic | where(x > 0, 1, 0) - where(x < 0, 1, 0) | – | yes |
| `aten.sin` | supported | pointwise sin (range reduction + Taylor series) | accuracy drops for very large arguments | yes |
| `aten.sinh` | generic | (exp(x) - exp(-x)) / 2 | – | yes |
| `aten.sqrt` | supported | pointwise sqrt (1 / rsqrt) | – | yes |
| `aten.sub.Scalar` | supported | pointwise sub | – | yes |
| `aten.sub.Tensor` | supported | pointwise sub with broadcasting | – | yes |
| `aten.tan` | generic | sin(x) / cos(x) | – | yes |
| `aten.trunc` | unsupported | – | constant tensors fold at trace time; runtime values need a rounding op (raises Unsupported) | yes |
| `aten.where.self` | supported | pointwise where (copy b, ifbegin copy a) | – | yes |
| `aten.isinf` | unsupported | – | no non-finite semantics; VM constants must be finite | no |
| `aten.isnan` | unsupported | – | no NaN test; cmplt-based ne returns 0 for NaN | no |

### Core ATen: activations

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `aten.gelu` | supported | approximate='none' -> erf form; 'tanh' -> gelu_new | – | yes |
| `aten.relu` | supported | pointwise relu | – | yes |
| `aten.sigmoid` | supported | pointwise sigmoid (VM sigmoid) | – | yes |
| `aten.tanh` | supported | 2*sigmoid(2x) - 1 | – | yes |
| `aten.hardtanh` | generic | minimum(maximum(x, lo), hi) | – | yes |
| `aten.leaky_relu` | generic | where(x > 0, x, slope * x) | – | yes |
| `aten.elu` | generic | where(x > 0, x, alpha * (exp(x) - 1)) | – | yes |

### Core ATen: comparison, logical and bitwise

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `aten.eq.Scalar` | supported | pointwise eq (two cmplt), bool stored as 0/1 f32 | – | yes |
| `aten.eq.Tensor` | supported | pointwise eq, bool as 0/1 f32 | – | yes |
| `aten.ne.Scalar` | supported | pointwise ne, bool as 0/1 f32 | – | yes |
| `aten.ne.Tensor` | supported | pointwise ne, bool as 0/1 f32 | – | yes |
| `aten.lt.Scalar` | supported | cmplt | – | yes |
| `aten.lt.Tensor` | supported | cmplt | – | yes |
| `aten.le.Scalar` | supported | 1 - cmplt(b, a) | – | yes |
| `aten.le.Tensor` | supported | 1 - cmplt(b, a) | – | yes |
| `aten.gt.Scalar` | supported | cmplt(b, a) | – | yes |
| `aten.gt.Tensor` | supported | cmplt(b, a) | – | yes |
| `aten.ge.Scalar` | supported | 1 - cmplt(a, b) | – | yes |
| `aten.ge.Tensor` | supported | 1 - cmplt(a, b) | – | yes |
| `aten.logical_and` | generic | a * b on 0/1 masks | – | yes |
| `aten.logical_or` | generic | maximum(a, b) on 0/1 masks | – | yes |
| `aten.logical_not` | generic | 1 - a on 0/1 masks | – | yes |
| `aten.logical_xor` | generic | a != b on 0/1 masks | – | yes |
| `aten.bitwise_and.Scalar` | generic | bool masks: mul on 0/1 values (& \| ^ ~ operators); integer constants fold | runtime integer tensors: bool masks only | yes |
| `aten.bitwise_and.Tensor` | generic | bool masks: mul on 0/1 values (& \| ^ ~ operators); integer constants fold | runtime integer tensors: bool masks only | yes |
| `aten.bitwise_or.Scalar` | generic | bool masks: maximum on 0/1 values (& \| ^ ~ operators); integer constants fold | runtime integer tensors: bool masks only | yes |
| `aten.bitwise_or.Tensor` | generic | bool masks: maximum on 0/1 values (& \| ^ ~ operators); integer constants fold | runtime integer tensors: bool masks only | yes |
| `aten.bitwise_xor.Scalar` | generic | bool masks: ne on 0/1 values (& \| ^ ~ operators); integer constants fold | runtime integer tensors: bool masks only | yes |
| `aten.bitwise_xor.Tensor` | generic | bool masks: ne on 0/1 values (& \| ^ ~ operators); integer constants fold | runtime integer tensors: bool masks only | yes |
| `aten.bitwise_not` | generic | bool masks: 1 - x on 0/1 values (& \| ^ ~ operators); integer constants fold | runtime integer tensors: bool masks only | yes |

### Core ATen: reductions

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `aten.sum.dim_IntList` | supported | reduction kernel: lane per kept row, loops over reduced dims; elementwise producer fused | – | yes |
| `aten.mean` | supported | reduction kernel (sum / count) | – | yes |
| `aten.mean.dim` | supported | reduction kernel (sum / count) | – | yes |
| `aten.amax` | supported | fir max reduction (values) | – | yes |
| `aten.amin` | generic | -max(-x) | – | yes |
| `aten.max.dim` | generic | values via fir max; indices via where(x == max, -arange, -n) + max (first index) | – | yes |
| `aten.min.dim` | generic | values via -max(-x); indices as for max.dim | – | yes |
| `aten.argmax` | generic | where(x == max, -arange(n), -n) + max: first index on ties, int64 tag | indices are f32 values (exact below 2**24) | yes |
| `aten.argmin` | generic | where(x == max, -arange(n), -n) + max: first index on ties, int64 tag | indices are f32 values (exact below 2**24) | yes |
| `aten.any` | generic | max over 0/1 mask | – | yes |
| `aten.any.dim` | generic | max over 0/1 mask along dim | – | yes |
| `aten.any.dims` | generic | max over 0/1 mask along dims | – | yes |
| `aten.prod` | unsupported | – | constant tensors fold at trace time; no runtime prod reduction kind | yes |
| `aten.prod.dim_int` | unsupported | – | constant tensors fold at trace time; no runtime prod reduction kind | yes |
| `aten.var.correction` | generic | mean((x - mean(x))**2) * n / (n - correction) | – | yes |
| `aten.var.dim` | generic | mean((x - mean(x))**2) * n / (n - correction) | – | yes |
| `aten._log_softmax` | generic | x - max - log(sum(exp(x - max))) | – | yes |
| `aten._softmax` | supported | softmax kernel (max, exp/sum, normalize); feeds attention_decode | – | yes |

### Core ATen: shape and view operations

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `aten.view` | supported | reshape -> strided view descriptor, copy kernel when not strided | – | yes |
| `aten.permute` | supported | strided view descriptor | – | yes |
| `aten.expand` | supported | broadcast view (stride 0) | – | yes |
| `aten.unsqueeze` | supported | reshape view | – | yes |
| `aten.squeeze.dim` | supported | reshape view | – | yes |
| `aten.squeeze.dims` | supported | reshape view (tuple dims) | – | yes |
| `aten.slice.Tensor` | supported | slice view (any step, negative steps included) | empty slices rejected | yes |
| `aten.select.int` | supported | slice view with an int index | – | yes |
| `aten.split_with_sizes` | supported | split -> slice views | – | yes |
| `aten.cat` | supported | grouped cat as view, irregular cat as copy kernel | – | yes |
| `aten.alias` | supported | identity | – | yes |
| `aten.clone` | generic | identity (SSA values are immutable) | – | yes |
| `aten.copy` | generic | expand(src) (functional copy_; the proxy is rebound) | – | no |
| `aten.as_strided` | unsupported | – | views are composed descriptors; no raw size/stride API | no |
| `aten.diagonal` | generic | x.reshape(-1)[::n + 1] for square matrices | – | yes |
| `aten.flip` | generic | negative-step slice x[..., ::-1] | – | yes |
| `aten.repeat` | generic | unsqueeze + expand + reshape (Tensor.repeat / tile) | – | yes |
| `aten.constant_pad_nd` | generic | cat with zeros/full constants | no named op | no |
| `aten.resize_` | unsupported | – | in-place resize; mutation of non-state tensors | no |

### Core ATen: indexing, gather and scatter

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `aten.embedding` | supported | ext fir.embedding for the token/position role, gather kernel otherwise | – | yes |
| `aten._embedding_bag` | generic | index_select + sum/mean | – | no |
| `aten.index_select` | supported | dynamic-offset gather, one lane per index | indices stored as f32 (exact below 2**24), no bounds check | yes |
| `aten.index.Tensor` | generic | index_select for one index tensor; constant bool masks become a static index list | several index tensors and runtime bool masks unsupported | no |
| `aten.gather` | unsupported | – | constant index tensors only (index_select of a flattened view); runtime per-element gather missing | yes |
| `aten.index_put` | pattern | State.update(value, index, dim) -> indexed state write / attention_decode cache write | only one complete slice of a persistent state at a scalar index | no |
| `aten.select_scatter` | generic | where(arange(n) == i, value, x); state slice write for caches | – | no |
| `aten.slice_scatter` | generic | where(mask of the slice, value, x) | – | no |
| `aten.scatter.src` | unsupported | – | scatter with dynamic indices missing | no |
| `aten.scatter.value` | unsupported | – | scatter with dynamic indices missing | no |
| `aten.scatter_add` | unsupported | – | scatter with dynamic indices missing (MoE combine, repetition penalty) | no |
| `aten.scatter_reduce.two` | unsupported | – | scatter with dynamic indices missing | no |
| `aten.masked_scatter` | unsupported | – | needs a prefix-sum over the mask and data-dependent offsets | no |
| `aten.nonzero` | unsupported | – | data-dependent output shape: constant tensors only | yes |
| `aten._local_scalar_dense` | unsupported | – | item() on constant tensors only (data-dependent Python value otherwise) | no |

### Core ATen: linear algebra

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `aten.mm` | supported | single row: ext fir.linear/linear_t; otherwise script.matmul | – | yes |
| `aten.bmm` | supported | script.matmul (lane over the largest output dim) | – | yes |
| `aten.addmm` | generic | matmul + add (bias fused into the following pointwise kernel) | – | yes |
| `aten._cdist_forward` | generic | sqrt(sum((a[:, None] - b[None]) ** 2, -1)) for p = 2 | – | no |
| `aten._pdist_forward` | generic | cdist + index_select of a constant upper-triangle index | – | no |
| `aten._fft_r2c` | unsupported | – | no complex dtype | no |

### Core ATen: convolution, pooling, padding and resampling

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `aten.convolution` | unsupported | 1-D with zero padding and stride: F.conv1d over the causal conv1d kernel (cat zeros + strided slice) | dilation, 2-D/3-D and transposed convolution missing | no |
| `aten.avg_pool1d` | generic | reshape into windows + mean | windows must tile the input (stride == kernel, no padding) | no |
| `aten.avg_pool2d` | generic | reshape into windows + mean | windows must tile the input | no |
| `aten.avg_pool3d` | generic | reshape into windows + mean | windows must tile the input | no |
| `aten.adaptive_avg_pool1d` | generic | reshape + mean when the size divides | non-dividing sizes unsupported | no |
| `aten._adaptive_avg_pool2d` | generic | reshape + mean when the size divides | non-dividing sizes unsupported | no |
| `aten._adaptive_avg_pool3d` | generic | reshape + mean when the size divides | non-dividing sizes unsupported | no |
| `aten.max_pool2d_with_indices` | generic | values: reshape into windows + max | indices output unsupported; overlapping windows unsupported | no |
| `aten.max_pool3d_with_indices` | generic | values: reshape into windows + max | indices output unsupported | no |
| `aten.col2im` | unsupported | – | overlapping scatter-add | no |
| `aten.grid_sampler_2d` | unsupported | – | data-dependent gather with floor | no |
| `aten.upsample_nearest2d.vec` | generic | unsqueeze + expand + reshape for integer scales | – | no |
| `aten.upsample_bilinear2d.vec` | generic | matmul with constant interpolation matrices (fixed sizes) | – | no |
| `aten.reflection_pad1d` | generic | negative-step slices + cat | – | no |
| `aten.reflection_pad2d` | generic | negative-step slices + cat | – | no |
| `aten.reflection_pad3d` | generic | negative-step slices + cat | – | no |
| `aten.replication_pad2d` | generic | edge slices + expand + cat | – | no |
| `aten.replication_pad3d` | generic | edge slices + expand + cat | – | no |

### Core ATen: normalization

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `aten.native_layer_norm` | pattern | fir.LayerNorm -> layernorm kernel; residual producer fused (script.add_layernorm) | mean/rstd outputs not produced | no |
| `aten.native_group_norm` | generic | reshape to groups + mean/var + pointwise affine | – | no |
| `aten._native_batch_norm_legit_no_training` | generic | (x - running_mean) * rsqrt(running_var + eps) * w + b | running stats are parameters | no |
| `aten._native_batch_norm_legit` | generic | batch statistics via mean/var | – | no |
| `aten._native_batch_norm_legit.no_stats` | generic | batch statistics via mean/var | – | no |

### Core ATen: sorting, top-k and cumulative

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `aten.cumsum` | supported | cumsum kernel: lane per row, running sum | – | yes |
| `aten.sort` | unsupported | – | no sort kernel; needs integer index outputs | no |
| `aten.topk` | unsupported | – | no top-k kernel; needs integer index outputs (MoE routing, top-k sampling) | no |

### Core ATen: random and dropout

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `aten.native_dropout` | supported | identity at inference (fir.Dropout) | – | no |
| `aten.rand` | unsupported | – | no RNG state or VM random instruction | no |
| `aten.randn` | unsupported | – | no RNG | no |
| `aten.randperm` | unsupported | – | no RNG, integer output | no |

### Core ATen: creation, dtype casting and symbolic sizes

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `aten.arange.start_step` | supported | torch-order arange(start, end, step, dtype); affine constants become loop counters | – | yes |
| `aten.full` | generic | constant (torch.full / full_like / tensor) | – | yes |
| `aten.full_like` | generic | constant (torch.full / full_like / tensor) | – | yes |
| `aten.scalar_tensor` | generic | constant (torch.full / full_like / tensor) | – | yes |
| `aten.empty.memory_format` | generic | zeros constant (torch.empty) | – | yes |
| `aten.empty_strided` | generic | fir.zeros(shape) | strides ignored | no |
| `aten.fill.Scalar` | generic | full_like (fill_ rebinds the proxy) | – | no |
| `aten._to_copy` | supported | cast: logical dtype tag over f32 values (torch.to / float / half / long / bool) | values stay f32 (no bf16/f16 rounding); float->int of runtime values only for token/position inputs | yes |
| `aten.sym_size.int` | supported | static Python int from the traced shape | – | no |
| `aten.sym_numel` | supported | static Python int (numel()) | – | no |
| `aten.sym_stride.int` | unsupported | – | no stride API on proxies | no |
| `aten.sym_storage_offset` | unsupported | – | no storage API on proxies | no |

### Core ATen: backward operators (training only)

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `aten.convolution_backward` | unsupported | – | autograd is out of scope for inference export | no |
| `aten.embedding_dense_backward` | unsupported | – | autograd is out of scope | no |
| `aten.native_layer_norm_backward` | unsupported | – | autograd is out of scope | no |
| `aten.native_group_norm_backward` | unsupported | – | autograd is out of scope | no |
| `aten.avg_pool2d_backward` | unsupported | – | autograd is out of scope | no |
| `aten._adaptive_avg_pool2d_backward` | unsupported | – | autograd is out of scope | no |
| `aten.max_pool2d_with_indices_backward` | unsupported | – | autograd is out of scope | no |

### Control-flow and higher-order operators

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `python.static_branch` | supported | Python if/for over static values runs during tracing | – | n/a |
| `python.data_dependent_branch` | unsupported | – | ProxyTensor.__bool__ raises; use where or masks | n/a |
| `torch.cond` | generic | where(pred, true_branch, false_branch) with both branches traced | both branches are evaluated; branches with effects (state writes) unsupported | no |
| `torch._higher_order_ops.while_loop` | unsupported | – | no runtime loop node; data-dependent trip count | no |
| `torch._higher_order_ops.scan` | generic | fir.scan: static unrolled scan over a dimension | unrolled per element; no runtime loop node | no |
| `torch._higher_order_ops.associative_scan` | generic | cumsum for add combine; fir.scan for others | other combine functions unroll | no |
| `torch._higher_order_ops.map` | generic | static Python loop + cat | unrolled | no |
| `torch._higher_order_ops.flex_attention` | generic | score_mod / mask_mod written as pointwise ops over q @ k^T | generic matmul/softmax kernels; no fused kernel | no |
| `torch._higher_order_ops.invoke_subgraph` | generic | subgraph inlined by the Python call | – | no |
| `torch._higher_order_ops.auto_functionalized` | unsupported | – | custom mutating ops are not ingested by from_torch | no |
| `torch._higher_order_ops.with_effects` | unsupported | – | effect tokens are not modelled | no |
| `torch._higher_order_ops.run_with_rng_state` | unsupported | – | no RNG | no |

### torch.nn modules

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `nn.Linear` | supported | fir.Linear: single row -> ext fir.linear (io, f32/q4) / fir.linear_t (oi, f32); batched -> script.linear | q4 only for io layout | yes |
| `nn.Bilinear` | generic | einsum written as matmuls | – | no |
| `nn.Identity` | supported | identity | – | yes |
| `nn.Embedding` | supported | fir.Embedding: ext fir.embedding for token/position, gather otherwise; project() for tied heads | max_norm unsupported | yes |
| `nn.EmbeddingBag` | generic | index_select + sum/mean | – | no |
| `nn.LayerNorm` | pattern | fir.LayerNorm -> layernorm / add_layernorm kernels | – | yes |
| `nn.RMSNorm` | pattern | fir.RMSNorm -> rmsnorm / add_rmsnorm kernels | – | yes |
| `nn.GroupNorm` | generic | reshape + mean/var + affine | – | no |
| `nn.BatchNorm1d` | generic | eval-mode affine with running statistics | – | no |
| `nn.BatchNorm2d` | generic | eval-mode affine with running statistics | – | no |
| `nn.InstanceNorm1d` | generic | mean/var over the length dim | – | no |
| `nn.Conv1d` | pattern | fir.Conv1d causal kernel; torch_compat.nn.Conv1d (padding, stride) via F.conv1d; conv1d_step pattern with state | dilation != 1 and non-zero padding modes unsupported | yes |
| `nn.Conv2d` | generic | patch embedding (kernel == stride, no padding): reshape/permute + matmul | general Conv2d unsupported | no |
| `nn.Conv3d` | generic | patch embedding (kernel == stride): reshape/permute + matmul | general Conv3d unsupported | no |
| `nn.ConvTranspose1d` | unsupported | – | transposed convolution missing | no |
| `nn.ConvTranspose2d` | unsupported | – | transposed convolution missing | no |
| `nn.MaxPool1d` | generic | reshape into windows + max | tiling windows only | no |
| `nn.MaxPool2d` | generic | reshape into windows + max | tiling windows only | no |
| `nn.AvgPool1d` | generic | reshape into windows + mean | tiling windows only | no |
| `nn.AvgPool2d` | generic | reshape into windows + mean | tiling windows only | no |
| `nn.AdaptiveAvgPool1d` | generic | reshape + mean | dividing sizes only | no |
| `nn.AdaptiveAvgPool2d` | generic | reshape + mean | dividing sizes only | no |
| `nn.Upsample` | generic | expand + reshape (nearest, integer scale) | – | no |
| `nn.PixelShuffle` | generic | reshape + permute | – | no |
| `nn.Unfold` | generic | reshape/permute when kernel == stride | overlapping patches unsupported | no |
| `nn.Flatten` | generic | reshape | – | yes |
| `nn.Unflatten` | generic | reshape | – | no |
| `nn.Dropout` | supported | identity | – | yes |
| `nn.ReLU` | supported | pointwise relu | – | yes |
| `nn.ReLU6` | generic | minimum(relu(x), 6) | – | yes |
| `nn.GELU` | supported | fir.GELU / gelu(approximate) | – | yes |
| `nn.SiLU` | supported | pointwise silu (x * sigmoid) | – | yes |
| `nn.Sigmoid` | supported | pointwise sigmoid | – | yes |
| `nn.Tanh` | supported | 2*sigmoid(2x) - 1 | – | yes |
| `nn.Softplus` | supported | VM softplus; softplus(beta * x) / beta with the threshold switch | – | yes |
| `nn.Mish` | generic | x * tanh(softplus(x)) | – | yes |
| `nn.Hardswish` | generic | x * minimum(maximum(x + 3, 0), 6) / 6 | – | yes |
| `nn.Hardsigmoid` | generic | minimum(maximum(x + 3, 0), 6) / 6 | – | yes |
| `nn.Hardtanh` | generic | minimum(maximum(x, lo), hi) | – | yes |
| `nn.LeakyReLU` | generic | where(x > 0, x, slope * x) | – | yes |
| `nn.PReLU` | generic | where(x > 0, x, weight * x) | – | no |
| `nn.ELU` | generic | where(x > 0, x, alpha * (exp(x) - 1)) | – | yes |
| `nn.SELU` | generic | scale * elu(x, alpha) | – | yes |
| `nn.CELU` | generic | maximum(x, 0) + minimum(alpha * (exp(x / alpha) - 1), 0) | – | no |
| `nn.GLU` | generic | a, b = chunk(2); a * sigmoid(b) | – | no |
| `nn.LogSigmoid` | generic | -softplus(-x) | – | yes |
| `nn.Softsign` | generic | x / (1 + abs(x)) | – | yes |
| `nn.Tanhshrink` | generic | x - tanh(x) | – | yes |
| `nn.Threshold` | generic | where(x > t, x, value) | – | no |
| `nn.Softmax` | supported | softmax kernel | – | yes |
| `nn.LogSoftmax` | generic | x - max - log(sum(exp(x - max))) | – | no |
| `nn.Softmin` | generic | softmax(-x) | – | no |
| `nn.MultiheadAttention` | generic | packed in_proj split + scaled_dot_product_attention; attention_decode when written against a state cache | no fir module; written with F.scaled_dot_product_attention | no |
| `nn.TransformerEncoderLayer` | generic | composition of Linear, LayerNorm, sdpa | bidirectional attention over a sequence: evaluate only (decode ABI) | no |
| `nn.TransformerDecoderLayer` | generic | composition with cross attention | cross-attention memory input has no export role | no |
| `nn.Transformer` | unsupported | – | encoder-decoder export ABI missing | no |
| `nn.RNNCell` | generic | tanh(x W_ih^T + h W_hh^T + b) with State h | – | no |
| `nn.GRUCell` | generic | gates via sigmoid/tanh, State h | – | no |
| `nn.LSTMCell` | generic | gates via sigmoid/tanh, States h and c | – | no |
| `nn.RNN` | generic | cell per time step (decode) or fir.scan (sequence) | multi-layer/bidirectional written by hand | no |
| `nn.GRU` | generic | cell per time step (decode) or fir.scan | – | no |
| `nn.LSTM` | generic | cell per time step (decode) or fir.scan | proj_size written by hand | no |
| `nn.ModuleList` | supported | fir.ModuleList (static stack, repeat layers extraction) | – | yes |
| `nn.Sequential` | supported | fir.Sequential | – | yes |
| `nn.ModuleDict` | supported | fir.nn container (checkpoint paths as in torch) | – | yes |
| `nn.ParameterList` | supported | fir.nn container (checkpoint paths as in torch) | – | yes |
| `nn.ParameterDict` | supported | fir.nn container (checkpoint paths as in torch) | – | yes |
| `nn.Module.register_buffer` | supported | buffer values become graph constants (evaluated at construction); persistent buffers appear in state_dict | persistent buffers are not checkpoint-bound weights | yes |
| `nn.CosineSimilarity` | generic | sum(a*b) * rsqrt(sum(a*a) * sum(b*b)) | – | no |
| `nn.PairwiseDistance` | generic | sqrt(sum((a - b)**2)) | – | no |

### torch.nn.functional

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `F.linear` | supported | x @ w.T + b; refuse_linear lowers a checkpoint weight to ext fir.linear_t | – | yes |
| `F.bilinear` | generic | matmuls | – | yes |
| `F.embedding` | supported | ext fir.embedding for the token/position role, index_select otherwise | – | yes |
| `F.embedding_bag` | generic | index_select + sum/mean | – | no |
| `F.one_hot` | generic | arange(n) == ids.unsqueeze(-1) | – | yes |
| `F.layer_norm` | pattern | layernorm kernel; residual producer fused (add_layernorm) | – | yes |
| `F.rms_norm` | pattern | rmsnorm kernel; residual producer fused (add_rmsnorm) | – | yes |
| `F.group_norm` | generic | reshape + mean/var + affine | – | yes |
| `F.batch_norm` | generic | eval-mode affine | – | yes |
| `F.instance_norm` | generic | mean/var over spatial dims | – | yes |
| `F.normalize` | generic | x * rsqrt(maximum(sum(x*x), eps*eps)) | – | yes |
| `F.softmax` | supported | softmax kernel | – | yes |
| `F.log_softmax` | generic | x - max - log(sum(exp(x - max))) | – | yes |
| `F.softmin` | generic | softmax(-x) | – | yes |
| `F.gelu` | supported | gelu(approximate) | – | yes |
| `F.silu` | supported | pointwise silu | – | yes |
| `F.relu` | supported | pointwise relu | – | yes |
| `F.relu6` | generic | minimum(relu(x), 6) | – | yes |
| `F.leaky_relu` | generic | where(x > 0, x, slope * x) | – | yes |
| `F.elu` | generic | where(x > 0, x, alpha * (exp(x) - 1)) | – | yes |
| `F.selu` | generic | scale * elu | – | yes |
| `F.celu` | generic | elu variant | – | yes |
| `F.softplus` | supported | VM softplus | – | yes |
| `F.mish` | generic | x * tanh(softplus(x)) | – | yes |
| `F.hardswish` | generic | clamp composition | – | yes |
| `F.hardsigmoid` | generic | clamp composition | – | yes |
| `F.hardtanh` | generic | clamp composition | – | yes |
| `F.logsigmoid` | generic | -softplus(-x) | – | yes |
| `F.sigmoid` | supported | pointwise sigmoid | – | yes |
| `F.tanh` | supported | 2*sigmoid(2x) - 1 | – | yes |
| `F.glu` | generic | chunk + sigmoid + mul | – | yes |
| `F.softsign` | generic | x / (1 + abs(x)) | – | yes |
| `F.tanhshrink` | generic | x - tanh(x) | – | yes |
| `F.threshold` | generic | where | – | yes |
| `F.scaled_dot_product_attention` | pattern | attn_mask (bool or additive), is_causal, scale, enable_gqa; attention_decode against a state KV cache | dropout_p > 0 raises (no RNG) | yes |
| `F.multi_head_attention_forward` | generic | split projections + sdpa | – | no |
| `F.conv1d` | generic | zero padding and stride over the causal conv1d kernel (cat zeros + strided slice) | dilation != 1 unsupported | yes |
| `F.conv2d` | generic | patch embedding via reshape/permute + matmul (kernel == stride) | general conv2d unsupported | no |
| `F.conv3d` | generic | patch embedding via reshape/permute + matmul (kernel == stride) | general conv3d unsupported | no |
| `F.conv_transpose1d` | unsupported | – | transposed convolution missing | no |
| `F.conv_transpose2d` | unsupported | – | transposed convolution missing | no |
| `F.avg_pool1d` | generic | reshape + mean | tiling windows only | no |
| `F.avg_pool2d` | generic | reshape + mean | tiling windows only | no |
| `F.max_pool1d` | generic | reshape + max | tiling windows only | no |
| `F.max_pool2d` | generic | reshape + max | tiling windows only | no |
| `F.adaptive_avg_pool1d` | generic | reshape + mean | dividing sizes only | no |
| `F.adaptive_avg_pool2d` | generic | reshape + mean | dividing sizes only | no |
| `F.interpolate` | generic | nearest: expand + reshape; bilinear: constant matrices | fixed sizes only; F.interpolate raises, written by hand | no |
| `F.pad` | generic | constant: cat with constants; reflect/circular: slices + cat; replicate: edge expand + cat | – | yes |
| `F.unfold` | generic | reshape/permute when kernel == stride | overlapping patches unsupported | no |
| `F.fold` | unsupported | – | overlapping scatter-add | no |
| `F.pixel_shuffle` | generic | reshape + permute | – | no |
| `F.grid_sample` | unsupported | – | data-dependent gather | no |
| `F.dropout` | supported | identity at inference | training=True with p > 0 raises (no RNG) | yes |
| `F.cosine_similarity` | generic | sum/rsqrt composition | – | yes |
| `F.pairwise_distance` | generic | sub/pow/sum/sqrt | – | yes |
| `F.gumbel_softmax` | unsupported | – | no RNG | no |
| `F.cross_entropy` | unsupported | – | loss (training only); needs gather | no |

### Tensor methods and attributes

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `Tensor.__add__/__radd__/__sub__/__rsub__/__mul__/__rmul__/__truediv__/__rtruediv__` | supported | broadcast pointwise ops | – | yes |
| `Tensor.__pow__/__rpow__` | supported | pow (constant integer exponent in [-32, 32] at lowering) | – | yes |
| `Tensor.__neg__` | supported | neg | – | yes |
| `Tensor.__matmul__/__rmatmul__` | supported | matmul | – | yes |
| `Tensor.__lt__/__le__/__gt__/__ge__/__eq__/__ne__` | supported | comparisons (bool as 0/1 f32) | – | yes |
| `Tensor.__iadd__/__imul__ (x += y)` | supported | Python falls back to __add__/__mul__ and rebinds the name | aliases of the old value do not see the update | yes |
| `Tensor.__floordiv__/__mod__` | unsupported | – | constant tensors fold at trace time; runtime values need a rounding op (raises Unsupported) | yes |
| `Tensor.__and__/__or__/__xor__/__invert__` | generic | bool masks: mul / maximum / ne / 1 - x | – | yes |
| `Tensor.__abs__` | supported | abs | – | yes |
| `Tensor.__getitem__ (int, slice, Ellipsis)` | supported | slice view | – | yes |
| `Tensor.__getitem__ (None)` | supported | reshape view (with Ellipsis, ints, slices) | – | yes |
| `Tensor.__getitem__ (tensor / bool mask)` | generic | one index tensor -> index_select; constant bool mask -> static index list | runtime bool masks (data-dependent shape) and several index tensors unsupported | yes |
| `Tensor.__bool__/__iter__` | generic | __iter__ unbinds the first dim; __bool__/__int__ on constant tensors | bool of a runtime value raises (data-dependent control flow) | yes |
| `Tensor.shape` | supported | static tuple of ints | – | yes |
| `Tensor.ndim/dim()` | supported | static int | – | yes |
| `Tensor.numel()` | supported | static int | – | yes |
| `Tensor.size()` | supported | torch.Size / int | – | yes |
| `Tensor.dtype` | supported | dtype objects (float32, bfloat16, int64, bool, ...) with torch promotion; finfo/iinfo | values are computed in f32 | yes |
| `Tensor.device` | supported | device('cpu'); .to(device) is a no-op | – | yes |
| `Tensor.reshape/view` | supported | reshape view | – | yes |
| `Tensor.permute/transpose/T` | supported | permute view | – | yes |
| `Tensor.mT/swapaxes/movedim` | supported | permute view | – | yes |
| `Tensor.unsqueeze` | supported | reshape view | – | yes |
| `Tensor.squeeze` | supported | reshape view (int, tuple or all dims) | – | yes |
| `Tensor.expand` | supported | broadcast view | – | yes |
| `Tensor.expand_as/view_as/reshape_as` | supported | broadcast / reshape views | – | yes |
| `Tensor.flatten/unflatten` | supported | reshape view | – | yes |
| `Tensor.split/chunk` | supported | slice views | – | yes |
| `Tensor.unbind` | supported | slice views | – | yes |
| `Tensor.narrow/select` | supported | slice view | – | yes |
| `Tensor.contiguous` | supported | identity | – | yes |
| `Tensor.to/float/type_as` | supported | cast: logical dtype tag | values stay f32 | yes |
| `Tensor.half/bfloat16/double/int/long/bool/type` | supported | cast: logical dtype tag | values stay f32; float->int of runtime values only for token/position inputs | yes |
| `Tensor.clone/detach/requires_grad_` | supported | identity (clone returns a new proxy) | – | yes |
| `Tensor.sum/mean` | supported | reduction kernels | – | yes |
| `Tensor.max` | supported | max reduction; max(dim) returns (values, indices) | – | yes |
| `Tensor.min/amax/amin` | generic | -max(-x); min(dim) returns (values, indices) | – | yes |
| `Tensor.argmax/argmin/topk/sort/argsort` | generic | argmax/argmin: where + max composition; topk/sort/argsort: constant tensors | runtime topk/sort need an index-producing kernel | yes |
| `Tensor.var/std/norm/prod/logsumexp/any/all` | generic | compositions of sum/mean/max/sqrt/log/exp; prod on constant tensors | runtime prod unsupported | yes |
| `Tensor.softmax` | supported | softmax kernel | – | yes |
| `Tensor.log_softmax` | generic | x - max - log(sum(exp(x - max))) | – | yes |
| `Tensor.cumsum` | supported | cumsum kernel | – | yes |
| `Tensor.cumprod` | unsupported | – | constant tensors only | yes |
| `Tensor.exp/log/sqrt/rsqrt/sin/cos/abs` | supported | pointwise | – | yes |
| `Tensor.relu/silu/gelu/tanh/sigmoid/softplus` | supported | pointwise | – | yes |
| `Tensor.maximum/minimum` | supported | pointwise | – | yes |
| `Tensor.masked_fill` | supported | where(mask, value, x) | – | yes |
| `Tensor.add/sub/mul/div/pow/neg/matmul` | supported | as the operators; alpha and rounding_mode accepted | – | yes |
| `Tensor.clamp/clamp_min/clamp_max` | generic | minimum / maximum | – | yes |
| `Tensor.square/reciprocal/erf/log1p/expm1/sign/floor/round/ceil/trunc` | generic | square/reciprocal/log1p/expm1/sign compose; erf and rounding fold on constant tensors | runtime erf and rounding unsupported | yes |
| `Tensor.isnan/isinf/nan_to_num` | unsupported | – | no non-finite semantics | no |
| `Tensor.where/lerp` | generic | where / a + w * (b - a) | – | yes |
| `Tensor.repeat/repeat_interleave/tile` | generic | unsqueeze + expand + reshape | – | yes |
| `Tensor.tril/triu` | generic | where with a constant triangular mask | – | yes |
| `Tensor.flip/roll` | generic | negative-step slices / slice + cat | – | yes |
| `Tensor.index_select` | supported | index_select | – | yes |
| `Tensor.gather/scatter/scatter_add/index_put/index_copy/masked_scatter` | unsupported | – | gather with constant indices only; scatter, index_put, index_copy and masked_scatter missing | no |
| `Tensor.copy_/fill_/zero_/add_/mul_/index_copy_ (in place)` | generic | functionalized: the proxy is rebound to the out-of-place result | in place on views, viewed tensors, states and weights raises; index_copy_ missing | yes |
| `Tensor.new_zeros/new_ones/new_full/new_empty/new_tensor` | supported | constants | – | yes |
| `Tensor.item/tolist/numpy/cpu/cuda` | generic | item/tolist on constant tensors; cpu/cuda are no-ops | runtime values are data-dependent; numpy absent | yes |
| `Tensor.stride/storage_offset/is_contiguous/data_ptr` | unsupported | – | no storage model on proxies | no |
| `Tensor.unfold` | unsupported | – | method absent | no |

### torch.* functions

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `torch.matmul/mm/bmm` | supported | fir.matmul | – | yes |
| `torch.einsum` | generic | einsum parser: permute/reshape + matmul (ellipsis, broadcasting) | repeated indices within one operand unsupported | yes |
| `torch.addmm/baddbmm/outer` | generic | matmul + add / broadcast mul | – | yes |
| `torch.cat/concat/concatenate` | supported | fir.cat | – | yes |
| `torch.stack` | generic | cat of unsqueezed tensors | – | yes |
| `torch.split/chunk` | supported | fir.split / fir.chunk | – | yes |
| `torch.reshape/transpose/permute` | supported | views | – | yes |
| `torch.where` | supported | fir.where | where(cond) on constant tensors only | yes |
| `torch.softmax` | supported | fir.softmax | – | yes |
| `torch.cumsum` | supported | fir.cumsum | – | yes |
| `torch.sum/mean/max` | supported | fir reductions | – | yes |
| `torch.exp/log/sqrt/rsqrt/sin/cos/abs/tanh/sigmoid/relu` | supported | pointwise | – | yes |
| `torch.maximum/minimum` | supported | pointwise | – | yes |
| `torch.clamp/clip` | generic | maximum/minimum | – | yes |
| `torch.index_select` | supported | fir.index_select | – | yes |
| `torch.gather/scatter/scatter_add/take_along_dim` | unsupported | – | gather/take_along_dim with constant indices only; scatter missing | no |
| `torch.topk/sort/argsort/argmax/argmin` | generic | argmax/argmin composed; topk/sort/argsort on constant tensors | runtime topk/sort unsupported | yes |
| `torch.multinomial/rand/randn/randint/bernoulli` | unsupported | – | no RNG; rand/randn outside a trace are shape-only placeholders for parameter init | no |
| `torch.arange` | supported | torch.arange(start, end, step, dtype) | – | yes |
| `torch.tensor` | supported | fir.tensor (nested lists -> constant) | – | yes |
| `torch.zeros/ones` | supported | fir.zeros / fir.ones constants | – | yes |
| `torch.full/empty/zeros_like/ones_like/full_like/empty_like` | supported | constants | – | yes |
| `torch.tril/triu (static masks)` | generic | constant masks; where with a triangular mask for runtime tensors | – | yes |
| `torch.triu_indices/tril_indices` | generic | Python-side constant lists | – | no |
| `torch.repeat_interleave` | generic | unsqueeze + expand + reshape (static repeats) | – | yes |
| `torch.flip/roll` | generic | negative-step slices / slice + cat | – | yes |
| `torch.diff` | generic | slice - slice | – | yes |
| `torch.cumprod` | generic | exp(cumsum(log(x))) for positive x; torch.cumprod folds constant tensors | – | yes |
| `torch.logsumexp` | generic | max + log(sum(exp(x - max))) | – | yes |
| `torch.linalg.vector_norm/norm` | generic | sqrt(sum(x*x)) | – | yes |
| `torch.lerp` | generic | a + w * (b - a) | – | yes |
| `torch.outer` | generic | a.unsqueeze(-1) * b | – | yes |
| `torch.polar/view_as_complex/view_as_real` | unsupported | – | no complex dtype (Meta Llama reference RoPE) | no |
| `torch.bucketize/searchsorted/histc` | unsupported | – | data-dependent integer outputs | no |
| `torch.finfo/iinfo` | supported | dtype-aware finfo/iinfo (finite values) | – | no |
| `torch.no_grad/inference_mode` | supported | no-op context managers and decorators (no autograd); autocast(enabled=False) | – | n/a |
| `torch.nn.functional.scaled_dot_product_attention` | pattern | fir.scaled_dot_product_attention | dropout_p > 0 raises | yes |
| `fir.rotary_embedding` | pattern | rotary kernel; folded into attention_decode when q/k feed it | default inv_freq only | n/a |
| `fir.scan` | supported | static unrolled scan | – | n/a |

### dtype system and promotion

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `dtype.float32 compute` | supported | all buffers and VM registers are f32 (rounded after every op) | – | n/a |
| `dtype.float32 checkpoint` | supported | fmLoadSafeF32 loads F32 safetensors | – | n/a |
| `dtype.bfloat16 checkpoint` | unsupported | – | fmLoadSafeF32 rejects non-F32 tensors; no widening at load | n/a |
| `dtype.float16 checkpoint` | unsupported | – | fmLoadSafeF32 rejects non-F32 tensors | n/a |
| `dtype.bfloat16/float16 compute` | unsupported | – | dtype tags and promotion follow torch; values are computed in f32 (no half rounding) | n/a |
| `dtype.q4 weights` | supported | fir.linear [in,out]: int4 asymmetric grouped (32) quantized at load | linear_t / [out,in] and micro-kernel weights stay f32 | n/a |
| `dtype.q8/int8 weights` | unsupported | – | no q8 weight dtype in FIR grammar | n/a |
| `dtype.prequantized checkpoints (GPTQ/AWQ/bnb/FP8/MXFP4)` | unsupported | – | no packed-int or fp8 loaders; q4 is produced from f32 | n/a |
| `dtype.bool` | supported | torch.bool tag over 0/1 f32 values; & \| ^ ~ on masks | – | n/a |
| `dtype.int32/int64` | generic | integer dtype tags over f32 values; integer arithmetic folds on constant tensors | runtime integer division and overflow unsupported; exact below 2**24 | n/a |
| `dtype.uint8/int8/int16` | generic | integer dtype tags over f32 values; integer arithmetic folds on constant tensors | runtime integer division and overflow unsupported; exact below 2**24 | n/a |
| `dtype.complex64` | unsupported | – | no complex dtype | n/a |
| `dtype.float8_e4m3fn/e5m2` | unsupported | – | no fp8 dtype | n/a |
| `dtype.quantized tensors (qint8, quint8)` | unsupported | – | torch.ao quantized ops not traced | n/a |
| `promotion.float_float` | supported | single f32 category: every float op is f32 | – | n/a |
| `promotion.int_float` | supported | torch promotion (categories, zero-dim and scalar rules) | – | n/a |
| `promotion.bool_arith` | supported | bool values are 0/1 f32, arithmetic promotes to f32 | – | n/a |
| `promotion.autocast/mixed precision` | unsupported | – | autocast to f16/bf16 raises; autocast(enabled=False) is a no-op | n/a |
| `promotion.softmax/norm upcast (dtype=torch.float32)` | supported | already f32 | – | n/a |

### Tracing semantics

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `trace.example_specialization` | supported | fir.trace: one specialization per config/TensorSpec (like torch.jit.trace) | – | n/a |
| `trace.symbolic_config_dims` | supported | Dim('name', v) / config symbol names resolved at trace time; require sym lines in FIR | – | n/a |
| `trace.dynamic_dims (torch.export Dim/Dim.AUTO)` | unsupported | – | shapes are concrete at trace time; no symbolic sequence length or batch | n/a |
| `trace.static_python_control_flow` | supported | executed during tracing | – | n/a |
| `trace.data_dependent_control_flow` | unsupported | – | bool() of a runtime value raises; constants evaluate eagerly | n/a |
| `trace.torch_jit_script` | unsupported | – | no TorchScript/AST compilation; loops are unrolled | n/a |
| `trace.torch_fx_symbolic_trace` | generic | fir.from_torch replays the FX graph against torch_compat | tested with a fake torch.fx object model (torch is optional) | yes |
| `trace.torch_export_program` | unsupported | – | from_torch does not read ExportedProgram graph_signature (lifted params/buffers, BUFFER_MUTATION outputs) | no |
| `trace.core_aten_decomposition` | unsupported | – | no run_decompositions table mapping Core ATen to fir ops | no |
| `trace.parameter_lifting` | supported | Parameters become weights named by checkpoint path; payloads never read | – | n/a |
| `trace.buffer_lifting` | supported | register_buffer values become constants | – | yes |
| `trace.constant_lifting` | supported | constants folded; non-uniform constants written by an init node | – | n/a |
| `trace.functionalization` | supported | SSA graph; State.update produces versioned writes (in place when safe) | – | n/a |
| `trace.inplace_tensor_ops` | generic | functionalized by rebinding the proxy | in place on views, states and weights raises | yes |
| `trace.aliasing_views` | supported | views compose into strided descriptors over the base buffer | no writes through views | n/a |
| `trace.multiple_outputs` | generic | graph.outputs: fir.evaluate(all_outputs=True), compare_graphs | export writes one logits output | n/a |
| `trace.decode_abi` | supported | scalar token + position inputs, logits output, one token per step | – | n/a |
| `trace.prefill_sequence` | generic | fir.evaluate over [L, ...] inputs | export rejects sequence/batched inputs | n/a |
| `trace.batch_gt_1` | unsupported | – | export ABI is batch 1 | n/a |
| `trace.layer_repeat` | supported | extract_repeats: one homogeneous stack -> repeat layers | one stack per file | n/a |
| `trace.heterogeneous_layers` | generic | unrolled layers | file size grows with depth; no per-layer-kind repeat | n/a |
| `trace.autograd` | unsupported | – | out of scope | n/a |

### HF transformers: KV cache and state conventions

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `hf.past_key_values (legacy tuples)` | unsupported | – | graph IO carries no cache tensors; persistent states replace them | no |
| `hf.DynamicCache` | generic | rewritten as StaticCache-style persistent states | growing torch.cat cache needs dynamic shapes | no |
| `hf.StaticCache` | pattern | torch_compat.transformers.StaticCache: per-layer states [kv_heads, P, D] written at cache_position -> attention_decode; the layer stack repeats | batch 1, one row per step | no |
| `hf.cache_position / position_ids` | supported | position input role (scalar) | vectors of positions (prefill) not exported | no |
| `hf.SlidingWindowCache / SlidingWindowLayer (ring buffer)` | unsupported | – | slot = position % window needs remainder; roll/shift of a full window | no |
| `hf.sliding window over a full cache` | generic | full [P, ...] cache + window mask | mask breaks the attention_decode pattern; generic kernels | no |
| `hf.HybridCache / HybridChunkedCache` | generic | global layers via attention_decode, sliding layers generic | mixed layer kinds break repeat extraction | no |
| `hf.QuantizedCache` | unsupported | – | no int8/int4 state buffers | no |
| `hf.OffloadedCache / OffloadedStaticCache` | unsupported | – | host offload is a runtime policy outside FIR | no |
| `hf.EncoderDecoderCache` | unsupported | – | no encoder input role or cross-attention cache fill | no |
| `hf.MambaCache (conv_states, ssm_states)` | pattern | conv state -> conv1d_step, ssm state -> state_update_reduce | – | no |
| `hf.attention_mask (padding)` | unsupported | – | no mask input; batch 1 | no |
| `hf.state reset at position 0` | supported | reset kernel zeroes state at position 0 | – | n/a |

### HF transformers: rotary embedding variants

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `rope.default (rotate_half, NeoX layout)` | pattern | fir.rotary_embedding, or HF inv_freq buffer + rotate_half re-fused by refuse_rotary; fused into attention_decode | – | n/a |
| `rope.interleaved (GPT-J rotate_every_two)` | generic | reshape [..., D/2, 2] + sin/cos of position * inv_freq constant | no fused kernel | n/a |
| `rope.partial_rotary_factor (NeoX rotary_pct, Phi, GLM)` | generic | slice rotary part + rotary + cat | rotary_dim slicing breaks the fused attention+rope match | n/a |
| `rope.linear` | generic | rotary_embedding(x, position / factor) | scaled position does not match the position role: rope not fused into attention | n/a |
| `rope.dynamic (dynamic NTK)` | unsupported | – | base depends on the running sequence length; needs runtime-dependent inv_freq | n/a |
| `rope.yarn` | generic | constant inv_freq table + attention_factor: sin/cos of position * inv_freq composed by hand | rotary op derives inv_freq from base only; sin/cos series lose accuracy for large angles | n/a |
| `rope.longrope` | generic | constant inv_freq table (short or long factor fixed at export) composed by hand | switch at original_max_position_embeddings needs a runtime select | n/a |
| `rope.llama3` | generic | constant inv_freq table composed by hand | rotary op derives inv_freq from base only | n/a |
| `rope.proportional` | generic | partial rotary + scaled constant inv_freq | – | n/a |
| `rope.mrope (Qwen2-VL multimodal sections)` | unsupported | – | needs three position components per token; ABI has one scalar position (text-only reduces to default) | n/a |
| `rope.complex (Meta reference freqs_cis)` | unsupported | – | no complex dtype; rewrite as rotate pairs | n/a |
| `rope.nope layers (Llama 4 / SmolLM3)` | supported | omit rotary on those layers | per-layer difference breaks repeat extraction | n/a |

### HF transformers: attention variants

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `attn.MHA decode` | pattern | attention_decode (one lane per KV head) | – | n/a |
| `attn.GQA decode` | pattern | attention_decode (slice/cat, expand/reshape or HF repeat_kv of KV heads) | – | n/a |
| `attn.MQA decode` | pattern | attention_decode with one KV head | – | n/a |
| `attn.MLA (DeepSeek-V2/V3 latent cache)` | generic | latent state + matmuls + decoupled rope slice | latent cache is not a K/V cache: no fused kernel | n/a |
| `attn.sliding window mask` | generic | where(arange > position - W, ...) on full cache | pattern requires exactly arange(P) <= position | n/a |
| `attn.ALiBi bias` | generic | scores + slope * (arange - position) | additive bias breaks the pattern | n/a |
| `attn.logit soft-capping (Gemma 2)` | generic | tanh(scores / cap) * cap | breaks the pattern | n/a |
| `attn.attention sinks (gpt-oss)` | generic | extra logit column in softmax | breaks the pattern | n/a |
| `attn.custom scale (query_pre_attn_scalar)` | pattern | scale attribute of attention_decode | – | n/a |
| `attn.QK-norm (Qwen3, OLMo 2, Gemma 3)` | pattern | rmsnorm over head_dim before rotary; attention_decode | – | n/a |
| `attn.causal prefill (is_causal)` | generic | sdpa with static causal mask | evaluate only (decode ABI) | n/a |
| `attn.bidirectional encoder` | generic | sdpa without mask | evaluate only | n/a |
| `attn.cross attention` | unsupported | – | no encoder-memory input role | n/a |
| `attn.flash / sdpa / eager backends` | pattern | eager (matmul + additive causal mask + softmax), sdpa and repeat_kv re-fuse into attention_decode | – | n/a |
| `attn.contexts beyond the VM step limit` | unsupported | – | one dispatch per KV head is limited to 2**24 VM steps on CPU; no split-K | n/a |
| `attn.differential attention` | generic | two softmax maps subtracted | – | n/a |

### HF transformers: mixture-of-experts routing

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `moe.router softmax` | supported | softmax kernel | – | n/a |
| `moe.top-k selection` | generic | k rounds of max + eq mask (dense mask) | ties select several experts; no topk op | n/a |
| `moe.dense evaluation of all experts` | generic | every expert computed, weighted by the masked router | E/k times the compute of sparse dispatch | n/a |
| `moe.sparse dispatch (grouped experts)` | unsupported | – | needs top-k indices, integer dtype and a dynamic-expert linear | n/a |
| `moe.shared experts (DeepSeek, Qwen-MoE)` | supported | plain MLP added to the routed output | – | n/a |
| `moe.group-limited routing (DeepSeek-V3 noaux_tc)` | unsupported | – | needs top-k over groups and sigmoid scoring with bias | n/a |
| `moe.expert weights [E, out, in]` | generic | rank-3 f32 arena weight + index_select by expert id | whole expert tensor in the f32 arena; no q4 | n/a |

### HF transformers: SSM and linear-attention families

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `ssm.Mamba-1 selective scan step` | pattern | conv1d_step + state_update_reduce (examples/mamba.py) | – | n/a |
| `ssm.Mamba-1 prefill scan` | generic | fir.scan / cumsum composition | evaluate only | n/a |
| `ssm.Mamba-2 SSD decode step` | generic | conv1d_step + per-head state update + gated grouped RMSNorm | per-head scalar decay; no dedicated example or pattern test | n/a |
| `ssm.Mamba-2 chunked SSD (segsum)` | generic | cumsum + exp + masked matmuls | evaluate only | n/a |
| `ssm.Mamba-3 (trapezoidal, complex/rotary state, MIMO)` | generic | state updates + sin/cos rotations of B/C | data-dependent angles composed by hand; no fused kernel | n/a |
| `ssm.RWKV token shift` | supported | State holding the previous x (update(value)) | – | n/a |
| `ssm.RWKV-4/5/6 WKV step` | generic | matrix state update + matmul readout; per-head GroupNorm | matrix-state update is not pointwise: copy-mode state write | n/a |
| `ssm.RWKV-7 generalized delta rule` | generic | S = S*diag(w) + (S a) b^T + v k^T via matmuls | no fused matrix-state kernel | n/a |
| `ssm.RetNet recurrent` | generic | S = gamma*S + k^T v; o = q S | – | n/a |
| `ssm.GLA recurrent` | generic | S = diag(alpha) S + k^T v | – | n/a |
| `ssm.DeltaNet / Gated DeltaNet (Qwen3-Next)` | generic | S = a*S - b*(S k) k^T + b v k^T; L2-normalized q/k; short conv via conv1d_step | no fused matrix-state kernel | n/a |
| `ssm.xLSTM mLSTM/sLSTM` | generic | matrix memory + normalizer + max stabilizer states | – | n/a |
| `ssm.hybrid attention/SSM stacks (Jamba, Zamba, Nemotron-H)` | generic | unrolled heterogeneous layers | one repeat stack per file | n/a |

### HF transformers: vision and multimodal front ends

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `mm.ViT/SigLIP/CLIP patch embedding (Conv2d, kernel == stride)` | generic | reshape/permute + matmul | pixel input has no export role | n/a |
| `mm.Qwen2-VL Conv3d patch embedding` | generic | reshape/permute + matmul | pixel input has no export role | n/a |
| `mm.vision encoder tower` | generic | bidirectional attention over patches | evaluate only | n/a |
| `mm.projector (MLP / pixel shuffle / pooling)` | generic | Linear + reshape/permute | – | n/a |
| `mm.inputs_embeds merge (image tokens into text)` | unsupported | – | no embedding-vector input role; masked_scatter missing | n/a |
| `mm.Whisper audio conv front end (stride 2, padding 1)` | unsupported | – | strided/padded Conv1d missing; encoder export ABI missing | n/a |
| `mm.2-D rotary for vision patches` | generic | rotary composed per axis | – | n/a |

### HF transformers: embeddings and output heads

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `emb.token embedding` | supported | ext fir.embedding | – | n/a |
| `emb.learned absolute positions (GPT-2, OPT offset)` | supported | ext fir.embedding of the position role | OPT offset of 2 needs position + 2 (gather kernel) | n/a |
| `emb.sinusoidal positions` | generic | constant table or sin/cos of position | – | n/a |
| `emb.scaled embedding (Gemma sqrt(d))` | supported | pointwise mul, fused | – | n/a |
| `emb.tied lm_head` | supported | Embedding.project -> ext fir.linear_t | – | n/a |
| `emb.untied lm_head` | supported | Linear oi -> ext fir.linear_t | f32 only (no q4 for oi) | n/a |
| `emb.final logit soft-capping (Gemma 2)` | supported | tanh(logits / cap) * cap pointwise | – | n/a |
| `emb.logit scaling (Cohere, Granite)` | supported | pointwise mul | – | n/a |
| `emb.Gemma RMSNorm (1 + weight)` | generic | decomposed rmsnorm with (1 + w) | refuse_norms absorbs a plain weight multiply only | n/a |
| `emb.multi-token prediction heads (DeepSeek-V3 MTP)` | unsupported | – | export writes one output (graph.outputs evaluate) | n/a |

### Sampling heads

| Item | Status | Lowering | Blocker / limit | from_torch |
| --- | --- | --- | --- | --- |
| `sample.greedy argmax` | supported | host sampler (temperature 0) over logits | no in-graph argmax | n/a |
| `sample.temperature` | supported | host sampler | – | n/a |
| `sample.top-p` | supported | host sampler | – | n/a |
| `sample.min-p` | supported | host sampler | – | n/a |
| `sample.repetition penalty / banned ids` | supported | host sampler | – | n/a |
| `sample.top-k` | unsupported | – | host sampler has no top-k; no in-graph topk | n/a |
| `sample.in-graph sampling (multinomial)` | unsupported | – | no RNG, no sort/topk ops | n/a |
| `sample.logprobs (log_softmax)` | generic | log_softmax composition in graph or on host | – | n/a |

## Prioritized gaps

Ranking is by the number of model families, out of this reference set of 24,
that need the capability to run from their published HF checkpoint and modeling
code: Llama 2, Llama 3.x, Mistral 7B, Mixtral, Qwen2.5, Qwen3, Qwen3-MoE,
Qwen3-Next, Gemma 2, Gemma 3, Phi-3/3.5, Phi-4, DeepSeek-V3/R1, gpt-oss, GPT-2,
Pythia/GPT-NeoX, Falcon, OLMo 2, SmolLM, Mamba, Mamba-2, RWKV-7, Jamba,
Qwen2.5-VL. The rank weighs that count by severity: a gap that prevents loading
or tracing a model ranks above one that only costs speed or memory, and among
equals the item more later gaps depend on ranks first.

| # | Capability | Models (of 24) | Minimal feature | Depends on |
| --- | --- | --- | --- | --- |
| 1 | BF16/F16 checkpoint loading | 22 (all except GPT-2 and Mamba-1 f32 releases) | widen `BF16`/`F16` safetensors to f32 in `fmLoadSafeF32` (and for `SOURCE.fir_flat` copies) | – |
| 2 | Prefill export (L tokens per step) | 24 | `[L]` token/position inputs, attention pattern writing L cache rows with an in-chunk causal mask, last-row logits | 17 (fixed chunk first, symbolic L later) |
| 3 | Unmodified `import torch` modeling files | 23 (all HF Python models) | an import alias mapping `torch`, `torch.nn`, `torch.nn.functional` and the used `transformers` modules to `formers.fir.torch_compat` while a modeling file loads (today the file's imports are changed by hand); `torch.cond` → `where` | – |
| 4 | Re-fusion of the remaining HF idioms | 16 (non-default RoPE, Gemma norms, partial rotary, QK-norm with composed RoPE) | `refuse_rotary` for a constant `inv_freq` table and partial `rotary_dim` slices, `refuse_norms` for `(1 + w)` weights and non-affine norms (RMSNorm, `x @ w.T`, `rotate_half` default RoPE, `repeat_kv`, causal masks and GELU/SiLU forms re-fuse today) | 9, 29 |
| 5 | `torch.export` / Core ATen ingestion | 24 | read `ExportedProgram.graph_signature` (lifted PARAMETER/BUFFER/CONSTANT_TENSOR inputs, `BUFFER_MUTATION` outputs → `State.update`), a decomposition table mapping every `supported`/`generic` Core ATen op | 3, 4 |
| 6 | Checkpoint-bound buffers | 3 (checkpoints that store buffers: older GPT-2/NeoX `attn.bias`/`masked_bias`, batch-norm running statistics) | persistent buffers read from the checkpoint instead of evaluated at construction (buffers are constants today) | – |
| 7 | q4 (and q8) for `[out,in]` linears | 23 (every `nn.Linear` checkpoint) | quantize `fir.linear_t` weights at load (no transpose needed), q8 dtype | – |
| 8 | Attention beyond the VM step limit | 22 (all attention models at ≥ 2k context) | split-K attention: lanes over (KV head, slot chunk) plus a combine kernel with running max/sum | – |
| 9 | General RoPE frequencies | 14 (Llama 3.x llama3; Qwen2.5/Qwen3/DeepSeek/gpt-oss yarn; Phi-3/Phi-4 longrope; Pythia/Phi partial; Gemma 3 linear; Qwen2.5-VL mrope; SmolLM3 NoPE; Jamba none) | `rotary` attributes `inv_freq` (constant vector), `attention_factor`, `rotary_dim`; kept fusable into `attention_decode` | 6 |
| 10 | Host top-k sampling | 24 (HF `generation_config.json` commonly sets `top_k`) | top-k step in `fmSample` before top-p | – |
| 11 | Integer dtypes | 10 (MoE, sliding ring buffers, in-graph sampling, large vocab ids) | `i32` tensors and buffers, integer pointwise ops, f32↔i32 casts, `div` rounding modes | – |
| 12 | Heterogeneous layer stacks | 10 (Gemma 2/3, gpt-oss, DeepSeek-V3, Qwen3-Next, Jamba, Llama 4, SmolLM3, Phi-3 small, Mistral hybrids) | several `repeat` groups per file or per-layer variant selection by layer index | – |
| 13 | Sliding-window attention | 6 (Mistral 7B v0.1, Gemma 2, Gemma 3, Phi-3, gpt-oss, Qwen2.5 optional) | `attention_decode` with a window attribute over a full cache; ring-buffer cache with `position % window` | 11 (ring buffer), 12 |
| 14 | top-k operators | 7 (Mixtral, Qwen3-MoE, Qwen3-Next, DeepSeek-V3, gpt-oss, Jamba, Phi-3.5-MoE) | small-k `topk` and `sort` on runtime tensors returning values and indices (`argmax`/`argmin` compose from `max` today) | 11 |
| 15 | Sparse MoE dispatch | 7 (same set) | expert linear with a dynamic expert index (q4 rank-3 expert weights), weighted combine, shared experts, DeepSeek group-limited sigmoid routing | 11, 14, 7 |
| 16 | Attention pattern extensions | 6 (Gemma 2 soft-cap, gpt-oss sinks, Falcon/MPT-style ALiBi, Phi-4 mini, differential attention models, OLMo) | optional `softcap`, `sink` logit, additive per-head slope bias in `attention_decode` | – |
| 17 | Dynamic sequence dimension | 24 (prefill, chunked prefill) | symbolic `L` in shapes with min/max (like `torch.export.Dim`), runtime count registers in kernels | – |
| 18 | Multiple exported outputs | 6 (DeepSeek-V3 MTP, speculative decoding heads, embedding models, router-logit outputs, hidden-state taps) | named output buffers beyond `logits` in the emitter and runner (`graph.outputs` traces and evaluates) | – |
| 19 | Matrix-state recurrent kernel | 5 (Mamba-2, RWKV-7, Qwen3-Next Gated DeltaNet, Jamba Mamba layers, xLSTM/RetNet/GLA) | fused `state_matmul_step`: `S = f(S) + u v^T`, `y = S q`, in place, reset at position 0 | – |
| 20 | Mamba-2 SSD decode verified | 1 (Mamba-2) plus Codestral Mamba and the Nemotron-H/Granite-4 hybrids | example + numerics test: per-head scalar decay, `n_groups` B/C broadcast, gated grouped RMSNorm | 19 |
| 21 | Native f16/bf16 weights | 22 | `f16`/`bf16` weight dtypes and VM half loads (half the resident memory) | 1 |
| 22 | Pre-quantized checkpoints | 4 (DeepSeek-V3 FP8, gpt-oss MXFP4, GPTQ/AWQ community builds of every model) | loaders for packed int4 with scales/zeros, FP8 block scales, MXFP4 | 7, 21 |
| 23 | Rounding and special pointwise ops | 7 (ring buffers, Whisper/ViT exact GELU via erf, YaRN tables computed in graph, `isnan` guards) | `floor`, `trunc`, `round`, `remainder`, standalone `erf`, `atan`, `isnan`/`isinf` | 11 |
| 24 | gather / scatter | 7 (MoE combine, repetition penalty in graph, Qwen2.5-VL token merge, Mamba-2 chunk indices) | `gather` along a dim, `scatter`/`scatter_add`/`index_add` with dynamic indices | 11 |
| 25 | Multi-input ABI | 2 (Qwen2.5-VL, Gemma 3 vision) plus LLaVA/Phi-3-vision outside the set | runner roles for an `inputs_embeds` vector, pixels and multi-component positions (mrope); `fir.trace` and `fir.evaluate` accept extra inputs, export rejects them | 2, 17 |
| 26 | Vision and audio front ends | 2 (Qwen2.5-VL Conv3d, Gemma 3 SigLIP) plus Whisper | named patch-embed op (kernel == stride), strided/padded `conv1d`/`conv2d`, bidirectional encoder export | 2, 25 |
| 27 | Batch > 1 and padding masks | 24 (serving) | batch dimension in the decode ABI, per-sequence positions, padding mask input | 2, 17 |
| 28 | Accurate `sin`/`cos` for large angles | 14 (rope models when frequencies are composed by hand, positions ≥ 32k) | two-constant (Cody–Waite) range reduction or cos/sin tables indexed by position | 9 |
| 29 | Norm variants | 5 (Gemma 1/2/3 `1 + w`, OLMo 2 non-affine/QK norms, Cohere LayerNorm without bias, Mamba-2 gated norm) | `rmsnorm`/`layernorm` attributes `offset`, `bias=False`, `gate` input | 4 |
| 30 | Control flow and RNG operators | 2 (in-graph sampling, early-exit / dynamic-NTK models) | `cond` with effects, `while_loop`, RNG state for `multinomial` | 11, 14 |

### Dependency order

1. Loader and quantization groundwork, no graph changes: BF16/F16 widening (1),
   q4/q8 for `linear_t` (7), host top-k (10).
2. Front-end completeness: the `import torch` alias (3), checkpoint-bound buffers
   (6), the remaining re-fusion patterns (4), then `torch.export` / Core ATen
   ingestion (5).
3. Attention and RoPE generality on the decode ABI: general RoPE frequencies (9)
   with accurate `sin`/`cos` (28), attention extensions (16), split-K attention
   (8), norm variants (29).
4. Integer dtypes (11), then rounding ops (23), gather/scatter (24), argmax/top-k
   (14). These unlock sliding-window ring buffers (13), sparse MoE (15) and
   in-graph sampling (30).
5. Graph structure: heterogeneous layer stacks (12), multiple outputs (18),
   matrix-state recurrent kernel (19) and the Mamba-2 example (20).
6. Shapes and ABI: dynamic sequence dimension (17), prefill export (2), batch and
   padding (27), multi-input ABI (25), vision/audio front ends (26).
7. Memory: native half weights (21), pre-quantized checkpoints (22).
