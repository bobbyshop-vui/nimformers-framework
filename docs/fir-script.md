# FIR script: Python forward passes to model.fir

`formers.fir` traces a static Python decode step into a typed SSA graph, rewrites
recognised subgraphs (attention, norms, pointwise chains, convolution and SSM
state steps) into fused operations, and emits self-contained FIR v1 for the
generic `runtime graph` runner. Every fused operation becomes one compact
loop-based `session_vm` micro-kernel; everything else uses generic loop-based
kernels. The package uses only the Python standard library. Tracing records
shapes, checkpoint names, operations and effects; it never opens a checkpoint.

Run from `other/formers.n/python` (or add that directory to `PYTHONPATH`).

## Quick start

```python
from formers import fir

class Block(fir.Module):
    def __init__(self):
        super().__init__()
        self.table = fir.Embedding(16, 4)
        self.norm = fir.LayerNorm(4)
        self.fc = fir.Linear(4, 4)

    def forward(self, token, position):
        x = self.table(token)
        return self.table.project(x + self.fc(self.norm(x)).gelu())

inputs = (fir.TensorSpec(1, "token"), fir.TensorSpec(1, "position"))
config = {"vocab_size": 16, "n_positions": 8}
graph = fir.trace(Block(), inputs, config=config)          # optimized graph
path = fir.export(graph, "/tmp/fir-block", report=True)    # model.fir + config.json
```

```sh
python3 -m formers.fir.examples.gpt2_125m /tmp/gpt2-fir --report   # real-size GPT-2 125M
python3 -m formers.fir.examples /tmp/fir-examples --report           # tiny GPT-2, Llama, Mamba
python3 -m formers.fir.report /tmp/gpt2-fir/model.fir                # statistics of any FIR file
../formers-cli --fir-check /tmp/gpt2-fir/model.fir
```

## API

| Function | Purpose |
| --- | --- |
| `fir.trace(model, example_inputs, *, config=None, name="scripted", optimize=True)` | Trace one specialization; `optimize=True` runs `PASSES`. `fir.script` is the same call. `example_inputs` is a sequence (positional) or a mapping (keyword arguments); `TensorSpec` entries become graph inputs, other values are passed through. `config` is a dict or an object with `to_dict()` (a `PretrainedConfig`). |
| `fir.export(graph_or_module, directory, *, example_inputs=None, config=None, name=None, optimize=None, prefix=None, require_config=(), report=False)` | Write `model.fir` and `config.json`; returns the model path. |
| `fir.emit(graph, *, prefix=None, require_config=())` | FIR text without writing files. |
| `fir.optimize_graph(graph)`, `fir.PASSES` | The pass pipeline (in place). |
| `fir.evaluate(graph, inputs, parameters, states, all_outputs=False)` | Pure-Python reference evaluation of any graph (raw or optimized); `all_outputs=True` returns every graph output. |
| `fir.compare_graphs(a, b, steps, parameters, rel_tol=, abs_tol=)` | Step-by-step comparison of all outputs and persistent states. |
| `fir.random_parameters(graph, seed, scale)` | Synthetic weights by checkpoint name. |
| `fir.FirModel(path_or_text, config=None, weights=None)` | Python execution of a FIR graph file (`step(token)`, `reset()`). |
| `fir.report(text, graph=None, config=None)`, `fir.fir_stats(text, config)` | Node, buffer, kernel and instruction statistics. |
| `fir.from_torch(module, example_inputs, ...)` | Optional torch.fx adapter; imports torch only when called. |
| `fir.torch` (`formers.fir.torch_compat`) | `torch`-shaped namespace with `nn`, `nn.functional` and `transformers` pieces, so torch/HF modeling code traces after changing its imports. |

`export` options:

- `optimize=None` emits the graph as given. `optimize=True` optimizes a copy of an
  unoptimized graph. `optimize=False` emits an unoptimized graph with generic
  kernels only (an optimized graph raises `ValueError`).
- `prefix="transformer."` strips the prefix from checkpoint names and writes a
  `prefix` line; the runner tries the plain name, then the prefixed name, so the
  file loads both HF GPT-2 layouts (`h.0.…` and `transformer.h.0.…`).
- `require_config` is a list of config keys (their traced values become
  requirements) or a mapping key → value (also for keys the config omits). Each
  becomes `require config KEY VALUE`.
- `report=True` prints the statistics of the written file.

A module passed to `export` is traced with `example_inputs`, `config` and `name`.
Errors are `fir.LoweringError` (a `ValueError`) and list every failing node with
scope, operation and output. Nothing is written when emission fails.

## Front end

`Module` registers parameters, buffers, child modules and states by attribute
assignment, `register_parameter`, `register_buffer` and `register_state`, and has
the torch registry API: `named_modules`, `named_children`, `named_parameters`,
`named_buffers`, `get_submodule`, `state_dict` (parameters and persistent buffers),
`load_state_dict` (shape checks, missing/unexpected keys, buffer values), `apply`,
`train`/`eval` (modules start in eval mode), `requires_grad_`, and `to`/`float`/
`half`/`cuda`/`cpu` as no-ops. Containers: `ModuleList`, `ModuleDict`,
`Sequential`, `ParameterList`, `ParameterDict`. Leaf modules: `Linear`
(`layout="oi"` `[out,in]` default for unnamed layers, `layout="io"` `[in,out]`
default for named legacy layers, `dtype="q4"` for `io` layers), `Embedding` (with
`project` for a tied head), `LayerNorm`, `RMSNorm`, causal `Conv1d` (groups,
depthwise), `GELU`, `Dropout` (identity in eval mode; raises in training mode
with `p > 0`), `Identity`. An explicit parameter name overrides the module path. Parameters support arithmetic, views and every tensor method
(`A_log.exp()`, `weight.T`, …).

Tensor proxies have the torch `Tensor` API. `shape` and `size()` return
`torch.Size`, `size(d)` an int; `dtype` is a dtype object and `device` is
`cpu`. Operators: broadcast arithmetic, comparisons, `@`, `//`, `%`, and `&`,
`|`, `^`, `~` on bool masks. Indexing takes ints, slices (negative steps
included), `None`, `Ellipsis`, one index tensor (`index_select`) and constant
bool masks; `x[i] = v` and the in-place methods (`add_`, `mul_`, `masked_fill_`,
`copy_`, `fill_`, …) are functionalized by rebinding the proxy to the
out-of-place result (in place on views, viewed tensors, states and weights
raises). Methods: the arithmetic forms (`add(alpha=)`, `div(rounding_mode=)`,
`pow`, `clamp`, …), reductions (`sum`, `mean`, `max`/`min` with `(values,
indices)` named tuples, `amax`, `argmax`, `var`, `std`, `norm`, `logsumexp`,
`any`, `all`), `softmax(dtype=)`, `log_softmax`, `cumsum`, every view (`view`,
`reshape`, `view_as`, `transpose`, `t`, `mT`, `permute`, `movedim`, `unsqueeze`,
`squeeze`, `flatten`, `unflatten`, `expand`, `expand_as`, `split`, `chunk`,
`unbind`, `narrow`, `select`), `repeat`, `repeat_interleave`, `tril`/`triu`,
`flip`, `roll`, casts (`to`, `float`, `half`, `bfloat16`, `long`, `bool`,
`type_as`), `new_zeros`/`new_full`/…, `clone`/`detach`/`contiguous`, and the
activations listed below.

Module functions follow torch argument orders: `arange(start, end, step,
dtype=)`, `full`, `zeros`, `ones`, `empty`, `eye`, `*_like`, `tensor`, `cat`,
`stack`, `where`, `einsum` (ellipsis, broadcasting), `outer`, `addmm`, `baddbmm`,
`tril`, `triu`, `cumsum`, `flip`, `roll`, `gather` (constant indices), static
`scan`, `rotary_embedding` (HF `rotate_half`), `scaled_dot_product_attention`
(bool or additive `attn_mask`, `is_causal`, `scale`, `enable_gqa`; `dropout_p >
0` raises).

Values are computed in f32; dtypes are logical tags with torch promotion
(`dtypes.result_type`: bool < integer < float categories, zero-dim tensors and
Python scalars do not raise the category's width). Casting a runtime float to
an integer dtype is allowed for token/position inputs (and their views); casts
of constants truncate. Integer rounding (`floor`, `trunc`, `round`,
`remainder`, rounding `div`), `erf`, `prod`, `cumprod`, `sort`, `topk`,
`nonzero` and `item()`/`tolist()` evaluate on constant tensors only and raise
`fir.Unsupported` (a `NotImplementedError`) on runtime values. Infinite scalars
become ±F32 max.

Outside a trace, tensor functions evaluate eagerly on Python values, so
construction-time tensors (`inv_freq`, `torch.tril(torch.ones(P, P))` masks,
`register_buffer` values) are constants; a trace imports them as constants.
`torch.empty`/`randn` outside a trace are shape-only placeholders for parameter
initialization; inside a trace random functions raise.

### torch_compat

```python
from formers.fir import torch_compat as torch
from formers.fir.torch_compat import nn
from formers.fir.torch_compat.nn import functional as F
from formers.fir.torch_compat.transformers import ACT2FN, PretrainedConfig, StaticCache, StaticCacheDecoder
```

- `torch`: `Tensor`, `Size`, `device`, dtype objects (`float32`, `float16`,
  `bfloat16`, `float64`, `int64`, `int32`, `int16`, `int8`, `uint8`, `bool`),
  `finfo`/`iinfo`, `result_type`, the creation and math functions above,
  `no_grad`/`inference_mode`/`enable_grad` (context managers and decorators),
  `autocast(enabled=False)`, `compile`, `jit`, `cuda`, `backends`, `linalg`,
  `special` stubs.
- `torch.nn`: the fir containers and `Module`, plus torch-signature `Linear`
  (`[out,in]`), `Embedding`, `LayerNorm`, `RMSNorm`, `Conv1d` (zero padding,
  `same`/`valid`, stride), `Dropout`, `GELU(approximate=)`, `SiLU`, `ReLU`,
  `Tanh`, `Sigmoid`, `Softplus`, `LeakyReLU`, `ELU`, `SELU`, `Mish`,
  `Hardswish`, `GLU`, `Softmax`, `Flatten`, `Unflatten` and the other
  parameter-free activations; `nn.init` initializers are no-ops.
- `torch.nn.functional`: `linear`, `embedding`, `layer_norm`, `rms_norm`,
  `group_norm`, `batch_norm`, `normalize`, `softmax`/`log_softmax` (`dtype=`),
  `gelu`, `silu`, the other activations, `dropout`, `scaled_dot_product_attention`,
  `conv1d` (zero padding and stride over the causal kernel), `pad`
  (constant/reflect/replicate/circular), `one_hot`, `cosine_similarity`.
- `transformers`: `PretrainedConfig` (attribute access over config.json keys,
  `attribute_map` and GPT-2 style aliases, `head_dim`/`num_key_value_heads`
  defaults, `from_pretrained`/`from_json_file`; `to_dict()` returns the given
  keys), `ACT2FN`, the HF GELU classes, `Conv1D`, `PreTrainedModel`,
  `ModelOutput` classes, `ROPE_INIT_FUNCTIONS["default"]`, `StaticCache`
  (per-layer K/V states `[kv_heads, max_cache_len, head_dim]` written at
  `cache_position`) and `StaticCacheDecoder(model)`, a `forward(token, position)`
  wrapper that calls the causal LM with `input_ids`, `position_ids`,
  `cache_position` and `past_key_values` and returns `logits`, keeping the
  model's checkpoint paths.

`tests/hf_style_models.py` is Llama (`LlamaRMSNorm` with the float32 round trip,
`LlamaRotaryEmbedding` with an `inv_freq` buffer, `rotate_half`, `repeat_kv`,
eager and sdpa attention with the additive static-cache causal mask, SwiGLU via
`ACT2FN`) and GPT-2 (`Conv1D`/`addmm`, the `bias` mask buffer,
`NewGELUActivation`) written as in transformers with only the imports changed.
Both trace into the kernels of the hand-written examples: Llama gives the same
FIR kernels as `examples/llama.py` (rmsnorm, attention with RoPE and grouped
heads, add_rmsnorm, silu_mul, add) and GPT-2 the same kernels as
`examples/gpt2.py` (the attention kernel differs only by the StaticCache
layout), with the layer stack as one `repeat layers` and HF checkpoint names.

`TensorSpec` and state shapes accept `Dim("width", 4)` or config symbol names.
Python loops and branches over static values run during tracing; data-dependent
control flow raises an error (use `where` or masks).

State: `register_state(name, shape)` creates persistent F32 storage; `read()` is
an SSA snapshot; `update(value)` replaces it; `update(value, index, dim=d)`
replaces one slice at a scalar index (KV cache rows).

## Operations and kernels

| Traced operation | Emitted form |
| --- | --- |
| `embedding` of the token/position role | `ext fir.embedding` |
| single-row `linear` `[in,out]` (f32/q4), `linear_t` `[out,in]` (f32) | `ext fir.linear`, `ext fir.linear_t` |
| other linear/matmul (batched, broadcast) | `script.matmul` / `script.linear`: lane over the largest output dim, runtime loop over the contraction |
| add, sub, mul, div, neg, pow (integer exponent), comparisons, where, maximum, minimum, abs | pointwise kernel; `where` copies the selected value (`cmplt` + `ifbegin`) |
| relu, silu, sigmoid, tanh, exp, softplus, rsqrt, sqrt | pointwise; tanh via sigmoid, sqrt via rsqrt |
| gelu (`approximate="tanh"` → `gelu_new`), exact erf `gelu` (`"none"`), `gelu_quick` | tanh form via sigmoid; erf via Abramowitz–Stegun 7.1.26 (abs. error ≤ 1.5e-7); quick = x·sigmoid(1.702x) |
| log, sin, cos | log: eight square roots plus atanh series (positive inputs); sin/cos: reduction to [-π, π] plus truncated Taylor series |
| sum, mean, max | one lane per kept row, runtime loops over reduced dims |
| softmax | one lane per row: max, exp/sum, normalize |
| layernorm, rmsnorm | one lane per row, three (two) passes over the row |
| cumsum, scan | one lane per row, running sum loop |
| index_select | dynamic-offset gather, one lane per index |
| reshape, permute, slice, expand, grouped cat | no kernel: strided reads of the base buffer |
| irregular cat, non-strided reshape, view needed as a buffer | copy kernel |
| conv1d | lane per output channel, loops over input channels and taps; selected time steps only |
| rotary | lane per row, loop over pairs (i, i + D/2) |
| state_read / state_write | in place when no stale version is read; reset kernel at position 0 |
| non-uniform constants | uniform → literal, affine (`arange`) → loop counters, others → constant buffer written by an init node |

Kernels contain runtime loops (`forbegin` / `forend` over count registers), so
their length depends on tensor rank and fused expression size, never on element
counts. One dimension of each iteration space is spread over threads
(`threads=` the lane count, or `threads=out0.size` when it equals the output
size, which lets equal bodies of different widths share one kernel); the other
dimensions are serial loops with per-operand offset registers (shared when two
operands advance identically, coalesced when contiguous). View chains compose into
`offset + Σ (index / group) · stride` descriptors, so reshape/permute/slice,
broadcast and grouped-query head expansion are read without copies. Negative
strides are emitted as 32-bit two's complement immediates. Identical kernel bodies
are emitted once.

## Pattern passes

`PASSES` runs in this order: `eliminate_dead_code`, `constant_fold`,
`common_subexpressions`, `fold_views`, the re-fusion passes
(`refuse_additive_masks`, `refuse_rotary`, `refuse_linear`, `refuse_norms`,
`refuse_activations`), `fuse_attention`, `fuse_rotary_attention`,
`push_slice_into_conv`, `fuse_conv_state`, `fuse_pointwise`,
`fuse_state_update_reduction`, `fuse_reduction_prologue`, `fuse_norms`,
`select_q4` (hints), `deduplicate_kernels` (hints). Every pass is deterministic and
leaves unmatched subgraphs to the generic kernels.

The re-fusion passes rewrite decomposed torch/HF idioms into the operations the
hand-written examples trace directly:

| Idiom | Rewritten to |
| --- | --- |
| `scores + min · (arange(P) > position)` (HF static-cache causal mask, broadcast views included), `where(arange(P) > position, min, scores)` | `where(arange(P) <= position, scores, -big)` |
| `q·cos + rotate_half(q)·sin` where cos/sin come from the position input (any computation: `inv_freq @ position`, outer, einsum, `cat(freqs, freqs)`) and equal the default `base^(-2i/D)` table at probe positions | `rotary(q, position, base)` (times `attention_scaling` when not 1) |
| `x @ W.T` with a checkpoint `W [out,in]`; `x @ W` with `W [in,out]` (GPT-2 `Conv1D`); `view(linear(view(x, -1, in)) + b, x.shape[:-1] + (out,))` | `linear_t`; `linear`; the linear layer on `x` with the bias add left for pointwise fusion |
| `x · rsqrt(mean(x·x or x², -1, keepdim) + eps)` or `x / sqrt(…)`, then `· weight` | `rmsnorm` with the weight |
| a single-input smooth pointwise subgraph equal to `gelu_new`, `gelu`, `gelu_quick`, `silu`, `sigmoid`, `tanh` or `softplus` at probe points (to 1e-9) | that activation (HF `NewGELUActivation`, `FastGELUActivation`, `QuickGELUActivation`, `x·sigmoid(x)`) |

| Pattern in the traced graph | Fused op | Kernel |
| --- | --- | --- |
| `softmax(where(arange(P) <= position, scale·q@kᵀ, -big)) @ v` with k/v from indexed cache writes at `position`; MHA, GQA (slice/cat or expand/reshape), any cache layout expressible as a strided view (`[H,P,D]`, `[P,H·D]`, …) | `attention_decode` | `script.attention`: one lane per KV head, loop over the group's query heads; writes this token's K/V row at `position`, then scores with a running maximum, exponentials and weighted values over slots `0..position` only |
| `rotary_embedding(q, position)` and `rotary_embedding(k, position)` feeding that attention | `attention_decode` + `rope` | the same kernel rotates q into scratch and k while writing the cache row |
| `cat(state.read(), x)`, `state.update(window[..., 1:])`, depthwise `conv1d(window)[..., -1]` | `conv1d_step` | `script.conv_step`: lane per channel; dot with the history, shift in place, append the sample; history cleared at position 0 |
| full state update `h' = f(h, …)` (pointwise) followed by `sum/mean(g(h', …))` | `state_update_reduce` | `script.ssm_step`: lane per state row, the update and the reduction in one loop; state zeroed at position 0 |
| single-use elementwise DAGs: bias adds, residual adds, activations, SwiGLU `silu(gate)·up`, gates | `fused` (expression tree, ≤ 8 inputs, ≤ 40 ops) | one pointwise kernel named after its ops (`add_gelu_new`, `silu_mul`, …) |
| elementwise producer of LayerNorm/RMSNorm (residual + norm), the sum kept when reused | `norm` (`extra` = residual output) | `script.add_layernorm` / `script.add_rmsnorm` with two outputs |
| elementwise producer of a reduction | reduction with `tree` | reduction kernel evaluating the tree inline |
| slice of a convolution along time | conv1d with `times` | computes only the kept time steps |

## Emission

- `extract_repeats` folds one homogeneous layer stack with one activation carry
  into `repeat layers`, `{i}` checkpoint templates and `per_layer` buffers.
  States owned outside the layer modules (an HF `StaticCache` holds
  `past_key_values.layers.<i>.keys`) are matched through their layer-index
  segment. When
  the last node of a layer reads the carry other than pointwise, its result goes
  through a temporary and a copy.
- Persistent state: if no stale SSA version is read after a write, all versions
  share the state buffer and writes happen in place (indexed row writes for KV
  caches, the producing kernel writes a full update directly). Otherwise state
  updates are emitted with copy semantics. Reads become a reset kernel that zeroes
  the state at position 0, except when the first write replaces the whole state.
  Fused attention needs no reset: it writes each slot before reading it and reads
  only slots `0..position`.
- The header contains `layout script`, the epsilon key matching the norms
  (`layer_norm_epsilon`, `rms_norm_eps`, …), `sym layers` bound to the layer
  config key when one matches, and `require sym KEY = VALUE` for every positive
  integer config key except token ids, since kernels are specialized to them.
- Names come from module scopes and checkpoint names (`ln_1_norm`,
  `c_attn_weight`, `key_cache`, `script.attention`), so exports are byte-identical
  across runs and small model edits give small diffs.

Weights: rank-1 F32 weights and (with `layout script`) F32 weights of rank other
than 2 are arena-bound under their checkpoint name and full shape. Rank-2 weights
are resident and serve the external operations. A rank-2 weight read by a micro
kernel is declared as `SOURCE.fir_flat` `[elements]` with a `# checkpoint-map`
comment; that tensor is a row-major copy supplied with the checkpoint. q4 applies
to `fir.linear` (`[in,out]`) only.

## Kernel op tags

`fir.export(..., tags=True)` (and `--tags` on the GPT-2 and TinyLlama example commands) adds `tag=` to every
kernel header: `aten::scaled_dot_product_attention` for attention, `aten::native_layer_norm`,
`aten::rms_norm`, `aten::add+aten::gelu` for fused kernels, `aten::<op>` for generic kernels and
`fir::rotary_embedding` / `fir::ssm_step` for ops without an ATen name (table `KERNEL_TAGS` in
`emitter.py`). Tags are off by default so existing exports are unchanged; the runner accepts them only
with `layout script` and carries them into probe logs.

## Examples

| Example | Contents | Fused kernels |
| --- | --- | --- |
| `examples/gpt2_125m.py` | HF GPT-2 125M names (`prefix transformer.`), q4 linears, tied f32 head, `[n_positions, n_embd]` KV caches per layer, the hand-written file's `require config` lines | add, layernorm, attention, add_layernorm, add_gelu_new, add (3 inputs) |
| `examples/gpt2.py` | tiny 2-layer GPT-2, same layout | same |
| `examples/llama.py` | RMSNorm, RoPE, GQA (`num_key_value_heads`), SwiGLU, untied `lm_head` | rmsnorm, attention (+RoPE, grouped), add_rmsnorm, silu_mul, add |
| `examples/mamba.py` | Mamba-1 style block: in_proj → causal depthwise conv with state → SiLU → x_proj → softplus(dt_proj) → selective scan step `h = h·exp(dt·A) + dt·B·u`, `y = Σ h·C + D·u` → `y·silu(z)` → out_proj; flat config keys | rmsnorm, conv_step, ssm_step, silu, add_softplus, silu_mul_add, add |

GPT-2 125M (real size) against the hand-written `models/gpt2-125m/model.fir`:

| | emitted (optimized) | hand-written | emitted (`--no-optimize`) |
| --- | --- | --- | --- |
| emission time (trace + passes + emit) | 0.19 s | – | 0.04 s |
| kernels | 6 | 4 | 13 |
| kernel instructions (total) | 186 | 123 | 196 |
| nodes | 15 (8 micro, 7 ext) | 19 (12 micro, 7 ext) | 29 |
| dispatches per token | 74 micro + 51 ext | 122 micro + 51 ext | 242 micro + 51 ext |
| buffers / weights | 20 / 16 | 14 / 16 | 28 / 16 |
| attention VM steps at position 1023 (one layer) | 11,258,580 | 11,291,436 | – |

| kernel | instructions | | hand-written | instructions |
| --- | --- | --- | --- | --- |
| `script.add` | 5 | | `g2.add` | 5 |
| `script.layernorm` | 35 | | `g2.layernorm` | 33 |
| `script.add_layernorm` (residual + norm, 2 outputs) | 40 | | – | – |
| `script.add_gelu_new` (bias + GELU) | 12 | | `g2.gelu_new` | 10 |
| `script.add_2` (bias + residual) | 7 | | – | – |
| `script.attention` | 87 | | `g2.attn_decode` | 75 |

The weight declarations (checkpoint names, dtypes, shapes), the `prefix`, the
`require config` lines, the `key_cache`/`value_cache` buffers
(`n_positions·n_embd`, `per_layer`) and the `layers`/`positions`/`vocab` symbols
are identical to the hand-written file, so the emitted file loads the same
checkpoint. The emitted graph fuses bias, residual and norm steps that the
hand-written file runs as separate `g2.add` nodes, which gives two more kernel
bodies and 48 fewer dispatches per token; the extra buffers are the fused
kernels' outputs (6,144 more arena floats).

## Validation

`fir.FirModel` executes FIR graph files in Python the way the runner does:
symbols from integer config keys, `sym`/`require`, buffer bindings in one f32 arena
(per-layer slices), `prefix` resolution, kernels assembled from mnemonic lines and
nodes executed in order with `repeat layers`. The interpreter mirrors
`vm_interp.nim`: R0 holds the thread id, float results are rounded to f32 after
every operation, integer registers wrap at 32 bits, `forbegin` re-reads its count.
Each dispatch is limited to 2²⁴ steps (the CPU VM limit), and every load and store
is checked against the buffers bound to the node's ports, stores against output
ports only. Interpreted mnemonics: all arithmetic, `cmplt`, loops, conditionals,
`uconst`/`uaddk`/`umulk`/`uadd`/`umul`/`uandk`/`ushrk`/`ushlk`, `rtou`/`utor`,
`uloadf32`/`ustoref32`, `loadconstf32`, `fma`. Thread-group, atomic and legacy
R-indexed memory mnemonics (`barriertg`, `loadlid`, `loadtgid`, `utgid`, `ulid`,
`uatomicadd`, `loadf32`, `storef32`, `tgloadf32`, `tgstoref32`) are rejected with an
error. Resident externals are computed in Python; q4 weights are used as f32.

`tests/test_fir_numerics.py` decodes tiny random-weight 2-layer GPT-2, Llama and
Mamba models for two sequences (the arena is not cleared in between, so resets at
position 0 are exercised) and requires agreement within 1e-4 between:

- the raw and the optimized graph (`compare_graphs`, outputs and states);
- the reference evaluator and independent straight-line Python decoders;
- the emitted FIR files (optimized and `--no-optimize`) run by `FirModel` and the
  straight-line decoders;
- the hand-written `models/gpt2-125m/model.fir` on the tiny config, run by
  `FirModel`, and the straight-line GPT-2 decoder;
- Llama variants: MHA and GQA, with and without RoPE.

`tests/test_fir_frontend.py` checks the tensor API against Python values
(eager and traced), buffers and the module registry, `PretrainedConfig` and
`ACT2FN`, the functional ops (norms, sdpa options, `conv1d` padding and stride
against a direct convolution), extra inputs and multiple outputs (evaluated, and
rejected by export), and `from_torch` against a fake `torch.fx` object model
(leaf modules, aten overloads, methods, buffers, error messages). The HF-style
Llama (eager and sdpa, MHA and GQA) and GPT-2 decode two sequences within 1e-4
of the straight-line decoders, for the raw/optimized graph comparison, the
reference evaluator and the emitted FIR in `FirModel`, and their exports pass
`--fir-check` and match the example kernels.

Observed maximum differences are below 1e-7 for the VM runs and below 1e-15
between the reference evaluator and the straight-line decoders. `tests/test_fir.py` exports synthetic
operator graphs (views, negative strides, matmul shapes, convolutions, scans,
state ordering, every activation, standalone RoPE), checks the FIR grammar, runs
`formers-cli --fir-check` when the CLI is built and executes every export in
`FirModel` against `fir.evaluate`. `tests/test_gpt2_125m.py` checks real-size
emission time, determinism, drop-in declarations against the hand-written file,
the `--report` mode and `--fir-check` of the optimized and unoptimized real-size
exports.

```sh
cd other/formers.n/python
python3 -m unittest
```

## Limits

- Export targets the scalar-token decode ABI (one token, one position per step);
  prefill/batched inputs are evaluated by `fir.evaluate` but not exported.
- One `repeat layers` stack per file; a later layer's weight used outside the
  stack or a pre-stack residual used after it keeps the stack unrolled.
- Kernels use at most 10 input and 10 output ports, 31 float and 32 integer
  registers and 8 nested loops; larger fused expressions are split by the fusion
  limits, other overflows raise `LoweringError`.
- The CPU VM allows 2²⁴ steps per dispatch. GPT-2 125M attention at position 1023
  takes 11.3 M steps; longer contexts or larger head counts per kernel exceed it
  on the CPU backend.
- Gathers (`index_select`) and indexed state writes do not bounds-check indices.
- `pow` needs a constant integer exponent in [-32, 32]; `log` expects positive
  inputs; sin/cos use series after range reduction (accuracy decreases for very
  large arguments).
- Convolutions are stride 1, dilation 1, causal with left zero padding. The
  generic `conv1d` kernel unrolls the computed time steps; streaming models use the
  `conv1d_step` pattern (depthwise, kernel size ≥ 2, batch 1).
- The attention pattern requires the cache mask `arange(n_positions) <= position`
  and indexed cache writes at `position`; other attention forms use the generic
  matmul/softmax kernels.
- Rank-2 weights read by micro kernels need the flattened `SOURCE.fir_flat`
  tensor; q4 is available for `[in,out]` `fir.linear` only (Llama-style `[out,in]`
  projections stay f32).
- Export accepts token and position inputs and one output; graphs with other
  inputs or several outputs trace and evaluate, and `export` raises
  `LoweringError` naming the inputs.
- Runtime integer arithmetic (rounding, remainder), runtime `sort`/`topk`/`prod`,
  gather with runtime indices, scatter other than state slice writes, sampling,
  dynamic shapes, data-dependent control flow and autograd are not traced.
- `fir.from_torch` replays an FX graph against `torch_compat`: `call_function`
  targets resolve by name (Python operators, `torch.*`, `torch.nn.functional.*`,
  `aten.*` with aten argument orders for `_softmax`, `embedding`, `slice`,
  `rsub`, `_to_copy`, `addmm` and the view/copy aliases), `call_method` uses the
  tensor method, and `call_module` maps Linear, Embedding, LayerNorm, RMSNorm,
  Conv1d, Dropout, Identity, Flatten, GELU, Softmax and the parameter-free
  activations. Parameter payloads are never read; buffers are read as constants.
  Other targets raise `ValueError` naming them. It is tested with a fake
  `torch.fx` object model, since torch is optional.
