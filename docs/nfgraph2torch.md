# nfgraph2torch (torch graph → nimformers)

`other/nfgraph2torch` converts a torch module into a nimformers FIR graph and writes `model.fir`.
It is Python. The reverse direction (`to_torch`) is described below.

```python
import nfgraph2torch as nf2t

graph = nf2t.from_torch(model, (x,))                 # a formers.fir graph
nf2t.export(model, (token, position), "out_dir")     # out_dir/model.fir + config.json
values, shape = nf2t.evaluate(graph, model, (x,))    # Python reference run, to compare with torch
print(nf2t.coverage(model, (x,)))                    # which ATen ops are needed and how each is lowered
```

```bash
python -m nfgraph2torch package.module:make_model out_dir
python -m nfgraph2torch package.module:make_model --coverage
```

`make_model` returns `(torch_module, example_inputs)`. `nfgraph2torch` finds the `formers` package in
`other/formers.n/python`; set `NFGRAPH2TORCH_FORMERS` to use another location.

## API

| Function | Meaning |
|---|---|
| `from_torch(module, example_inputs, *, inputs=None, kwargs=None, config=None, name, optimize=True, strict=False, decompositions=None)` | Returns a `formers.fir` graph. `inputs` is a list of `Input(role, dtype)`, one per user input (`role` is `"token"`, `"position"` or `""`). `config` is a dict or an object with `to_dict()`. |
| `export(module, example_inputs, directory, *, report=False, **options)` | `from_torch` plus `fir.export`; returns the path of `model.fir`. |
| `evaluate(graph, module, example_inputs)` | Runs the graph with the FIR reference evaluator using `module.state_dict()`. Returns `(flat_values, shape)`. |
| `coverage(module, example_inputs=None, **options)` | Report of the ATen ops in the module and how each is lowered. |
| `UnsupportedOp` | Raised for an ATen op with no kernel and no micro-kernel; the message names the op and the node. |

## How it works

1. `torch.export` captures the module and `run_decompositions()` lowers it to Core ATen (about 160 ops).
2. Each node is replayed on the `formers.fir` tracer. ATen ops FIR already has become ordinary FIR
   kernels. Missing ops become micro-kernels, or are composed from existing ops.
3. `fir.export` writes `model.fir` and `config.json`. Parameters keep their `state_dict` names and are
   never read; buffers and constants are embedded.

## Micro-kernels

Registered through the custom-op hook of `formers.fir` (`formers/fir/custom.py`):

- `prod`, `cumprod`, `scatter` / `scatter_add`, `sort` / `argsort` / `topk`, `trunc`.
- `window`: N-d convolution (stride, padding, dilation, groups) and max/avg pooling.
- Compositions: floor, ceil, round, fmod, remainder, floor_divide, erf, norm, reflect/replicate pads,
  slice_scatter, index, index_put, adaptive avg pool, nearest upsample.

### Custom-op hook (`formers.fir.custom`)

| Function | Meaning |
|---|---|
| `register_op(name, *, reference, lower, replace=False)` | Add a graph op. `reference(inputs, shapes, out_shape, attrs)` returns the flat result (used by `fir.evaluate`); `lower(emitter, node, ctx, kernel_builder)` generates the micro-kernel and returns `(kernel_name, threads)`. |
| `custom_op(name, inputs, shape, attrs=None, dtype="f32")` | Record the op on the active trace and return the result tensor. Parameters among `inputs` become checkpoint weights. |
| `register_tree_op(kind, emit)` | Add a scalar operation to the pointwise kernel builder. |

## Reverse direction: model.fir → torch

```python
back = nf2t.to_torch("out_dir/model.fir", weights=dict(model.state_dict()))   # a torch.nn.Module
logits = back(token, position)
```

`to_torch(source, weights=None, config=None)` reads a `runtime graph` `model.fir` (path or text) and
returns `FirTorch`, a module whose `forward(token, position)` returns the logits. `weights` maps
checkpoint tensor names to tensors; they become `nn.Parameter`s with the same names. Only elementary
ATen operations are used, so any FIR file the Python VM can run also runs in torch:

- `ext` nodes: `fir.embedding` is an `index_select`; `fir.linear` / `fir.linear_t` are `torch.matmul`.
- `micro` nodes: a vectorised interpreter runs the kernel bytecode with one tensor entry per thread.
  `add`/`sub`/`mul`/`div`/`exp`/`sigmoid`/`rsqrt`... are the same ATen op, `uloadf32` is a gather,
  `ustoref32` an `index_put_`, `ifbegin` a mask (`where`) and `forbegin` a Python loop over tensor ops.

Speed is far below the native runner (a convolution costs one tensor op per multiply-accumulate; a
ResNet-18 with 8 base channels on a 32×32 image takes about 35 s on CPU). Coverage is complete for
everything the Python VM interprets; kernels using thread-group, atomic or barrier instructions are
rejected, as in the VM. Stores to the same address from several threads have no defined order.

A torch → FIR → torch round trip of a ResNet-18 (wrapped as a token → image → logits model, because
the runner ABI only accepts `token` and `position`) matches the original to 4e-7.

## Limits

- The FIR runner feeds only `token` and `position` and reads `logits`, so `model.fir` export needs an
  LM-shaped graph. Other graphs (for example an image classifier such as ResNet-18) convert to a FIR
  graph and run on the reference evaluator, but cannot be exported to a runnable `model.fir` yet.
- `sort` / `topk` are O(n²); not meant for huge vocabularies.
- Not lowered: transposed convolution, `ceil_mode` pooling, uneven adaptive pooling, bilinear upsample,
  grid_sampler, as_strided, col2im, multi-index advanced indexing, inverse trig, `scatter_reduce`
  other than sum.

## Tests

```bash
cd other/nfgraph2torch/tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest test_ops.py test_resnet.py test_reverse.py   # reference evaluator
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest test_vm.py                   # model.fir on the FIR VM
```
