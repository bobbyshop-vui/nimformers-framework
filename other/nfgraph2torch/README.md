# nfgraph2torch

Convert between torch graphs and nimformers FIR. Direction implemented: **torch graph → nimformers**
(`model.fir`). Direction **model.fir → torch** is `nf2t.to_torch(path, weights)`: ext nodes become `index_select` and `torch.matmul`, micro-kernels run on a vectorised bytecode interpreter built from ATen ops (slow, complete). See docs/nfgraph2torch.md.

```python
import nfgraph2torch as nf2t

graph = nf2t.from_torch(model, (token, position), inputs=[nf2t.Input("token"), nf2t.Input("position")])
nf2t.export(model, (token, position), "out_dir")      # out_dir/model.fir + config.json
print(nf2t.coverage(model, example_inputs))           # which ATen ops are needed and how each is lowered
```

```bash
python -m nfgraph2torch package.module:make_model out_dir
python -m nfgraph2torch package.module:make_model --coverage
```

`make_model` returns `(torch_module, example_inputs)`.

## How it works

1. `torch.export` captures the module, `run_decompositions()` lowers it to Core ATen (about 160 ops).
2. Each node is replayed on the `formers.fir` tracer. ATen ops FIR already has become normal FIR
   kernels. Missing ones become micro-kernels (below) or are composed from existing ops.
3. `fir.export` writes `model.fir` and `config.json`. Parameters keep their `state_dict` names and
   are never read; buffers and constants are embedded.

## Micro-kernels (registered through `formers.fir.custom`)

`prod`, `cumprod`, `scatter` / `scatter_add`, `sort` / `argsort` / `topk`, `trunc`, and `window`
(N-d convolution, max/avg pooling). Compositions: floor, ceil, round, fmod, remainder,
floor_divide, erf, norm, reflect/replicate pads, slice_scatter, index, index_put, adaptive avg pool,
nearest upsample.

## Limits

- The FIR runner feeds only `token` and `position` and reads `logits`, so `model.fir` export needs an
  LM-shaped graph. Other graphs convert to a FIR graph and run on the reference evaluator.
- sort/topk are O(n²); not meant for huge vocabularies.
- Not lowered: transposed convolution, `ceil_mode` pooling, uneven adaptive pooling, bilinear
  upsample, grid_sampler, as_strided, col2im, multi-index advanced indexing, inverse trig,
  `scatter_reduce` other than sum. An unsupported node raises `UnsupportedOp` naming the ATen op.

## Tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_ops.py   # reference evaluator
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/test_vm.py    # model.fir on the FIR VM
```
