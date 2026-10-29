
# Use Nimformers from Python

[English](python-guide.md) | [Tiếng Việt](python-guide.vi.md)

The `nimformers/` package uses `ctypes` to call the shared library built from
`lib_c_api.nim`. It exposes NumPy operations, owned tensors, quantization,
NIMQ, resident weights, micro-VM, tokenizers, and Llama inference.


## Install from source

From the repository root, create an environment and build the library:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
make build-for-python
make python-example PYTHON=python
```

`pip install -e '.[test]'` installs the wrapper, NumPy, and pytest; it does
not compile Nim. `make build-for-python` produces `libnimformers.dylib` on
macOS, `libnimformers.so` on Linux, or `nimformers.dll` on Windows. GPU work
needs Nim, a C compiler, and a Metal/Vulkan runtime.

For package use without tests, `python -m pip install -e .` is sufficient.
See the [Python API](python-api.md) for the full reference.


## Select a backend and find the shared library

```python
from nimformers import Context, library_path

print(library_path())
with Context(backend="auto", forbid_cpu_fallback=True) as ctx:
    print(ctx.backend, ctx.abi_version, ctx.version)
```

`auto` detects a GPU; choose `metal` on macOS or `vulkan` explicitly.
`cuda`/`opencl` redirect to Vulkan. There is no `Context("cpu")`.
`forbid_cpu_fallback=True` raises if an operation cannot execute on the
GPU backend. This is not an autograd system like PyTorch.

When using another project, install the wrapper and point to the library file:

```sh
# macOS example; use libnimformers.so on Linux.
export NIMFORMERS_LIB="$PWD/libnimformers.dylib"
```

Alternatively pass `library="/absolute/path/libnimformers.dylib"` to `Context`.
The loader also searches the package, repository, current directory, and `build/`.


## First operations

Save as `python_demo.py` at the repository root and run `python python_demo.py`:

```python
import numpy as np
from nimformers import Context

with Context(forbid_cpu_fallback=True) as ctx:
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    b = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    y = ctx.matmul(a, b)
    np.testing.assert_allclose(y, [[4, 5], [10, 11]])
    print(y.shape)                         # (2, 2)
    p = ctx.softmax(y)
    print(p.sum(axis=-1))                   # approximately [1, 1]
    gamma = np.ones(2, dtype=np.float32)
    beta = np.zeros(2, dtype=np.float32)
    print(ctx.layernorm(y, gamma, beta))
```

Methods accept array-like inputs, convert them to contiguous `float32`, and
usually return NumPy arrays. `matmul` accepts `[M,K] @ [K,N]` and supports
batch-axis broadcasting. `matmul_nt(x, weight)` takes `[N,K]` weights and
computes `x @ weight.T`.

Common API families:

- Arithmetic: `add`, `sub`, `mul`, `div`, `relu`, `sigmoid`, `tanh`, `exp`, `log`.
- Neural networks: `softmax`, `layernorm`, `layernorm_backward`, `attention`,
  `attention_backward`, `embedding_lookup`.
- Reductions/layout: `sum`, `mean`, `max`, `argmax`, `permute`, `concat`, `split`.
- Scans/masks: `cumsum`, `cumprod`, `affine_scan`, `where`, `masked_fill`, `dropout`.

`attention` accepts `q/k/v` in `[batch, heads, seq, head_dim]` layout and
returns `(out, scores)`. Current fused Metal/Vulkan kernels are causal;
they do not apply an arbitrary supplied `mask`.


## Tensors and memory lifetime

```python
import numpy as np
from nimformers import Context

with Context() as ctx:
    with ctx.tensor(np.arange(12, dtype=np.float32).reshape(3, 4)) as tensor:
        print(tensor.shape, tensor.numel)   # (3, 4), 12
        tensor.reshape(2, 6)
        owned_copy = tensor.numpy()        # copy remains valid after close
        print(owned_copy.shape)            # (2, 6)
```

`ctx.tensor`, `ctx.zeros`, and `ctx.randn` create handles requiring `close()`
or `with`. `tensor.numpy(copy=False)` borrows memory; use it while the tensor
is open. Close resident GPU handles before the context. Operations such as
`ctx.matmul(...)` return independent NumPy arrays, which need no `close()`.


## Quantization and resident weights

This standalone example creates `[N,K]` weights, quantizes them, and runs linear:

```python
import numpy as np
from nimformers import Context

rng = np.random.default_rng(42)
x = rng.standard_normal((2, 64)).astype(np.float32)
weight = rng.standard_normal((32, 64)).astype(np.float32)

with Context() as ctx:
    with ctx.quantize(weight, "int4_asym") as quant:
        restored = quant.dequantize()
        y = ctx.linear_q4(x, quant)         # (2, 32)
        np.testing.assert_allclose(y, x @ restored.T, rtol=1e-4, atol=1e-3)
        with ctx.resident_q4(quant) as resident:
            print(resident.linear(x).shape) # upload the weights once
```

Kinds are `fp32`, `int8`, `int4`, `int4_asym`, `fp8_e4m3`, `fp8_e5m2`,
`custom`, `auto`, and `fp4_km`. `fp4_km` requires 2D tensors with a last
dimension divisible by 32. `resident_q4` accepts `int4_asym`; `linear_q4`
dequantizes other kinds. `resident_f32` uses `[K,N]` weights, unlike
`resident_q4`'s `[N,K]` layout.


## Inspect a Spectron checkpoint in Python

```python
from nimformers import Context

with Context() as ctx:
    with ctx.load_state_dict("test/spectron/models/finetune.nimq", mapped=True) as sd:
        print("arch:", sd.arch)
        print("tensors:", len(sd), sd.names[:5])
        print("embedding:", sd["embed.weight"].shape)
        if "__meta_step__" in sd:
            print("trained step:", int(sd["__meta_step__"].dequantize()[0]))
```

`mapped=True` memory-maps checkpoint payloads; dequantize individual tensors
when needed. Spectron uses `arch=(vocab, embedDim, nHeads, nLayers, ffMult)`.
NIMQ stores five writer-defined architecture integers; consult the appropriate
loader for other checkpoint types.

`sd[name]` is a cached `QuantTensor` handle closed with its state dict.
Use borrowed NumPy payload views before closing it. Avoid `sd.to_dict()`
for large models when you only need metadata: it dequantizes every tensor.
Reading Spectron NIMQ does not instantiate a Python Spectron model class.

To create a state dict, use `ctx.create_state_dict(arch=(...))`, assign
`QuantTensor` objects from `ctx.quantize(...)` through `sd["name"] = quant`,
then call `sd.save(path)`.


## Write a micro-kernel in Python

```python
import numpy as np
from nimformers import Context

with Context() as ctx:
    with ctx.vm_program() as prog:
        prog.load_f32(1, 0, 0)              # R1 = memory[R0], R0 = thread id
        prog.add(1, 1, 1)
        prog.store_f32(1, 0, 0)
        prog.halt()
        data = np.arange(1, 65, dtype=np.float32)
        expected = data * 2                 # compute first because vm_run modifies data
        out = ctx.vm_run(prog, data, n_threads=64, tg_size=64)
        np.testing.assert_array_equal(out, expected)
```

`VMProgram` can save/load `.bvm` and lower to MSL/GLSL. The program is
responsible for buffer indexing and dispatch size; this example has exactly
64 elements and 64 threads. See the [micro-VM API](microkernel-api.md).


## Model inference and invoking trainers

Python provides `ctx.load_llama(configPath, nimqPath)` and `model.generate(...)`
for Llama/DeepSeek models compatible with the loader. See
[Python API — Llama inference](python-api.md#llama-inference) for an example.

Spectron and Codestral LoRA use their Nim/CLI entry points for their own
training/inference. There is no Python `SpectronModel.fit`, automatic
`backward()`, or model-level Adam optimizer. Python can invoke a trainer
with `subprocess.run(["make", "run-metal"], cwd=repo, check=True)`; Spectron
configuration remains in source constants. See [Spectron](spectron-guide.md).


## Tests and troubleshooting

```sh
make python-test PYTEST='python -m pytest' PYTEST_FLAGS=-q
```

- `ModuleNotFoundError: nimformers`: install into the correct interpreter
  with `python -m pip install -e .`, or run from the repository root.
- Shared library missing: run `make build-for-python`, inspect
  `library_path()`, or set `NIMFORMERS_LIB` to the full file path.
- Missing `nimf_*` symbols: rebuild and check for an older selected library.
- No GPU/Metal access: check the runtime and process access to the device;
  the package does not provide a replacement CPU backend.
- C ABI shape/backend errors become `NimformersError`; Python input
  validation usually raises `ValueError` or `TypeError`.

Pytest fixtures can skip if context creation fails. Inspect passed/skipped
counts before concluding GPU tests ran. Serialize GPU calls on a shared context.
