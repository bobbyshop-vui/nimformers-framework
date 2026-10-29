
# Getting Started

[English](getting-started.md) | [Tiếng Việt](getting-started.vi.md)


## Requirements

- Nim compiler (the project is written in Nim).
- A supported GPU runtime for the selected backend: Metal on macOS or Vulkan with a working Vulkan loader/ICD. Some operations and session encoders are backend-specific.
- Python/Transformers only for Hugging Face configuration/tokenizer/model conversion paths that call the Python ecosystem.


## Import the facade

`nimformers.nim` re-exports the core model, backend, quantization, custom-float, tensor-ops, BybyLang session, and micro-VM APIs.

```nim
import nimformers

proc main() =
  let ctx = newBackend("auto")
  defer: closeBackend(ctx)

  let a = @[1'f32, 2, 3, 4]
  let b = @[10'f32, 20, 30, 40]
  echo beAdd(ctx, a, b)

main()
```

Save this as `your_app.nim` at the repository root, then build/run with the
same module search paths used by the Makefile:

```sh
nim c -r -d:release --mm:arc --opt:speed \
  --path:. --path:test/codestral --path:test/deepseek-inference \
  --path:test/spectron --path:kernelfortest your_app.nim
```

The backend modules provide their platform linker directives. For the supplied
programs, use the [Make command guide](make-commands.md). `make build` builds
the Spectron trainer at `test/spectron/test_nimformer`; `make run` also starts
training. Running bare `make` starts training as well.


## Backend names

```nim
let ctx = newBackend("metal")  # macOS Metal
let ctx = newBackend("vulkan") # Vulkan
let ctx = newBackend("auto")   # probe available GPU backends
```

The legacy string `"cuda"` resolves to Vulkan in this framework; it does not enable CUDA execution. `newBackend("cpu")` is not a supported backend in the framework facade. If no GPU is available, `"auto"` raises rather than silently running the framework on CPU.

Some lower-level BybyLang operations retain CPU reference/fallback behavior. That does not make CPU a supported `Backend` choice for model execution.


## Where to go next

- Use the [Spectron walkthrough](spectron-guide.md), [Python walkthrough](python-guide.md), or [BybyLang walkthrough](bybylang-guide.md) for complete examples.
- Use [Backend API](backend-api.md) for GPU kernels.
- Use [Base Model API](base-model-api.md) for training-oriented transformer layers.
- Use [LLaMA Inference API](llama-inference-api.md) for Hugging Face-style decoder inference.
- Use [BybyLang GPU API and DSL](bybylang-api.md) to write GPU programs or DSL demos.
