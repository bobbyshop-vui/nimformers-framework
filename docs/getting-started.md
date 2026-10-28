# Getting Started

## Requirements

- Nim compiler (the project is written in Nim).
- A supported GPU runtime for the selected backend: Metal on macOS or Vulkan with a working Vulkan loader/ICD. Some operations and session encoders are backend-specific.
- Python/Transformers only for Hugging Face configuration/tokenizer/model conversion paths that call the Python ecosystem.

## Import the facade

`nimformers.nim` re-exports the core model, backend, quantization, custom-float, tensor-ops, BybyLang session, and micro-VM APIs.

```nim
import nimformers

let ctx = newBackend("auto")
defer: closeBackend(ctx)

let a = @[1'f32, 2, 3, 4]
let b = @[10'f32, 20, 30, 40]
echo beAdd(ctx, a, b)
```

Build an application from the repository root with `nim c --path:. your_app.nim` (add the project’s required C/Objective-C framework flags when bypassing the provided build scripts on macOS).

## Backend names

```nim
let ctx = newBackend("metal")  # macOS Metal
let ctx = newBackend("vulkan") # Vulkan
let ctx = newBackend("auto")   # probe available GPU backends
```

The legacy string `"cuda"` resolves to Vulkan in this framework; it does not enable CUDA execution. `newBackend("cpu")` is not a supported backend in the framework facade. If no GPU is available, `"auto"` raises rather than silently running the framework on CPU.

Some lower-level BybyLang operations retain CPU reference/fallback behavior. That does not make CPU a supported `Backend` choice for model execution.

## Where to go next

- Use [Backend API](backend-api.md) for GPU kernels.
- Use [Base Model API](base-model-api.md) for training-oriented transformer layers.
- Use [LLaMA Inference API](llama-inference-api.md) for Hugging Face-style decoder inference.
- Use [BybyLang GPU API and DSL](bybylang-api.md) to write GPU programs or DSL demos.
