
# BybyLang integration tests

[English](README.md) | [Tiếng Việt](README.vi.md)

These tests use the BybyLang DSL/compiler in
[`vendor/bybylang`](../../vendor/bybylang) to exercise Nimformers.
For a line-by-line explanation of the DSL and C ABI calls, see the
[BybyLang test API reference](../../docs/bybylang-tests.md).

- `syntax_test.bybylang` — language features: functions with parameters and
  return types, type/const sections, `case`/`try`, field/index/compound
  assignments and blank lines inside blocks.
- `capi_test.bybylang` — `import nim "../../lib_c_api"` and calls the C ABI
  from BybyLang: context/backend, elementwise add, matmul, softmax,
  layernorm, int8 quantization, micro-VM dispatch, and fused attention.
- `native_gpu_test.bybylang` — native `gpu ...` DSL commands backed by the
  same `vendor/bybylang/gpubackend.nim` runtime, plus a native
  `micro kernel ...:` block that lowers emitted bytecode through the
  Nimformers micro-VM.
- `cpu_test.bybylang` — the same DSL and a micro-kernel on the SIMD CPU
  backend (`gpu backend is "cpu"`), including a CPU `session:` block; no GPU
  required.
- `nim.cfg` — extra Nim search paths needed when the generated executable
  imports `lib_c_api` (root modules plus the model-specific paths it uses).
- `trace_test.bybylang`, `diag_test.bybylang` and `cpu_kernel_regression.nim`
  cover core tracing, source diagnostics and CPU kernel results.
- `run.sh` — AOT-compiles and runs the integration programs and checks their
  diagnostic output.

Run:

```bash
./test/bybylang_test/run.sh
# or
make bybylang-test
```

The runner sets `BYBYLANG_NIMCACHE` to `nimcache/bybylang-test`, so the
generated Nim files and binaries can be cleaned with the normal `nimcache`
cleanup. Generated `*_bin` / `*_bin.nim` files are git-ignored.

The `auto` cases use Metal or Vulkan when available. `cpu_test.bybylang`
always selects the CPU VM and also runs on machines without a GPU.

BybyLang parses every non-command line containing `=` as an assignment
(`left = right`).  Avoid `=` inside trailing comments on raw Nim lines; put
such comments on their own line instead.
