# Nim package facade and dynamic C ABI bindings.
version       = "2.0.0"
author        = "Nimformers contributors"
description   = "Nim inference, tensor kernels, and BybyLang runtime"
license       = "MIT"
srcDir        = "."
requires "nim >= 2.0.0"

# Keep the package allowlisted: tests, model weights, checkpoints, generated
# binaries, and repository build output are deliberately not shipped.
installFiles = @[
  "LICENSE", "nimformers.nim", "nimformers_shared.nim", "nimformers_worker.nim",
  "lib_c_api.nim", "include/nimformers.h",
  "examples/nim/native_process_group.nim", "examples/nim/shared_library.nim",
  "examples/bybylang/shared_library.bybylang",
  "nimformer.nim", "backend.nim", "customfloat.nim", "quant.nim",
  "tensor_ops.nim", "trace.nim", "mapped_weights.nim", "fp4_gemv_gpu.nim",
  "fp4_gemv_vm.nim", "mamba2_vm_kernels.nim", "mamba2_full_vm.nim",
  "prefill_profile.nim",
  "session_vm.nim", "session_vm_dispatch.nim", "jit_vm.nim",
  "parallel_vm.nim", "formers_parallel.nim", "mamba2_ssm_gpu.nim",
  "kernelfortest/fp4_gemv_gpu.nim", "kernelfortest/fp4_gemv_vm.nim",
  "test/codestral/prefill_profile.nim", "test/codestral/mamba2_vm_kernels.nim",
  "test/codestral/mamba2_full_vm.nim", "test/codestral/sentencepiece_native.nim",
  "test/deepseek-inference/nim_inference.nim",
  "test/deepseek-inference/native_hf_tokenizer.nim",
  "test/deepseek-inference/tokenizer_chat_template.nim",
  "test/deepseek-inference/tokenizer_regex.nim",
  "test/deepseek-inference/tokenizer_unicode.nim",
  "vendor/bybylang/bybylang.nim", "vendor/bybylang/diag.nim",
  "vendor/bybylang/distributed.nim", "vendor/bybylang/gpubackend.nim",
  "vendor/bybylang/mapped_file.nim", "vendor/bybylang/micro_kernel.nim",
  "vendor/bybylang/probe.nim", "vendor/bybylang/probe_data.nim",
  "vendor/bybylang/probe_weights.nim",
  "vendor/bybylang/backends", "vendor/bybylang/micro_kernel"
]

const nimFlags = "-d:release --hints:off --mm:arc --opt:speed --path:. " &
  "--path:test/codestral --path:test/deepseek-inference --path:kernelfortest"
const sharedLibrary = when defined(windows): "nimformers.dll"
  elif defined(macosx): "libnimformers.dylib"
  else: "libnimformers.so"
const workerExecutable = when defined(windows): "nimformers-worker.exe"
  else: "nimformers-worker"

task buildShared, "Build the Nimformers shared library and worker executable":
  exec "nim check " & nimFlags & " --nimcache:nimcache/c-api-check lib_c_api.nim"
  exec "nim c " & nimFlags & " --app:lib --nimcache:nimcache/c-api -o:" &
    sharedLibrary & " lib_c_api.nim"
  exec "nim c " & nimFlags & " --nimcache:nimcache/shared-worker -o:" &
    workerExecutable & " nimformers_worker.nim"
