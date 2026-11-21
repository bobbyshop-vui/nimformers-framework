# Nim package facade and dynamic C ABI bindings.
version       = "2.0.0"
author        = "Nimformers contributors"
description   = "Nim inference, tensor kernels, and BybyLang runtime"
license       = "MIT"
srcDir        = "."
requires "nim >= 2.0.0"

# Keep the package allowlisted: tests, model weights, checkpoints, and build
# caches are excluded. The target-built C ABI library and worker are added by
# the before-install hook below.
installFiles = @[
  "LICENSE", "nimformers.nim", "nimformers_shared.nim",
  "bindings/nimformers_shared.nim",
  "docs/forward-hooks.md", "docs/forward-hooks.vi.md",
  "nimformers_lib.nim", "include/nimformers.h",
  "examples/nim/native_process_group.nim", "examples/nim/shared_library.nim",
  "examples/nim/custom_forward_hooks.nim", "examples/bybylang/shared_library.bybylang",
  "examples/nim/plugins/token_hook_plugin.nim",
  "nimformer.nim", "backend.nim", "customfloat.nim", "quant.nim",
  "tensor_ops.nim", "trace.nim", "mapped_weights.nim", "fp4_gemv_gpu.nim",
  "fp4_gemv_vm.nim", "mamba2_vm_kernels.nim", "mamba2_full_vm.nim",
  "prefill_profile.nim",
  "session_vm.nim", "session_vm_dispatch.nim", "jit_vm.nim",
  "parallel/vm.nim", "parallel/formers_batch.nim", "parallel/worker.nim",
  "mamba2_ssm_gpu.nim",
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
]
installDirs = @[
  "vendor/bybylang/backends", "vendor/bybylang/micro_kernel"
]

const nimFlags = "-d:release --hints:off --mm:arc --opt:speed --path:. " &
  "--path:test/codestral --path:test/deepseek-inference --path:kernelfortest"
const sharedLibrary = when defined(windows): "nimformers.dll"
  elif defined(macosx): "libnimformers.dylib"
  else: "libnimformers.so"
const workerExecutable = when defined(windows): "nimformers-worker.exe"
  else: "nimformers-worker"
const sharedLibraryPath = "lib/" & sharedLibrary
const workerExecutablePath = "lib/" & workerExecutable

installFiles.add(sharedLibraryPath)
installFiles.add(workerExecutablePath)

task buildShared, "Build the Nimformers shared library and worker executable":
  when defined(windows):
    exec "if not exist lib mkdir lib"
  else:
    exec "mkdir -p lib"
  exec "nim c " & nimFlags & " --app:lib --nimcache:nimcache/c-api -o:" &
    sharedLibraryPath & " nimformers_lib.nim"
  exec "nim c " & nimFlags & " --nimcache:nimcache/shared-worker -o:" &
    workerExecutablePath & " parallel/worker.nim"

before build:
  buildSharedTask()

before install:
  buildSharedTask()
