# Codestral: mmap, 16 GiB machines and Vulkan Tensor Cores

Codestral inference defaults to `BYBY_MODEL_MMAP=1`. NIMQ2 payloads stay in a
read-only file mapping. NIMBIN1/NIMBIN4 LoRA A/B and large raw `newBlock::* .weight`
matrices also stay mapped. Small metadata/state arrays are copied. Views retain
ownership of the mapping, including after a reader or temporary table goes away.
No model-sized `readFile()` buffer is allocated.

Packed FP4-KM stays packed on GPU. LoRA runs as `W*x + scale*B*(A*x)` in the GPU
session. Vulkan uploads directly from mapped views through a per-tensor staging
allocation to device-local VRAM on discrete GPUs; Metal uses shared buffers.
These uploads can copy into GPU memory: mmap does not make system RAM and a
T4's separate VRAM the same allocation. After upload, redundant host payload
references are cleared. Raw sidecars use mapped FP32 views until upload.

## Run

```sh
make infer-lora
```

`make infer-lora` defaults to `auto`. To request Vulkan explicitly:

```sh
make infer-lora BYBY_INFER_BACKEND=vulkan
```

Use a Vulkan driver and `glslc` on PATH for that override.
The selected device is classified using Vulkan's physical `deviceType`:
only `DISCRETE_GPU` permits resident uploads to device-local/private VRAM;
iGPU/UMA uses host-visible coherent/shared buffers. Even
`BYBY_VULKAN_DEVICE_LOCAL=1` cannot force private weights on an iGPU.
The default is `auto`; `0` can disable device-local uploads on a dGPU.
The startup log prints `type=iGPU residentWeights=host-visible/shared` or
`type=dGPU residentWeights=device-local/VRAM`. The Makefile prepares the companion
BybyLang source patch automatically. That patch lives in
`patches/bybylang-mmap-workgroups.patch`, because this workspace's populated
submodule directory has no Git metadata and Git cannot display its file diffs.
`make prepare-backend` is idempotent and stops on conflicting backend changes.

`BYBY_GEMV_TG=max` is the FP4 default. It requests the largest supported group
that fits the matrix geometry, device thread limit, and shared memory. The
result is checked against the reference before use; failed candidates retain
the reference path after compilation/validation rejection. Unrecoverable GPU
device errors are reported. A reduction kernel is never silently dispatched at a
smaller width than it was compiled for.

- FP4 candidates: 32, 64, 128, 256, 512, 1024 lanes.
- Each lane processes eight input columns: K=4096 permits up to 512 lanes;
  K=8192 permits up to 1024 when the GPU/shared-memory limits also permit it.
- Batch-eight scratch is `(272 + 8*tg)*4` bytes; a 32 KiB limit excludes 1024.
- `BYBY_GEMV_TG=auto` validates and times all legal widths, choosing the fastest.
  The largest width is not necessarily the fastest.
- A numeric `BYBY_GEMV_TG` requests that exact legal width, with reference
  fallback when unavailable. `BYBY_GPU_AUTOTUNE=0` retains the reference kernel.
- Intel/MoltenVK batch FP4 is capped at 512 after a reproduced timeout at
  1024; this does not cap NVIDIA/Linux or native Metal.
- Vulkan Q4 tuning also considers widths through 1024 within device limits.

## Merge within the memory budget

Keep `BYBY_MERGE_LORA=0` for inference. It preserves the original adapter delta
and avoids expanding the model to FP32. FP4-KM cannot generally represent an
arbitrary FP32 LoRA update exactly.

If `BYBY_MERGE_LORA=1` is requested, merging still handles one tensor at a time
and requires a quantization-error check relative to the LoRA delta.
`BYBY_MERGE_SCRATCH_MIB=256` caps the conservative estimate for temporary dense
arrays. An oversized tensor retains its original base plus fused adapter.
Low-bit requantization is skipped unless `BYBY_MERGE_LORA_TRY_FP4=1`; the memory
and numerical guards apply even then. This is an in-memory guarded merge, not a
streamed export of a permanently merged model.

A 16 GiB inference limit also depends on prompt length, batch size, model state,
GPU allocations and other running processes. Loader measurements alone do not
establish the full inference peak. Host RAM and the T4's 16 GB VRAM are separate
budgets.

## NVIDIA T4 / Tensor Cores

The Vulkan path queries `VK_KHR_cooperative_matrix`, shader FP16, Vulkan memory
model support and the advertised FP16-input / FP32-accumulator tile. With a
supporting driver, FP4-KM is decoded into small FP16 shared-memory tiles and
multiplied using cooperative matrix operations. Weights remain packed; there
is no full-model FP16 expansion. Batch FP4 entry points and resident prefill
both reach this path. Token-at-a-time GEMV keeps the scalar bandwidth-oriented
kernel.

The T4 has Tensor Cores for FP16/FP32 mixed precision and integer types; FP4-KM
is a storage format here, not a native T4 FP4 multiply instruction. See the
[NVIDIA T4 datasheet](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf)
and [KHR cooperative matrix specification](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_cooperative_matrix.html).

`BYBY_VULKAN_TENSOR_CORES=1` is enabled by default. The device log reports
`matrix=KHR_cooperative_matrix FP16/FP32` when available, otherwise `matrix=scalar`.
Older NV-only drivers need a driver update for this KHR path. Unsupported devices
and shader compiler failures retain ordinary Vulkan compute.

## Verification

For code using the mapped APIs, read quant payloads through `qt.bytes` and raw
float payloads through `tensor.floatValues`; their owned `.data` sequences are
empty while mapped. `clearPayload()` releases both owned and mapped quant
storage. Legacy `loadQuantStateDict(path)` keeps owned sequences; Codestral
explicitly selects `mapped=true` unless mmap is disabled.

The CPU regressions cover mapped/owned equivalence, mapping lifetime,
serialization, corrupt/truncated payloads, adapter attachment, merge budget
rejection, and FP4 GEMV/GEMM correctness for every width from 32 through 1024.
`test/codestral/test_mmap_gpu.nim` compares mapped FP4 + LoRA, mapped dense
sidecars and batch outputs with CPU references; set
`BYBY_BENCH_BACKEND=vulkan` to run it on a T4 host.

`test/codestral/bench_mmap_load.nim` measures loading without initializing GPU or
dequantizing weights. On the files in this workspace (579 base tensors, 256
adapter tensors, 12 raw sidecar matrices), it mapped 5711 MiB of payload, copied
less than 1 MiB of raw state, and measured 4.3 MiB maximum RSS with a warm cache.
This result describes lazy loading only.

After installing the real tokenizer, a complete Metal run of this checkpoint
(64 blocks, prompt prefill plus one generated token) finished successfully on
the Intel Iris 640 host. `/usr/bin/time -l` reported maximum RSS 4,460,523,520 bytes
(4.15 GiB), peak memory footprint 7,051,587,584 bytes (6.57 GiB), and zero swaps.
This is one short-prompt case, including initial GPU preparation; longer prompts,
batches and T4 VRAM usage still need measurement on the target configuration.

If tokenizer assets are missing, run `make download-codestral-tokenizer` without
redownloading weights. Nimpy reports its embedded Python version on package
import failure. Dependencies can be isolated in `.runtime/codestral-tokenizer/X.Y`
using the matching `pythonX.Y` and
`test/codestral/tokenizer-requirements.txt`. The loader checks the tokenizer
before constructing or dequantizing the model.
