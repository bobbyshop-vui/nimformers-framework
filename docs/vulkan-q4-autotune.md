
# Vulkan Q4 Autotuner

[English](vulkan-q4-autotune.md) | [Tiếng Việt](vulkan-q4-autotune.vi.md)

The dynamic Q4 Vulkan autotuner is implemented in `vendor/bybylang/backends/vulkan/q4_autotune.nim` and is reached through resident Q4/session matmul. It validates candidate kernels against a reference result, times candidates, and caches the selected plan by matrix geometry and quantization settings.


## Public low-level API

```nim
proc vulkanPrepareQ4Tuned*(w, scales, zeros: VulkanBufferHandle,
                           K, N, group, groups, mode: int): bool
proc vulkanQ4TuningStats*(): tuple[preparedPlans, wideDispatches: int]
proc vulkanSessionMatmulQ4Enc*(a: VulkanBufferHandle,
    wq, scales, zeros: VulkanBufferHandle, c: var VulkanBufferHandle,
    M, K, N, groupSize, nGroupsPerRow: int): bool
```

These APIs use internal Vulkan buffer handles; application code should normally use `beResidentUploadQ4`, `residentPrepareQ4`, and `sessionMatmulQ4` documented in [Backend API](backend-api.md) and [GPU Session API](gpu-session-api.md).


## Environment controls

- `BYBY_Q4_TUNED=0`: disable tuned plan selection and use the reference path.
- `BYBY_Q4_TUNED=wide`: prefer a validated wide-dispatch candidate when available.
- `BYBY_GPU_AUTOTUNE=0`: skip the prepare/tune phase.
- `glslc` must be available for candidate SPIR-V compilation. Unsupported or invalid candidates are skipped.

The tuner rejects candidates that fail validation or exceed device thread/shared-memory limits. Exact winning kernel and timing depend on the GPU and driver; the source and runtime log are authoritative for a particular run.
