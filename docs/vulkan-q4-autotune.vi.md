
# Bộ tự tinh chỉnh Q4 Vulkan

[English](vulkan-q4-autotune.md) | [Tiếng Việt](vulkan-q4-autotune.vi.md)

Bộ autotune Q4 động nằm trong
`vendor/bybylang/backends/vulkan/q4_autotune.nim`, được gọi qua resident
Q4/session matmul. Nó đối chiếu candidate với tham chiếu, đo thời gian và
cache phương án được chọn theo geometry ma trận và cấu hình lượng tử hóa.


## API công khai cấp thấp

```nim
proc vulkanPrepareQ4Tuned*(w, scales, zeros: VulkanBufferHandle,
                           K, N, group, groups, mode: int): bool
proc vulkanQ4TuningStats*(): tuple[preparedPlans, wideDispatches: int]
proc vulkanSessionMatmulQ4Enc*(a: VulkanBufferHandle,
    wq, scales, zeros: VulkanBufferHandle, c: var VulkanBufferHandle,
    M, K, N, groupSize, nGroupsPerRow: int): bool
```

Các API dùng buffer handle Vulkan nội bộ. Ứng dụng thường nên dùng
`beResidentUploadQ4`, `residentPrepareQ4`, `sessionMatmulQ4` trong
[API backend](backend-api.vi.md) và [API session GPU](gpu-session-api.vi.md).


## Biến môi trường

- `BYBY_Q4_TUNED=0`: tắt chọn phương án tuned, dùng nhánh tham chiếu.
- `BYBY_Q4_TUNED=wide`: ưu tiên candidate wide-dispatch đã kiểm tra khi có.
- `BYBY_GPU_AUTOTUNE=0`: bỏ giai đoạn chuẩn bị/tune.
- Cần `glslc` để compile candidate SPIR-V. Candidate không hỗ trợ hoặc không
  hợp lệ sẽ bị bỏ qua.

Tuner loại candidate sai kết quả hoặc vượt giới hạn thread/shared memory.
Kernel thắng và thời gian phụ thuộc GPU/driver; source và log runtime là
căn cứ cho lần chạy cụ thể.
