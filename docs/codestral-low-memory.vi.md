
# Codestral: mmap, máy 16 GiB và Vulkan Tensor Cores

[English](codestral-low-memory.md) | [Tiếng Việt](codestral-low-memory.vi.md)

Inference Codestral mặc định `BYBY_MODEL_MMAP=1`. Payload NIMQ2 nằm trong
mapping file chỉ đọc. LoRA A/B của NIMBIN1/NIMBIN4 và các ma trận raw lớn
`newBlock::* .weight` cũng được map. Metadata/state nhỏ được sao chép.
View giữ quyền sở hữu mapping kể cả sau khi reader hoặc bảng tạm bị hủy.
Không cấp buffer `readFile()` lớn bằng model.

FP4-KM vẫn giữ dạng đóng gói trên GPU. LoRA chạy
`W*x + scale*B*(A*x)` trong session GPU. Vulkan upload từ mapped view qua
staging allocation từng tensor tới VRAM device-local trên GPU rời; Metal
dùng shared buffer. Upload có thể sao chép vào bộ nhớ GPU: mmap không biến
RAM hệ thống và VRAM riêng của T4 thành cùng allocation. Sau upload, tham
chiếu payload host dư được xóa. Sidecar raw dùng mapped view FP32 đến lúc upload.


## Chạy

```sh
make infer-lora
```

`make infer-lora` mặc định `auto`. Chọn Vulkan tường minh:

```sh
make infer-lora BYBY_INFER_BACKEND=vulkan
```

Cần driver Vulkan và `glslc` trong `PATH`. Thiết bị được phân loại theo
`deviceType` vật lý của Vulkan: chỉ `DISCRETE_GPU` cho trọng số resident
vào VRAM device-local/private; iGPU/UMA dùng buffer host-visible
coherent/shared. `BYBY_VULKAN_DEVICE_LOCAL=1` cũng không ép iGPU dùng
private weights được. Mặc định là `auto`; `0` tắt device-local trên dGPU.
Log startup ghi `type=iGPU residentWeights=host-visible/shared` hoặc
`type=dGPU residentWeights=device-local/VRAM`. Checkout hiện dùng backend
trong `vendor/bybylang/`; Makefile gốc không có target `prepare-backend`.
Xem [hướng dẫn lệnh Make](make-commands.vi.md) cho lệnh build/run được hỗ trợ.

`BYBY_GEMV_TG=max` là mặc định FP4: yêu cầu group lớn nhất phù hợp geometry
ma trận, giới hạn thread thiết bị và shared memory. Kết quả được đối chiếu
tham chiếu trước khi dùng; candidate compile/validation thất bại sẽ giữ
nhánh tham chiếu. Lỗi thiết bị GPU không phục hồi được sẽ được báo.
Kernel reduction không âm thầm dispatch ở width nhỏ hơn width đã compile.

- Candidate FP4: 32, 64, 128, 256, 512, 1024 lane.
- Mỗi lane xử lý tám cột: K=4096 cho tối đa 512 lane; K=8192 cho tối đa
  1024 khi giới hạn GPU/shared memory cho phép.
- Scratch batch tám là `(272 + 8*tg)*4` bytes; giới hạn 32 KiB loại width 1024.
- `BYBY_GEMV_TG=auto` kiểm tra và đo mọi width hợp lệ để chọn nhanh nhất;
  width lớn nhất không nhất thiết nhanh nhất.
- Giá trị số của `BYBY_GEMV_TG` yêu cầu đúng width hợp lệ đó; nếu không dùng
  được thì fallback tham chiếu. `BYBY_GPU_AUTOTUNE=0` giữ kernel tham chiếu.
- Batch FP4 Intel/MoltenVK bị giới hạn 512 sau khi tái hiện timeout ở
  1024; giới hạn này không áp lên NVIDIA/Linux hay Metal native.
- Vulkan Q4 tuning cũng xét width đến 1024 trong giới hạn thiết bị.


## Merge trong ngân sách bộ nhớ

Giữ `BYBY_MERGE_LORA=0` khi inference để bảo toàn delta adapter và tránh
mở rộng model sang FP32. FP4-KM nhìn chung không biểu diễn chính xác mọi
cập nhật LoRA FP32.

Nếu chọn `BYBY_MERGE_LORA=1`, merge vẫn xử lý từng tensor và kiểm tra sai số
lượng tử tương đối với delta LoRA. `BYBY_MERGE_SCRATCH_MIB=256` giới hạn
ước lượng bảo thủ của mảng dense tạm. Tensor quá lớn giữ base và adapter
gộp trong kernel. Bỏ qua requantize low-bit trừ khi
`BYBY_MERGE_LORA_TRY_FP4=1`; kiểm tra bộ nhớ và sai số vẫn áp dụng. Đây là
merge có bảo vệ trong RAM, không phải xuất stream model đã merge vĩnh viễn.

Giới hạn inference 16 GiB còn phụ thuộc độ dài prompt, batch, state model,
allocation GPU và tiến trình khác. Đo loader chưa chứng minh được peak
inference đầy đủ. RAM host và 16 GB VRAM của T4 là hai ngân sách riêng.


## NVIDIA T4 và Tensor Cores

Nhánh Vulkan dò `VK_KHR_cooperative_matrix`, shader FP16, Vulkan memory
model và tile FP16 input/FP32 accumulator được thiết bị công bố. Với
driver hỗ trợ, FP4-KM được giải mã thành tile FP16 nhỏ trong shared memory
rồi nhân bằng cooperative matrix. Trọng số vẫn đóng gói; không mở rộng
cả model sang FP16. Batch FP4 và resident prefill đều đi qua nhánh này.
GEMV từng token giữ kernel scalar tối ưu theo băng thông.

T4 có Tensor Cores cho mixed precision FP16/FP32 và số nguyên; FP4-KM ở
đây là định dạng lưu, không phải lệnh nhân FP4 native trên T4. Xem
[datasheet NVIDIA T4](https://www.nvidia.com/content/dam/en-zz/Solutions/Data-Center/tesla-t4/t4-tensor-core-datasheet.pdf)
và [đặc tả KHR cooperative matrix](https://docs.vulkan.org/refpages/latest/refpages/source/VK_KHR_cooperative_matrix.html).

`BYBY_VULKAN_TENSOR_CORES=1` bật mặc định. Log ghi
`matrix=KHR_cooperative_matrix FP16/FP32` khi có hỗ trợ, nếu không là
`matrix=scalar`. Driver cũ chỉ có NV cần cập nhật cho nhánh KHR. Thiết bị
không hỗ trợ hoặc shader compile lỗi sẽ giữ Vulkan compute thông thường.


## Kiểm tra

Với API mapped, đọc payload lượng tử qua `qt.bytes` và raw float qua
`tensor.floatValues`; `.data` sở hữu bộ nhớ của chúng rỗng khi đang map.
`clearPayload()` giải phóng cả payload owned và mapped.
`loadQuantStateDict(path)` kiểu cũ vẫn tạo owned sequence; Codestral chọn
`mapped=true` trừ khi tắt mmap.

Regression CPU kiểm tra tương đương mapped/owned, vòng đời mapping,
serialization, payload hỏng/cụt, gắn adapter, từ chối merge vượt ngân sách
và độ đúng FP4 GEMV/GEMM ở mọi width 32–1024.
`test/codestral/test_mmap_gpu.nim` so FP4+LoRA mapped, sidecar dense mapped
và batch output với tham chiếu CPU; đặt `BYBY_BENCH_BACKEND=vulkan` trên máy T4.

`test/codestral/bench_mmap_load.nim` đo load mà không khởi tạo GPU hay giải
lượng tử trọng số. Với các file trong workspace (579 tensor base, 256
tensor adapter, 12 ma trận raw sidecar), nó map 5711 MiB payload, sao chép
dưới 1 MiB raw state và đo maximum RSS 4,3 MiB khi cache ấm. Đây chỉ là kết
quả lazy loading.

Sau khi cài tokenizer thật, một lần chạy Metal đầy đủ của checkpoint này
(64 block, prefill prompt và sinh một token) đã hoàn tất trên Intel Iris
640. `/usr/bin/time -l` ghi maximum RSS 4.460.523.520 bytes (4,15 GiB), peak
memory footprint 7.051.587.584 bytes (6,57 GiB) và không swap. Đây là một
trường hợp prompt ngắn có chuẩn bị GPU ban đầu; prompt dài, batch và VRAM
T4 cần được đo trên cấu hình đích.

Nếu thiếu tokenizer, chạy `make download-codestral-tokenizer` để không tải
lại trọng số. Inference Codestral load `tokenizer.model.v3` hoặc
`tokenizer.model` bằng Nim thuần, dựng vocabulary trie một lần, áp dụng
normalization nhúng và giữ hành vi whitespace/byte fallback. `make
build-infer-lora` không cần gói tokenizer Python, Nimpy, C++ shim hay thư
viện SentencePiece. Loader kiểm tra tokenizer trước khi dựng/giải lượng
tử model.
