# Flag runtime (Vulkan, native và FIR graph)

[English](runtime-flags.md) | [Tiếng Việt](runtime-flags.vi.md)

Biến môi trường mà `nim_inference`, `formers-cli` (route native và route `--fir`),
driver Vulkan trong `vendor/bybylang` và `firlift` đọc. Mọi tối ưu notebook Colab T4
(`notebooks/deepseek_colab_vulkan.ipynb`) bật đều đã là mặc định, nên chạy thường
không cần flag nào. Chỉ đặt flag để tắt một tối ưu, ép một lựa chọn hoặc lấy chẩn đoán.

## Giá trị notebook và mặc định

| Flag | Giá trị notebook | Mặc định | Ở đâu |
| --- | --- | --- | --- |
| `BYBY_MODEL_MMAP` | `1` | `1` | weight mmap từ `model.nimq` |
| `BYBY_VULKAN_DEVICE_LOCAL` | `auto` | `auto` | weight nằm ở VRAM trên dGPU |
| `BYBY_VULKAN_TENSOR_CORES` | `1` | `1` | GEMM prefill bằng cooperative matrix |
| `BYBY_UBATCH` | `512` | `512` | số hàng micro-batch prefill (native và FIR) |
| `BYBY_PREFILL_CB_MS` | `300` (native), `450` (lift) | `300` | thời gian đích một command buffer prefill native; FIR graph không đọc |
| `BYBY_VULKAN_Q4_SPLIT_QKV` | `0` / `1` (A/B) | `auto` | tự đo QKV gộp hay tách |
| `BYBY_FIR_Q4_BACKEND` | `micro` | `micro` | node Q4 của FIR chạy micro-kernel đã lift |
| `FIRLIFT_Q4_GEMM` | `tensor_core` | `auto` | `auto` lift GEMM Tensor Core khi trace ghi nhận hỗ trợ Vulkan cooperative matrix hoặc Metal simdgroup matrix |
| `FIRLIFT_Q4_GEMV` | `word_parallel` / `auto` | `auto` | `auto` lift GEMV subgroup trên Vulkan (nhanh hơn `word_parallel`) |
| `BYBY_FIR_PROBE_STEPS` | `prefill:2,decode:1` | mọi bước | các bước recorder giữ (chỉ khi lift) |
| `BYBY_FIR_PROBE_DATA` | `0` | `1` dưới `firlift --run` | pipe dữ liệu upload (chỉ khi lift) |
| `BYBY_FIR_PROFILE` | `1` | `0` | chỉ để chẩn đoán |

Chồng encode với GPU không cần flag: bước decode FIR được ghi một lần rồi phát lại
(`BYBY_FIR_DECODE_REPLAY`), và bước nào phải encode thì được submit theo đoạn, không
chờ (`BYBY_FIR_DECODE_CB_NODES`, `BYBY_VULKAN_COMMIT_WAIT`).

## Runtime FIR graph (`formers-cli --fir`)

| Flag | Mặc định | Tác dụng |
| --- | --- | --- |
| `BYBY_FIR_DECODE_REPLAY` | `1` | Vulkan và Metal: ở bước đầu ghi bước decode một lần (Vulkan: một command buffer dùng lại được; Metal: các dispatch của bước, xem `BYBY_METAL_REPLAY_ICB`); các bước sau chỉ ghi token/position vào buffer host-visible, phát lại, rồi đọc logits từ buffer đã map (không encode trên host). Hàng embedding được gather trên GPU. Các node session chính xác được hỗ trợ (RMSNorm, vector/activation, FP4 Metal GEMM và NT GEMM) cũng phát lại thao tác session native. `0`: encode mỗi bước. Tự tắt khi có probe, `BYBY_FIR_PROFILE=2` hoặc `BYBY_FIR_Q4_BACKEND=native`, và khi bước upload, cấp phát hoặc giải phóng buffer trong lúc ghi. |
| `BYBY_METAL_REPLAY_ICB` | `1` | Metal phát lại session đã ghi (`BYBY_FIR_DECODE_REPLAY`): `1` dựng một indirect command buffer một lần rồi chạy bằng `executeCommandsInBuffer` khi GPU hỗ trợ; `0` phát lại các dispatch đã ghi trên encoder thường trong một vòng lặp (cả hai đều không duyệt graph). |
| `BYBY_METAL_REPLAY_LOG` | không đặt | Đặt: log mỗi lần Metal ghi session (số lệnh, pipeline, buffer) và lý do không giữ được bản ghi. |
| `BYBY_FIR_DECODE_REPLAY_VERBOSE` | `0` | `1` in `[fir] decode replay on/off: ...` (cũng in khi có `BYBY_FIR_PROFILE`). |
| `BYBY_FIR_DECODE_CB_NODES` | `64` | Bước decode phải encode (replay tắt): submit không chờ sau mỗi chừng này node (lần đầu sau 16) để GPU chạy trong lúc host encode. `0`: một command buffer. |
| `BYBY_FIR_Q4_BACKEND` | `micro` | `micro`: micro-kernel Q4 đã lift. `native`: công tắc chẩn đoán qua `sessionMatmulQ4` (không dùng được với node Tensor Core). |
| `BYBY_FIR_SESSION_DIRECT` | `1` | `0` chuyển các lần phát lại session chính xác qua buffer scratch; `1` phát lại trực tiếp các thao tác session được hỗ trợ trên buffer graph. |
| `BYBY_FIR_WARM_RELOAD` | `0` | `1` nạp lại weight mô hình cho lần chạy warm thay vì reset trạng thái graph đã lift. |
| `BYBY_METAL_SIMDGROUP_MATRIX` | `1` | `0` ẩn hỗ trợ simdgroup matrix của MSL, nên trace Metal ghi `device="plain"` và không dựng micro-kernel Tensor Core nào. |
| `BYBY_FIR_Q4_GEMV_TUNE` | `1` | Vulkan: lúc load, đo mọi số hàng mỗi subgroup (1, 2, 4) và độ rộng workgroup (64, 128, 256) cho từng shape GEMV trên weight thật, giữ tổ hợp nhanh nhất. `0`: giữ lựa chọn lúc lift. |
| `BYBY_FIR_Q4_GEMV_ROWS` | theo lift | Ép số hàng output mỗi subgroup của GEMV (1, 2 hoặc 4). |
| `BYBY_FIR_Q4_GEMV_THREADS` | theo lift | Ép độ rộng workgroup GEMV (64, 128 hoặc 256). Ép cả hai: không tune. |
| `BYBY_FIR_Q4_GEMV_REDUCE` | `auto` | `subgroup` (cần subgroup rộng 32), `shared`, hoặc `auto` theo thiết bị. |
| `BYBY_FIR_Q4_COOP_ROWS` | theo dispatch | Ép tile hàng Tensor Core (16, 32 hoặc 64). |
| `BYBY_FIR_Q4_COOP_WG_PER_UNIT` | `2` | Số workgroup mỗi SM/CU mà việc chọn tile Tensor Core nhắm tới. |
| `BYBY_FIR_Q4_COOP_VERBOSE` | `0` | `1` in dòng tile Tensor Core và `[fir] Q4 GEMV tune ...` (cần `--verbose`). |
| `BYBY_FIR_Q4_MAP_LOG` | `0` | `1` log việc map Q4 resident (backend Q4 native). |
| `BYBY_FIR_RELOC` | `1` | Graph arena trên Metal: một pipeline mỗi kernel. `0`: một pipeline mỗi node. |
| `BYBY_FIR_PROFILE` | `0` | `1`: trung bình encode trên host / chờ GPU / đọc logits mỗi bước decode. `2`: kết thúc session sau mỗi node và báo thời gian từng kernel (chậm; chỉ để chẩn đoán). |
| `BYBY_UBATCH` | `512` | Giới hạn số hàng một lần chạy phase prompt (trừ phase có logits mỗi hàng). `0`: không giới hạn. |
| `BYBY_PREFILL_MIN_ROWS` | `5` | Codestral (native và FIR): prompt hoặc phần đuôi prompt có từ chừng này token trở lên chạy một lần prefill nhiều hàng; ít hơn thì chạy từng token. `1`: gộp mọi phần đuôi từ 2 token; `128` trở lên: chỉ các đoạn 128 hàng (hành vi cũ). Native và FIR phải dùng cùng giá trị, và FIR graph phải được lift với giá trị đó (graph có một phase cho mỗi cỡ đuôi). |

## Inference native (`nim_inference`, route native của `formers-cli`)

| Flag | Mặc định | Tác dụng |
| --- | --- | --- |
| `BYBY_MODEL_MMAP` | `1` | mmap `model.nimq` thay vì đọc cả file vào RAM. |
| `BYBY_PREFAULT` | `1` | Đọc tuần tự một lượt file đã map để nạp page cache. |
| `BYBY_EMBED_RAM` | `1` | Giữ embedding không tied trong RAM. |
| `BYBY_LM_HEAD` | `q4` | Lượng tử hoá `lm_head` fp32 sang Q4 để chạy GEMV đã tune. `fp32`: giữ nguyên. |
| `BYBY_UBATCH` | `512` | Số hàng micro-batch prefill. `0`: không giới hạn trên thiết bị có pacing. |
| `BYBY_PREFILL_CB_MS` | `300` | Thời gian GPU đích của một command buffer prefill. |
| `BYBY_PREFILL_MAX_ROWS` | `512` | Số hàng tối đa của một đoạn prefill. |
| `BYBY_PREFILL_SESSION_MB` | `1024` | Ngân sách scratch của một session prefill. |
| `BYBY_PREFILL_LAYERS_PER_CB` | `4` | Số layer mỗi command buffer prefill trên thiết bị không pacing. |
| `BYBY_FLASH_PREFILL` | `1` | Prefill bằng flash attention. |
| `BYBY_DECODE_CB_MS` | `1500` | Thời gian GPU đích của một command buffer decode. |
| `BYBY_DECODE_LAYERS_PER_CB` | tự động | Số layer mỗi command buffer decode (`0`: một buffer mỗi token). |
| `BYBY_DECODE_FIRST_CB` | `0` | Commit sau chừng này layer đầu để GPU bắt đầu sớm. |
| `BYBY_MERGE_QKV_ATTN_SESSION` | `1` | Gộp QKV và attention vào một session. |
| `BYBY_MERGE_ALL_LAYERS_SESSION` | `1` | Giữ mọi layer của một token trong một session. |
| `BYBY_HEAD_IN_SESSION` | `1` | Chạy norm cuối và `lm_head` trong session decode. |
| `BYBY_VULKAN_Q4_SPLIT_QKV` | `auto` | Đo QKV gộp và tách một lần cho mỗi thiết bị/shape; `0` gộp, `1` tách. |
| `BYBY_PREFILL_HEAD_COPIES` | `0` | Giữ bản sao KV từng head trên Metal. |
| `BYBY_SPARSITY_PCT` | `0.0` | Thử nghiệm sparsity FFN (tắt). |
| `BYBY_GPU_MAX_YIELD_MS` | `8.0` | Trần của khoảng nhường thích ứng giữa các token. |
| `BYBY_PROGRESS` | `auto` | Dòng tok/s trực tiếp; `1` ép bật trên Colab, `0` tắt. |
| `BYBY_PREFILL_PROFILE` | `-1` | Layer có op prefill chạy mỗi op một command buffer (kèm `BYBY_SESSION_LOG`). |
| `BYBY_SESSION_LOG` | không đặt | Log thời gian session prefill. |
| `BYBY_DECODE_PROFILE` | `0` | Profile N token decode đầu. |
| `BYBY_DEBUG_PATH` | `0` | Log đường GPU/CPU mỗi op đi. |
| `BYBY_NO_VERIFY` | không đặt | Bỏ bước kiểm KV resident. |
| `BYBY_DISABLE_KV_RESIDENT` | `0` | `1` tắt KV resident (debug). |
| `BYBY_DISABLE_SESSION_RMSNORM` | không đặt | Dùng đường RMSNorm cũ. |
| `BYBY_TRY_ATTN_DECODE_GQA` | không đặt | Thử lại attention GQA decode gộp sau khi nó bị tắt. |
| `BYBY_LEGACY_DECODE_GC` | không đặt | GC định kỳ kiểu cũ trong decode. |
| `BYBY_GPU_SAMPLE`, `BYBY_GPU_SAMPLE_AHEAD`, `BYBY_GRAPH` | chỉ Metal | Pipe lấy mẫu trên GPU và phát lại graph decode trên Metal. |

## Driver Vulkan (`vendor/bybylang/backends/vulkan`)

| Flag | Mặc định | Tác dụng |
| --- | --- | --- |
| `BYBY_VULKAN_DEVICE` | điểm cao nhất | Chỉ số thiết bị; mặc định ưu tiên GPU rời (NVIDIA trước). |
| `BYBY_VULKAN_DEVICE_LOCAL` | `auto` | Weight resident ở VRAM trên dGPU (`auto` hoặc `1`); không bao giờ trên iGPU. Giá trị khác giữ bộ nhớ host-visible. |
| `BYBY_VULKAN_SCRATCH` | `auto` | Scratch ở VRAM: `auto` (dGPU), `device` (luôn), `host` (không). |
| `BYBY_VULKAN_TENSOR_CORES` | `1` | GEMM cooperative matrix. Trên dGPU mọi GEMM prefill phải dùng; `force` làm vậy trên mọi thiết bị; `tune` đo so với kernel scalar; `0` tắt. |
| `BYBY_VULKAN_SPLIT_K` | `auto` | Số lát split-K của GEMM Tensor Core dạng tile (`1`: tắt, `<n>`: cố định). |
| `BYBY_VULKAN_SPLIT_K_VERBOSE` | `0` | Log mọi lựa chọn split-K. |
| `BYBY_VULKAN_MMAP_IMPORT` | `1` | Import weight đã mmap làm bộ nhớ host trên iGPU. |
| `BYBY_VULKAN_POOL_MB` | `2048` | Trần pool buffer dùng lại. |
| `BYBY_VULKAN_DISPATCH_CACHE` | `1` | Cache pipeline JIT và binding giả; prewarm song song. |
| `BYBY_VULKAN_COMMIT_WAIT` | `0` | `1`: commit giữa session chờ GPU (kiểu cũ). Khi đang ghi recording thì commit không bao giờ chờ. |
| `BYBY_VULKAN_MAX_TG` | theo thiết bị | Trần độ rộng workgroup. |
| `BYBY_VULKAN_JIT_DIR` | thư mục tạm | Thư mục cache SPIR-V và pipeline. |
| `BYBY_VULKAN_JIT_VERBOSE` | `0` | Log biên dịch JIT và binding dispatch. |
| `BYBY_JIT_DUMP_SOURCE` | `0` | In GLSL của mọi kernel JIT. |
| `BYBY_VULKAN_TIMESTAMPS` | `0` | Timestamp thiết bị mỗi session. |
| `BYBY_VULKAN_STRICT` | `0` | Báo lỗi khi offset binding vượt buffer thay vì kẹp lại. |
| `BYBY_Q4_TUNED` | `1` | Autotune GEMV/GEMM Q4 (`0`: kernel mặc định). |
| `BYBY_GPU_AUTOTUNE` | `1` | Công tắc chung của autotune GPU. |
| `BYBY_Q4_SUBGROUP` | `1` | Cho phép kernel Q4 dùng subgroup. |
| `BYBY_Q4_TUNE_VERBOSE` | `0` | Log các kernel bị loại vì sai parity. |
| `BYBY_FLASH_ATTENTION` | `auto` | Loại flash attention; `scalar` ép FP32. |
| `BYBY_FUSE_ROPE` | `1` | Gộp RoPE vào bước append KV. |
| `BYBY_SSD` | `auto` | Kernel SSD của Mamba2. Trong lúc đang ghi FIR, `auto` giữ đường scan micro-VM bật vì shader SSD native không thể được graph đã lift phát lại; `1` ép dùng SSD và `0` tắt nó. |

## Lift (`firlift --run`)

| Flag | Mặc định | Tác dụng |
| --- | --- | --- |
| `FIRLIFT_Q4_GEMV` | `auto` | Kernel matmul Q4 một hàng. `auto` trên Vulkan: micro-kernel subgroup (thuật toán decode native), không thì word-parallel. Còn có `scalar`, `scalar_fma`, `parallel`, `parallel_fma`, `simd`, `word_parallel`, `subgroup`. |
| `FIRLIFT_Q4_GEMM` | `auto` | Kernel matmul Q4 nhiều hàng. `auto`: micro-kernel Tensor Core khi trace ghi nhận Vulkan `device=coopmatrix` hoặc Metal `device=simdmatrix`, trừ khi đặt `FIRLIFT_GENERIC`; nếu không thì theo lựa chọn kernel generic. Còn có `scalar`, `scalar_fma`, `parallel`, `parallel_fma`, `tiled`, `matrix`, `tensor_core`/`coop` (cần trace Vulkan hoặc Metal). |
| `FIRLIFT_FP4_GEMM` | `auto` | GEMM prefill FP4-KM. `auto`: micro-kernel Tensor Core cho shape được hỗ trợ trên thiết bị Vulkan `coopmatrix` hoặc Metal `simdmatrix` đã ghi trace, trừ khi đặt `FIRLIFT_GENERIC`. `coop`/`tensor_core` ép dùng (cần trace Vulkan hoặc Metal; yêu cầu K chia hết cho 64 và N >= 16); `generic` chọn kernel mỗi lane một cột. |
| `FIRLIFT_SLICE_REPORT` | `0` | Giá trị bất kỳ ngoài `0` sẽ in lý do các op được phát lại sớm nhất được chọn vào lát lift. |
| `FIRLIFT_SLICE_REPORT_N` | `40` | Số dòng lý do tối đa được in khi bật `FIRLIFT_SLICE_REPORT`. |
| `FIRLIFT_Q4_WORD_MAX_K` | `4096` | K lớn nhất của GEMV word-parallel khi `auto`. |
| `FIRLIFT_Q4_LANES`, `FIRLIFT_TG` | theo kernel | Số lane / độ rộng threadgroup của kernel parallel. |
| `FIRLIFT_GEMM_TILE`, `FIRLIFT_GEMM_REG` | `64x128`, `8x8` | Tile và block thanh ghi của GEMM f32. |
| `FIRLIFT_SIMD` | theo backend | `1`/`0`: cho phép reduce bằng simdgroup (tắt với trace Vulkan). |
| `FIRLIFT_GENERIC` | không đặt | Kernel mỗi lane một output, backend VM nào cũng chạy được. |
| `FIRLIFT_TARGET` | không đặt | `apple`: graph được dùng `simdgroup_matrix`. |
| `FIRLIFT_FLASH` | tiled | `row`: lift attention prefill thành kernel decode song song theo hàng thay vì flash dạng tile. |
| `FIRLIFT_NOFUSE` | không đặt | Không gộp op activation / vector / RoPE vào op bên cạnh. |
| `FIRLIFT_NOSIDE` | không đặt | Không ghép các kernel decode độc lập liên tiếp thành một lần chạy. |
| `FIRLIFT_NOREUSE` | không đặt | Mỗi output có buffer riêng (không dùng lại buffer tạm). |
| `BYBY_FIR_PROBE_STEPS` | mọi bước | Các bước recorder giữ, ví dụ `prefill:2,decode:1`. |
| `BYBY_FIR_PROBE_DATA` | `1` dưới `--run` | Gửi dữ liệu upload chính xác qua pipe dữ liệu (cần cho upload không tên như LUT). `0` khi mọi weight đều có tên trong `model.nimq`. |

Để lift Codestral cần `BYBY_MERGE_LORA=1` và `BYBY_MEMORY_LIMIT_MB=10000`.

Header trace ghi `device="coopmatrix"` cho cooperative matrix của Vulkan, `device="simdmatrix"` cho simdgroup matrix của Metal, hoặc `device="plain"` nếu không có. Micro-kernel Tensor Core được tự động chọn cho Vulkan có cooperative matrix hoặc Metal trên Apple GPU family 7+; các thiết bị khác dùng kernel generic.

Trace ghi backend và tính năng thiết bị của lần chạy, nên graph lift trên T4 nhắm
tới Tensor Core còn graph lift trên thiết bị không có cooperative matrix thì không.
