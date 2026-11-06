# FIR (formers IR)

[English](fir.md) | [Tiếng Việt](fir.vi.md)

FIR là một file text đi kèm model, mô tả đồ thị decode: chạy **micro-kernel** nào, gọi
**kernel backend** nào, cùng tên tensor và kích thước (lấy từ `config.json` của model).
FIR không thêm VM hay assembler thứ hai. Micro-kernel là của framework (bytecode
`session_vm`, khối DSL `micro kernel` của BybyLang, JIT trong `jit_vm`) và được gọi theo
tên từ một registry. Nhờ vậy model có thể chọn hoặc thêm kernel mà framework không cần
hard-code kiến trúc đó.

Model đầu tiên là Mamba-3 SISO 1.5B: [`models/mamba3-1.5b/model.fir`](../models/mamba3-1.5b/model.fir).
Định dạng đầy đủ và ABI kernel nằm trong [`models/FIR.md`](../models/FIR.md) (tiếng Anh).

## Các loại node

| Loại | Ý nghĩa | Ví dụ |
| --- | --- | --- |
| `micro` | micro-kernel đã đăng ký (bytecode VM, JIT trên GPU) | `node scan micro m3.ssm_step ...` |
| `ext` | kernel backend theo tên (GEMV Q4, embedding) | `node project ext m3.gemv_q4 ...` |
| `builtin` | phép nhỏ (`rms_norm`, `add`) | `node r builtin add ...` |

## Cách dùng

```sh
make formers-cli
# kiểm tra file .fir mà không load weights: resolve mọi tên, build mọi micro program
other/formers.n/formers-cli --fir-check models/mamba3-1.5b/model.fir
# chạy Mamba-3 1.5B với micro-kernel do file FIR chọn
other/formers.n/formers-cli --models test/mamba3/models \
  -m test/mamba3/models/mamba3_siso_1.5b.nimq \
  --config test/mamba3/models/mamba3-siso-1.5b/config.json \
  --tokenizer test/mamba3/models/mamba3-siso-1.5b \
  --no-lora -p 'def fib(n):' -n 24 --temp 0 --backend metal \
  --fir models/mamba3-1.5b/model.fir
```

Khi có `--fir`, formers.n in `FIR: provider used for N programs`. Không có cờ này runtime
vẫn dùng builder trực tiếp; kết quả giống nhau.

## Test

```sh
make -C other/formers.n fir-test    # hoặc: make formers-fir-test
```

Test parse file 1.5B, build mọi micro program với kích thước 1.5B, chạy kernel
prepare/scan/carry của Mamba-3 trên trình thông dịch CPU và so với bản tham chiếu float
thường (sai số `1e-4`), rồi so program của FIR provider với builder trực tiếp.

## Đã kiểm chứng và chưa kiểm chứng

- Đã kiểm chứng: Mamba-3 1.5B (base, không LoRA), greedy 24 token, Metal và Vulkan
  (MoltenVK) trên Intel Iris Plus 640; chạy có và không có FIR cho cùng văn bản.
- Chưa kiểm chứng: GPU khác, chạy LoRA/Linear++, sinh dài. `BYBY_ONE_DISPATCH=1` không
  tương thích với `--fir`.
- FIR chọn phép đã đăng ký cho các vai trò activation cố định. Nó không đổi thứ tự lịch
  chạy hay thay phần weight/projection resident; kiến trúc có lịch khác còn cần adapter
  runtime.

## GPT-2 small (124M)

[`models/gpt2-125m/model.fir`](../models/gpt2-125m/model.fir) mô tả toàn bộ bước decode:
embedding token + vị trí, layernorm có gamma/beta, QKV gộp, attention nhân quả có KV
cache riêng từng layer, projection + bias + residual, MLP `gelu_new`, layernorm cuối
và lm_head dùng chung `wte`. Kích thước và epsilon lấy từ `config.json`.

```sh
make formers-cli
other/formers.n/formers-cli --fir-check models/gpt2-125m/model.fir
other/formers.n/formers-cli --models models/gpt2-125m \
  --fir models/gpt2-125m/model.fir --raw -p "Hello, my name is" -n 32 --temp 0 --backend cpu
# GPU check (outside the sandbox)
other/formers.n/formers-cli --models models/gpt2-125m \
  --fir models/gpt2-125m/model.fir --raw -p "Hello, my name is" -n 32 --temp 0 --backend metal
```

**formers.n không có gì dựng sẵn cho model này.** Thư mục model là model đồ thị tự chứa khi
`model.fir` của nó (hoặc file truyền bằng `--fir`) khai báo `runtime graph`; khi đó
formers.n chuyển thư mục đó sang bộ chạy đồ thị FIR chung, bất kể kiến trúc. Mọi thứ riêng
của model đều đến từ file:

| Khai báo | Ý nghĩa |
|---|---|
| `runtime graph` | file tự mang đồ thị, weight và kernel |
| `sym TÊN = BIỂU_THỨC` | symbol; mọi key số nguyên dương của `config.json` đã là symbol. Bộ chạy cần `layers`, `positions`, `vocab` |
| `epsilon KEY` | hằng số thực lấy từ một key config (`eps` trong kernel) |
| `require config KEY GIÁ_TRỊ` | giá trị config nếu có phải bằng GIÁ_TRỊ (vắng thì bỏ qua) |
| `require sym A = B` | hai biểu thức nguyên trên symbol phải bằng nhau |
| `prefix P` | tiền tố tên tensor thử thêm sau tên gốc |
| `buf TÊN f32 N input:token` / `input:position` / `output:logits` | hợp đồng vào/ra lúc chạy |
| `weight TÊN dtype TENSOR tensor [dims]` | tensor checkpoint (`{i}` = layer), `f32` hoặc `q4`, shape `[in,out]` cho linear |
| `node N ext fir.embedding IDX BẢNG -> OUT` | tra hàng (chỉ số lấy từ vai trò `input:`) |
| `node N ext fir.linear X W -> Y` | `Y = X*W`; weight `q4` chạy Q4 resident, `f32` chạy matmul F32 |
| `node N ext fir.linear_t X W -> Y` | `Y = X*W^T` cho weight `[out,in]` (lm_head dùng chung bảng embedding) |
| `node N micro TÊN ... -> ...` | kernel định nghĩa trong file (bên dưới) hoặc kernel của framework |

Cần checkpoint safetensors tensor F32, `config.json` và `tokenizer.json` HF
(`bos_token_id` / `eos_token_id` từ config). Hỗ trợ completion và chat `-cnv` (template lấy từ
`chat_template.jinja` hoặc `tokenizer_config.json` trong thư mục model). Không có
partial offload, vượt `positions` thì báo lỗi. Checkpoint được
đối chiếu với weight đã khai báo (tên, dtype, shape) trước khi nạp. `--fir-check` kiểm tra
file không cần weight. `--fir-f32` chạy weight `q4` ở fp32 (mốc đúng).

**Kernel nằm trong file model.** GPT-2 cần các phép framework chưa có (layernorm có
gamma/beta, `gelu_new`, cộng bias, attention nhân quả có cache), nên `model.fir` tự định
nghĩa chúng trong khối `kernel ... endkernel`. Mỗi dòng thân là đúng một lời gọi
`emit*` của `session_vm`: mnemonic là tên emit viết thường, bỏ `emit`
(`emitULoadF32` -> `uloadf32`), thanh ghi `0..31`, đối số là biểu thức số trên symbol của
model, `eps`, `pi`, `inN.size` / `outN.size` và địa chỉ buffer `@inN`, `@outN`, `@buffer`:

```
kernel g2.add in=2 out=1 size=in0,in1,out0 threads=out0.size tg=1 tgbytes=16
  rtou 0 0
  uloadf32 1 0 @in0
  uloadf32 2 0 @in1
  add 1 1 2
  ustoref32 1 0 @out0
endkernel
```

Tùy chọn: `in=`/`out=` số cổng, `threads=`/`tg=`/`tgbytes=` hình học dispatch, `size=` các
buffer phải cùng số phần tử, `expect=CỔNG:BIỂU_THỨC,...` số phần tử chính xác. Parser và
`--fir-check` từ chối mnemonic lạ, sai số đối số, thanh ghi trên 31, `forbegin`/`ifbegin`
không cân, tên lạ và sai shape. Kernel định nghĩa trong file được ưu tiên hơn kernel
cùng tên của registry (Mamba-3 vẫn dùng kernel `m3.*` có sẵn). Phần chữ này chỉ là
lớp vỏ cho `emit*`; FIR không có bytecode hay interpreter riêng.

**Thực thi.** `fir_graph.bybylang` duyệt node theo thứ tự file, lặp phạm vi layer. Các
lớp tuyến tính nằm resident: `c_attn`, `c_proj` của attention, `c_fc` và `c_proj` của MLP
được lượng tử hóa lúc nạp (int4 bất đối xứng theo nhóm 32) và giữ trên GPU;
`--fir-f32` giữ fp32 làm mốc đúng. lm_head dùng chung `wte` ở fp32. Một bước decode là
một session GPU, kernel FIR và linear resident được encode theo thứ tự; K/V và scratch
nằm trên GPU, chỉ đọc logits về. Metal chạy MSL đã biên dịch, Vulkan chạy GLSL JIT hiện
có, CPU chạy `cpu_backend.vmRun` của framework. Prefill chạy từng token qua cùng đường.

`make formers-fir-test` chạy thêm fixture ngẫu nhiên cho từng micro-kernel bằng
`runGrid`, kiểm tra không đọc KV tương lai, và hai đồ thị nhỏ hai layer qua CPU backend
so với công thức float độc lập (sai số `1e-4`). Kiểm tra model thật với HF cài sẵn,
hoàn toàn offline:

```sh
python3 models/gpt2-125m/check_hf.py --backend cpu
python3 models/gpt2-125m/check_hf.py --backend metal
```

Decode trên Intel Iris Plus 640 ở chế độ chat (xem [benchmark](../models/gpt2-125m/benchmark.md)):
Q4 8,57 tok/s trên Metal và 11,40 tok/s trên Vulkan; `--fir-f32` 10,78 tok/s trên Metal và
8,54 tok/s trên Vulkan. Đường fp32 khớp cả 32 token greedy của Hugging Face offline
trên CPU và Metal.
