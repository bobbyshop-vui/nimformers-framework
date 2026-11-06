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
