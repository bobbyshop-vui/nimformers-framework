
# Kiểm thử tích hợp BybyLang

[English](README.md) | [Tiếng Việt](README.vi.md)

Bộ test dùng DSL/compiler BybyLang trong
[`vendor/bybylang`](../../vendor/bybylang) để kiểm tra Nimformers.
Xem [tham khảo API test BybyLang](../../docs/bybylang-tests.vi.md) để biết
giải thích theo từng dòng cho DSL và các lời gọi C ABI.

- `syntax_test.bybylang` — cú pháp ngôn ngữ: function có tham số và kiểu trả
  về, khối type/const, `case`/`try`, gán field/index/toán tử gộp và dòng trống
  trong khối.
- `capi_test.bybylang`: import `../../nimformers_lib` qua `import nim` và gọi C
  ABI: context/backend, cộng từng phần tử, matmul, softmax, LayerNorm,
  lượng tử int8, dispatch micro-VM và fused attention.
- `native_gpu_test.bybylang`: lệnh native DSL `gpu ...` trên cùng runtime
  `vendor/bybylang/gpubackend.nim`, cùng block `micro kernel ...:` phát
  bytecode rồi lower qua micro-VM Nimformers.
- `cpu_test.bybylang`: chạy DSL, block `session:` và micro-kernel trên backend
  CPU được chọn bằng `gpu backend is "cpu"`.
- `trace_test.bybylang`, `diag_test.bybylang` và `cpu_kernel_regression.nim`:
  kiểm tra core trace, chẩn đoán lỗi nguồn và kết quả kernel CPU.
- `nim.cfg`: search path Nim khi executable sinh ra import `nimformers_lib`,
  gồm module gốc và đường dẫn các module model cần dùng.
- `run.sh`: compile AOT, chạy các chương trình tích hợp và kiểm tra chẩn đoán.

Chạy:

```bash
./test/bybylang_test/run.sh
# or
make bybylang-test
```

Runner đặt `BYBYLANG_NIMCACHE=nimcache/bybylang-test`, nên cache compiler
được dọn qua cleanup `nimcache` thông thường. Các file sinh ra `*_bin` và
`*_bin.nim` được git-ignore.

Các ca `auto` dùng Metal hoặc Vulkan khi có. `cpu_test.bybylang` luôn chọn
CPU và chạy được trên máy không có GPU.

BybyLang phân tích dòng không phải command có `=` thành assignment
`left = right`. Tránh `=` trong comment cuối dòng Nim thô; đặt comment
trên dòng riêng.
