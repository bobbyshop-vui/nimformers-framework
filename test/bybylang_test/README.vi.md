
# Kiểm thử tích hợp BybyLang

[English](README.md) | [Tiếng Việt](README.vi.md)

Bộ test dùng DSL/compiler BybyLang trong
[`vendor/bybylang`](../../vendor/bybylang) để kiểm tra Nimformers.
Xem [tham khảo API test BybyLang](../../docs/bybylang-tests.vi.md) để biết
giải thích theo từng dòng cho DSL và các lời gọi C ABI.

- `capi_test.bybylang`: import `../../lib_c_api` qua `import nim` và gọi C
  ABI: context/backend, cộng từng phần tử, matmul, softmax, LayerNorm,
  lượng tử int8, dispatch micro-VM và fused attention.
- `native_gpu_test.bybylang`: lệnh native DSL `gpu ...` trên cùng runtime
  `vendor/bybylang/gpubackend.nim`, cùng block `micro kernel ...:` phát
  bytecode rồi lower qua micro-VM Nimformers.
- `nim.cfg`: search path Nim khi executable sinh ra import `lib_c_api`,
  gồm module gốc và đường dẫn các module model cần dùng.
- `run.sh`: compile AOT và chạy cả hai test.

Chạy:

```bash
./test/bybylang_test/run.sh
# or
make bybylang-test
```

Runner đặt `BYBYLANG_NIMCACHE=nimcache/bybylang-test`, nên cache compiler
được dọn qua cleanup `nimcache` thông thường. Các file sinh ra `*_bin` và
`*_bin.nim` được git-ignore.

Cả hai test cần GPU Metal/Vulkan như framework.

BybyLang phân tích dòng không phải command có `=` thành assignment
`left = right`. Tránh `=` trong comment cuối dòng Nim thô; đặt comment
trên dòng riêng.
