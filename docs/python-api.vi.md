
# Python và C ABI

[English](python-api.md) | [Tiếng Việt](python-api.vi.md)

`lib_c_api.nim` cung cấp phần core Nimformers dưới dạng shared library chỉ
có ABI C ổn định: `libnimformers.dylib`, `libnimformers.so` hoặc
`nimformers.dll`. Package Python `nimformers/` là lớp `ctypes`/NumPy mỏng
trên ABI đó. Các symbol cũng dùng được từ C, C++, Rust, Go hoặc FFI gọi C.

Phiên bản ABI là **2**, được trả bởi `nimf_abi_version()`.

Xem [hướng dẫn Python](python-guide.vi.md) cho cài đặt, ví dụ hoàn chỉnh, xem
checkpoint và xử lý lỗi. [Hướng dẫn lệnh Make](make-commands.vi.md) mô tả input
và output của từng target.

## Build

```sh
make build-for-python      # checks and builds the shared library
make python-test           # builds, then runs the pytest suite
make python-example        # runs examples/python/basic.py
make c-example             # builds and runs examples/c/smoke.c
```

Cài source trong môi trường riêng:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
make build-for-python
make python-test PYTEST='python -m pytest' PYTEST_FLAGS=-q
```

`pip install .` chỉ cài lớp Python, không build thư viện Nim. Chạy
`make build-for-python` trước, rồi sao chép shared library vào `nimformers/`,
đặt `NIMFORMERS_LIB=/path/to/libnimformers`, hoặc giữ thư mục gốc repo
trong đường tìm thư viện.

Tên output theo nền tảng:

| Nền tảng | Thư viện |
| --- | --- |
| macOS | `libnimformers.dylib` |
| Linux | `libnimformers.so` |
| Windows | `nimformers.dll` |

Loader Python tìm theo thứ tự:

1. Đối số `library=` tường minh.
2. Biến `NIMFORMERS_LIB`.
3. Thư mục package `nimformers/`.
4. Thư mục gốc repo.
5. Thư mục hiện tại và `build/`.
6. Dynamic loader của nền tảng qua `find_library("nimformers")`.


## Bắt đầu nhanh

```python
import numpy as np
from nimformers import Context

with Context() as ctx:                     # auto -> Metal or Vulkan
    a = np.random.randn(4, 8).astype(np.float32)
    b = np.random.randn(8, 16).astype(np.float32)

    y = ctx.matmul(a, b)                   # [4, 16]
    p = ctx.softmax(a, axis=-1)            # rows sum to 1
    ln = ctx.layernorm(a, np.ones(8), np.zeros(8))
    q = ctx.quantize(a, "int4_asym")
    w = q.dequantize()
    q.close()
```

`Context` có alias `Nimformers` và `nimformers.capi.Nimformers` để tương
thích wrapper ban đầu.

## Context

| Thành phần | Ý nghĩa |
| --- | --- |
| `Context(backend="auto", library=None, forbid_cpu_fallback=None)` | Tạo context; backend: `auto`, `metal`, `vulkan`, `cuda`/`opencl` (alias Vulkan cũ). |
| `ctx.backend` / `ctx.backend_id` | Tên và ID backend. |
| `ctx.abi_version`, `ctx.version` | Phiên bản ABI và thư viện. |
| `ctx.sync()` | Đồng bộ thiết bị theo mức backend hỗ trợ. |
| `ctx.set_cpu_fallback(bool)` | Cho phép phép toán thiếu kernel GPU chuyển về CPU hay không. |
| `ctx.close()` | Giải phóng context; cũng có thể dùng `with`. |

Các hàm toán nhận array-like, đổi sang `float32` liên tục và trả NumPy
array. Broadcasting của phép từng phần tử tuân theo NumPy.


## Phép toán số


### Từng phần tử và activation

```python
ctx.add(a, b)
ctx.sub(a, b)
ctx.mul(a, b)
ctx.div(a, b)
ctx.add_scalar(x, 1.0)
ctx.mul_scalar(x, 2.0)
ctx.neg(x)
ctx.pow(x, 2.0)
ctx.relu(x)
ctx.sqrt(x)
ctx.exp(x)
ctx.log(x)
ctx.sigmoid(x)
ctx.tanh(x)
ctx.apflu(x, alpha=0.1, beta=0.1)
ctx.apflu_backward(x, dy, alpha=0.1, beta=0.1)
```


### Nhân ma trận

```python
ctx.matmul(a, b)                    # NumPy batch broadcasting
ctx.matmul_nt(a, b)                 # a @ b.T, b is [..., N, K]
ctx.matmul2(a1, b1, a2, b2)         # one GPU sync, returns (a1@b1, a2@b2)
ctx.matmul_q4(a, packed, scales, zeros,
              k=..., n=..., group_size=0, groups_per_row=1)
ctx.linear_q4(x, quant_tensor)      # kind-aware helper
```

`linear_q4` dùng kernel int4 cho `int4_asym`; các kind khác được giải lượng
tử trước khi nhân. Int4 đối xứng dùng nibble bù hai, còn FP4-KM xen bytes
scale, nên cả hai không khớp decoder `(nibble - zero) * scale` của kernel
Q4. Khi gọi nhiều lần, upload một lần bằng `ctx.resident_q4(q)`, rồi dùng
`resident.linear(x)`.


### Chuẩn hóa và attention

```python
ctx.softmax(x, axis=-1)
ctx.layernorm(x, gamma, beta, eps=1e-5, axis=-1)
ctx.add_layernorm(a, b, gamma, beta, eps=1e-5, axis=-1)   # -> (sum, normalized)
ctx.layernorm_backward(dy, x, gamma, beta, eps=1e-5)      # -> (dx, dgamma, dbeta)

ctx.attention(q, k, v, mask=None, scale=None)             # -> (out, scores)
ctx.attention_backward(q, k, v, scores, dy, scale=None)   # -> (dq, dk, dv)
```

`q`, `k`, `v` có shape `[batch,heads,seq,head_dim]`; `scores` là
`[batch,heads,seq,seq]`. Kernel fused Metal/Vulkan hiện là **nhân quả**;
nhận `mask` để tương thích ABI nhưng không áp dụng mask đó trong các kernel này.


### Di chuyển dữ liệu và giảm chiều

```python
ctx.embedding_lookup(table, indices)     # table [vocab, dim]
ctx.gather(flat_source, indices)
ctx.compare(a, b, op="gt")               # gt/lt/eq/ge/le -> bool array
ctx.dropout(x, p=0.5, training=True, seed=0)
ctx.dropout_mask(n, p=0.5, seed=0)
ctx.argmax(x, axis=-1)                   # int64 indices
ctx.argmax_with_values(x, axis=-1)       # -> (values, indices)
ctx.masked_fill(x, mask, value)
ctx.where(cond, a, b)

ctx.sum(x, axis=-1)
ctx.mean(x, axis=-1)
ctx.max(x, axis=-1)
ctx.cumsum(x, axis=-1)
ctx.cumprod(x, axis=-1)
ctx.affine_scan(a, b, axis=-1)           # h[t] = a[t]h[t-1] + b[t], h[-1]=0

ctx.permute(x, dims)                     # dims is the new axis order
ctx.split(x, split_size, axis=0)         # list of chunks
ctx.split_slice(x, axis, offset, size)
ctx.concat([a, b, ...], axis=0)
ctx.stack([a, b, ...], axis=0)
```


## Tensor

`ctx.tensor(value)`, `ctx.zeros(shape, fill=0)` và `ctx.randn(shape, scale=1)`
tạo tensor Nim **sở hữu bộ nhớ**:

```python
t = ctx.tensor(np.arange(12, dtype=np.float32).reshape(3, 4))
t.shape, t.rank, t.numel
t.reshape(2, 6)
t.fill(0)
t.set_data(np.zeros((2, 6), np.float32))
view = t.numpy(copy=False)       # valid until t.close()
copy = t.numpy()                 # safe copy (default)
clone = t.clone()
t.close()
```


## Lượng tử hóa

```python
q = ctx.quantize(weight, "int4_asym")       # or fp32/int8/int4/fp8_e4m3/
                                            # fp8_e5m2/custom/auto/fp4_km
q.kind, q.shape, q.numel, q.nbytes
q.group_size, q.cf_bits                     # (exponent_bits, mantissa_bits)
q.data(), q.scales(), q.zeros()             # payload views/copies (read-only
                                            # when the checkpoint is mmapped)
deq = q.dequantize()
deq_t = q.dequantize_transposed()      # 2-D: [cols, rows]
q.close()
```

`fp4_km` cần tensor 2D có chiều cuối chia hết cho 32. Kind `custom` nhận
`exponent_bits=` và `mantissa_bits=`.

CustomFloat và APF:

```python
raw = ctx.customfloat_encode(x, exponent_bits=4, mantissa_bits=3)
x2 = ctx.customfloat_decode(raw, x.size, 4, 3)

e, m = ctx.apf_build_dtype(x, grad=None, rel_error_tol=1e-3)
raw, (e, m) = ctx.apf_cast_training(x)
x3 = ctx.apf_decode_training(raw, e, m, x.size)
```


## State dict NIMQ

```python
sd = ctx.load_state_dict("model.nimq", mapped=True)   # mmap payloads
sd = ctx.create_state_dict(arch=(vocab, hidden, heads, layers, ff_mult))
sd["model.layers.0.self_attn.q_proj.weight"] = q
sd.append("model.norm.weight", norm_q)
sd.names, sd.arch, len(sd)
sd.save("model.nimq")
sd.close()
```

`sd[name]` trả handle `QuantTensor` được cache trong state dict. Đóng state
dict cũng đóng các handle đó. `sd.to_dict()` giải lượng tử mọi tensor,
thuận tiện khi xem checkpoint nhỏ.

Tuple kiến trúc ở ví dụ theo quy ước Spectron/exporter. NIMQ lưu năm số
nguyên mà không áp một schema model chung; writer khác có thể gán ý nghĩa
khác. Xem loader tương ứng trước khi tạo metadata kiến trúc. Load state
dict Spectron chỉ cung cấp tensor, không tạo model training Spectron Python.


## Micro-VM — dispatch micro-kernel

VM là máy thanh ghi nhỏ trên buffer `float32` phẳng. R0 là chỉ số thread;
U0 đến U15 là thanh ghi địa chỉ số nguyên.

```python
from nimformers import Op

prog = ctx.vm_program()
prog.load_f32(1, 0, 0)      # R1 = mem[R0]
prog.add(1, 1, 1)           # R1 = R1 + R1
prog.store_f32(1, 0, 0)     # mem[R0] = R1
prog.halt()

data = np.arange(1, 65, dtype=np.float32)
out = ctx.vm_run(prog, data, tg_size=64)
```

| API | Ý nghĩa |
| --- | --- |
| `ctx.vm_program()` | Tạo chương trình. |
| `prog.emit(Op.ADD, a, b, c)` | Phát lệnh tổng quát; immediate float dùng `f32=`. |
| `prog.emit_word(w)` / `prog.set_words(list)` / `prog.words` | Bytecode thô. |
| `prog.load_f32`, `store_f32`, `load_const`, `add`, `mul`, `fma`, `exp`, `sigmoid`, `for_begin`, `if_begin`, `u_*`, ... | Helper phát lệnh theo tên. |
| `prog.run(ctx, memory, n_threads=None, tg_floats=0, tg_size=0)` | Dispatch rồi trả mảng kết quả; sửa tại chỗ input liên tục. |
| `prog.save(path)` / `VMProgram.load(lib, path)` | Lưu/load `.bvm`. |
| `VMProgram.merge(lib, *programs)` | Ghép chương trình, bỏ `HALT` ở giữa. |
| `prog.lower_msl(name)` / `prog.lower_glsl(name)` | Source kernel. |
| `prog.key(tag)` | Khóa chương trình xác định, không cần session GPU. |

`ctx.micro_vm_dispatch(words, memory, ...)` là cách gọi tắt với bytecode
thô. `tg_floats` tính bộ nhớ threadgroup theo số ô float; `tg_size` cố định
số thread/group, 0 để backend chọn.


## Tokenizer

```python
tok = ctx.tokenizer("test/codestral/models/hf")  # tokenizer.json + config
ids = tok.encode("Hello world")
text = tok.decode(ids)
ids = tok.apply_chat_template(user, system)
tok.vocab_size, tok.total_vocab_size, tok.special_ids
tok.close()

sp = ctx.sentencepiece("tokenizer.model")
ids = sp.encode("Hello")
sp.decode(ids)
sp.piece_id("▁Hello")
sp.close()
```

`apply_chat_template` báo lỗi nếu model không có chat template được hỗ trợ.


## Inference Llama

```python
cfg = ctx.llama_config("model/config.json")
with ctx.load_llama("model/config.json", "model.nimq",
                    upload_resident=True) as model:
    tok = ctx.tokenizer("model")
    text = model.generate(tok, "Write a Fibonacci function",
                          max_new_tokens=128, temperature=0.7, top_p=0.9,
                          use_chat_template=False)
```

Nhánh decode tối ưu yêu cầu `upload_resident=True` (mặc định); loader đưa
trọng số lượng tử hóa lên GPU một lần. Checkpoint nhiều GB cần thời gian
và RAM khi load. Token prompt ngoài vocabulary model bị từ chối với lỗi rõ ràng.


## Dùng từ C

Header công khai là [`include/nimformers.h`](../include/nimformers.h);
[`examples/c/smoke.c`](../examples/c/smoke.c) là consumer tối thiểu:

```sh
cc -Iinclude examples/c/smoke.c -L. -lnimformers \
   -Wl,-rpath,"$PWD" -o /tmp/nimf_smoke
/tmp/nimf_smoke
```

Quy tắc sở hữu và lỗi nằm đầu header. Status `0` là thành công;
`nimf_last_error()` mô tả lỗi. Chuỗi thư viện trả về như tên state dict,
text decode, source lower và khóa chương trình phải được giải phóng bằng
`nimf_free_string()`.


## Phạm vi API

| Module Nim | Phần ABI |
| --- | --- |
| `backend.nim` (`be*`) | từng phần tử, matmul, softmax, LayerNorm, attention, embedding, gather, so sánh, dropout, argmax, permute/split/concat, APFLU, CustomFloat và resident weights |
| `tensor_ops.nim` (`be*`) | giảm chiều, cumsum/cumprod, affine scan, dropout, so sánh, masked fill/where |
| `quant.nim` | `nimf_quantize` / `nimf_quant_dequantize`, truy cập payload và state dict NIMQ |
| `customfloat.nim` | encode/decode, chọn dtype APF và cast training |
| `vendor/bybylang/micro_kernel` | tạo chương trình, I/O `.bvm`, lower và dispatch |
| `test/deepseek-inference/native_hf_tokenizer.nim` | tokenizer HF |
| `test/codestral/sentencepiece_native.nim` | tokenizer SentencePiece |
| `test/deepseek-inference/nim_inference.nim` | cấu hình Llama, load trọng số và sinh token |

Các class layer/model trong `nimformer.nim` (`Linear`, `LayerNorm`,
`CausalSelfAttention`, `FeedForward`, `TransformerBlock`, `NimformerModel`)
và cache Jamba/Mamba là tổ hợp kernel phía Nim. Python tổ hợp các primitive
tương tự như `matmul_nt`, `layernorm`, `attention`, `linear_q4`, resident
weights. Ứng dụng riêng như `infer_lora`, training Jamba/Hybrid và Spectron
vẫn là chương trình CLI; các thành phần cơ bản được cung cấp qua API trên.

Wrapper không có autograd hay API Adam/`fit` cấp model. Phải tự tổ hợp
backward kernel như `layernorm_backward`/`attention_backward`. Dùng entry
point Nim cho trainer Spectron/Codestral có sẵn; xem [Spectron](spectron-guide.vi.md).


## Thread và vòng đời

- Mỗi thread dùng một context; tuần tự hóa GPU calls trên cùng context.
- `nimf_last_error()` là thread-local.
- Handle tensor/quant/state dict sở hữu dữ liệu; view `numpy(copy=False)`
  mất hiệu lực sau `close()`.
- `Prog.run` sửa input C-contiguous tại chỗ và trả lại mảng; truyền bản sao
  nếu cần giữ input gốc.


## Test

Bộ test nằm trong [`test/python_test/`](../test/python_test/):

```sh
make python-test          # prints every test case and PASSED/FAILED
make python-test PYTEST_FLAGS=-q   # quiet summary only
```

Kiểm tra các nhóm phép toán với NumPy, round-trip lượng tử hóa, save/load
NIMQ (gồm checkpoint Spectron thật khi có), micro-VM run/save/load/lower/merge,
resident weights, tokenizer và sinh Llama nhỏ từ checkpoint tổng hợp.

Còn có test tích hợp BybyLang import `lib_c_api` từ `.bybylang`, gọi C ABI
trực tiếp cùng các kiểm tra native DSL `gpu ...`:

```sh
make bybylang-test
```

Xem [`test/bybylang_test/README.md`](../test/bybylang_test/README.vi.md).
