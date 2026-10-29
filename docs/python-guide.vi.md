
# Dùng Nimformers từ Python

[English](python-guide.md) | [Tiếng Việt](python-guide.vi.md)

Package `nimformers/` dùng `ctypes` gọi shared library được build từ
`lib_c_api.nim`. API thao tác với NumPy, tensor có sở hữu bộ nhớ, quantization,
NIMQ, resident weights, micro-VM, tokenizer và Llama inference.


## Cài từ source

Từ thư mục gốc repo, tạo môi trường riêng rồi build thư viện:

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install -e '.[test]'
make build-for-python
make python-example PYTHON=python
```

`pip install -e '.[test]'` cài Python wrapper, NumPy và pytest; không compile
Nim. `make build-for-python` mới tạo `libnimformers.dylib` trên macOS,
`libnimformers.so` trên Linux hoặc `nimformers.dll` trên Windows. Cần Nim,
C compiler và Metal/Vulkan runtime cho các phép tính GPU.

Nếu chỉ dùng package, có thể dùng `python -m pip install -e .` và không cài
test extra. Đọc toàn bộ signatures ở [Python API](python-api.vi.md).


## Chọn backend và tìm shared library

```python
from nimformers import Context, library_path

print(library_path())
with Context(backend="auto", forbid_cpu_fallback=True) as ctx:
    print(ctx.backend, ctx.abi_version, ctx.version)
```

`auto` tự dò GPU; chọn `metal` trên macOS hoặc `vulkan` khi muốn cố định.
`cuda`/`opencl` được chuyển sang Vulkan. Không có `Context("cpu")`.
`forbid_cpu_fallback=True` yêu cầu báo lỗi nếu phép toán không chạy được
trên backend GPU. Đây không phải một hệ thống autograd như PyTorch.

Khi chạy từ project khác, cài wrapper rồi trỏ tới file thư viện đầy đủ:

```sh
# macOS example; use libnimformers.so on Linux.
export NIMFORMERS_LIB="$PWD/libnimformers.dylib"
```

Hoặc truyền `library="/absolute/path/libnimformers.dylib"` cho `Context`.
Loader còn tìm trong thư mục package, repo, current directory và `build/`.


## Phép toán đầu tiên

Lưu thành `python_demo.py` ở thư mục gốc rồi chạy `python python_demo.py`:

```python
import numpy as np
from nimformers import Context

with Context(forbid_cpu_fallback=True) as ctx:
    a = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.float32)
    b = np.array([[1, 0], [0, 1], [1, 1]], dtype=np.float32)
    y = ctx.matmul(a, b)
    np.testing.assert_allclose(y, [[4, 5], [10, 11]])
    print(y.shape)                         # (2, 2)
    p = ctx.softmax(y)
    print(p.sum(axis=-1))                   # approximately [1, 1]
    gamma = np.ones(2, dtype=np.float32)
    beta = np.zeros(2, dtype=np.float32)
    print(ctx.layernorm(y, gamma, beta))
```

Phương thức nhận array-like, chuyển thành `float32` liên tục trong bộ nhớ
và thường trả NumPy array. `matmul` nhận `[M,K] @ [K,N]`; các trục batch có
thể broadcast. `matmul_nt(x, weight)` nhận weight `[N,K]` và tính `x @ weight.T`.

Các nhóm API thường dùng:

- Tính số: `add`, `sub`, `mul`, `div`, `relu`, `sigmoid`, `tanh`, `exp`, `log`.
- Neural network: `softmax`, `layernorm`, `layernorm_backward`, `attention`,
  `attention_backward`, `embedding_lookup`.
- Giảm/đổi shape: `sum`, `mean`, `max`, `argmax`, `permute`, `concat`, `split`.
- Scan và mask: `cumsum`, `cumprod`, `affine_scan`, `where`, `masked_fill`, `dropout`.

`attention` nhận `q/k/v` dạng `[batch, heads, seq, head_dim]`, trả `(out, scores)`.
Kernel fused Metal/Vulkan hiện dùng causal attention; đối số `mask` tùy ý
chưa được các kernel này áp dụng.


## Tensor và vòng đời bộ nhớ

```python
import numpy as np
from nimformers import Context

with Context() as ctx:
    with ctx.tensor(np.arange(12, dtype=np.float32).reshape(3, 4)) as tensor:
        print(tensor.shape, tensor.numel)   # (3, 4), 12
        tensor.reshape(2, 6)
        owned_copy = tensor.numpy()        # copy remains valid after close
        print(owned_copy.shape)            # (2, 6)
```

`ctx.tensor`, `ctx.zeros`, `ctx.randn` tạo handle cần `close()` hoặc `with`.
`tensor.numpy(copy=False)` là view mượn bộ nhớ; dùng nó trong thời gian tensor
còn mở. Các handle resident GPU phải được đóng trước context. Các phép
`ctx.matmul(...)` trả NumPy array độc lập, không cần gọi `close()` cho array.


## Quantization và resident weight

Ví dụ độc lập sau tạo weight `[N,K]`, lượng tử hóa rồi chạy linear:

```python
import numpy as np
from nimformers import Context

rng = np.random.default_rng(42)
x = rng.standard_normal((2, 64)).astype(np.float32)
weight = rng.standard_normal((32, 64)).astype(np.float32)

with Context() as ctx:
    with ctx.quantize(weight, "int4_asym") as quant:
        restored = quant.dequantize()
        y = ctx.linear_q4(x, quant)         # (2, 32)
        np.testing.assert_allclose(y, x @ restored.T, rtol=1e-4, atol=1e-3)
        with ctx.resident_q4(quant) as resident:
            print(resident.linear(x).shape) # upload the weights once
```

Các kind: `fp32`, `int8`, `int4`, `int4_asym`, `fp8_e4m3`, `fp8_e5m2`,
`custom`, `auto`, `fp4_km`. `fp4_km` cần tensor 2D có chiều cuối chia hết cho
32. `resident_q4` nhận `int4_asym`; `linear_q4` xử lý các kind khác qua
dequantize. Với `resident_f32`, weight có layout `[K,N]`, khác weight `[N,K]`
của `resident_q4`.


## Đọc checkpoint Spectron bằng Python

```python
from nimformers import Context

with Context() as ctx:
    with ctx.load_state_dict("test/spectron/models/finetune.nimq", mapped=True) as sd:
        print("arch:", sd.arch)
        print("tensors:", len(sd), sd.names[:5])
        print("embedding:", sd["embed.weight"].shape)
        if "__meta_step__" in sd:
            print("trained step:", int(sd["__meta_step__"].dequantize()[0]))
```

`mapped=True` ánh xạ payload checkpoint vào bộ nhớ; dequantize từng tensor
khi cần. Với Spectron, `arch` theo thứ tự `(vocab, embedDim, nHeads, nLayers,
ffMult)`. NIMQ giữ năm số kiến trúc do bên ghi định nghĩa, nên khi đọc loại
checkpoint khác cần xem loader tương ứng.

`sd[name]` là `QuantTensor` handle được cache và được đóng khi state dict
đóng. Các NumPy view mượn payload cần được dùng trước lúc đóng. Tránh
`sd.to_dict()` với model lớn nếu chỉ cần xem metadata, vì nó dequantize mọi
tensor. Đọc NIMQ của Spectron không đồng nghĩa có Python Spectron model class.

Để tạo một state dict riêng, dùng `ctx.create_state_dict(arch=(...))`, gán
`QuantTensor` từ `ctx.quantize(...)` qua `sd["name"] = quant`, rồi `sd.save(path)`.


## Viết micro-kernel từ Python

```python
import numpy as np
from nimformers import Context

with Context() as ctx:
    with ctx.vm_program() as prog:
        prog.load_f32(1, 0, 0)              # R1 = memory[R0], R0 = thread id
        prog.add(1, 1, 1)
        prog.store_f32(1, 0, 0)
        prog.halt()
        data = np.arange(1, 65, dtype=np.float32)
        expected = data * 2                 # compute first because vm_run modifies data
        out = ctx.vm_run(prog, data, n_threads=64, tg_size=64)
        np.testing.assert_array_equal(out, expected)
```

`VMProgram` cho phép lưu `.bvm`, load lại và lower sang MSL/GLSL. Chương trình
tự chịu trách nhiệm về chỉ số buffer và kích thước dispatch; ví dụ này có
đúng 64 phần tử và 64 thread. Xem [micro-VM API](microkernel-api.vi.md).


## Inference model và gọi trainer

Python có `ctx.load_llama(configPath, nimqPath)` và `model.generate(...)`
cho Llama/DeepSeek tương thích loader. Ví dụ đầy đủ tại
[Python API — inference Llama](python-api.vi.md#inference-llama).

Spectron và Codestral LoRA hiện dùng entry point Nim/CLI cho training và
inference riêng; Python API chưa có `SpectronModel.fit`, `backward()` tự
động hay optimizer Adam cấp model. Muốn điều khiển trainer từ Python có thể
dùng `subprocess.run(["make", "run-metal"], cwd=repo, check=True)`; cấu hình
Spectron vẫn ở constants trong source. Xem [Spectron](spectron-guide.vi.md).


## Test và lỗi thường gặp

```sh
make python-test PYTEST='python -m pytest' PYTEST_FLAGS=-q
```

- `ModuleNotFoundError: nimformers`: cài wrapper vào đúng interpreter bằng
  `python -m pip install -e .`, hoặc chạy từ thư mục repo.
- Không tìm thấy shared library: chạy `make build-for-python`, kiểm tra
  `library_path()` hoặc đặt `NIMFORMERS_LIB` tới file đầy đủ.
- Thiếu symbol `nimf_*`: rebuild thư viện và kiểm tra không trỏ nhầm một
  bản library cũ.
- Không có GPU/không truy cập được Metal: kiểm tra GPU runtime và quyền truy
  cập thiết bị của tiến trình; package không cung cấp backend CPU thay thế.
- Lỗi shape/backend từ C ABI được đưa lên thành `NimformersError`; lỗi input
  do Python wrapper kiểm tra thường là `ValueError` hoặc `TypeError`.

Các pytest fixture có thể skip khi không tạo được context; xem tổng số
passed/skipped trước khi kết luận bộ test đã thực thi GPU. GPU calls trên
cùng context cần được tuần tự hóa.
