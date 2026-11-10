# nfgraph2torch (graph torch → nimformers)

`other/nfgraph2torch` chuyển một module torch thành graph FIR của nimformers và ghi `model.fir`.
Viết bằng Python. Chiều ngược lại (`to_torch`) mô tả bên dưới.

```python
import nfgraph2torch as nf2t

graph = nf2t.from_torch(model, (x,))                 # graph formers.fir
nf2t.export(model, (token, position), "out_dir")     # out_dir/model.fir + config.json
values, shape = nf2t.evaluate(graph, model, (x,))    # chạy reference bằng Python để so với torch
print(nf2t.coverage(model, (x,)))                    # cần những ATen op nào và lower từng op ra sao
```

```bash
python -m nfgraph2torch package.module:make_model out_dir
python -m nfgraph2torch package.module:make_model --coverage
```

`make_model` trả về `(torch_module, example_inputs)`. `nfgraph2torch` tự tìm package `formers` trong
`other/formers.n/python`; đặt `NFGRAPH2TORCH_FORMERS` để dùng chỗ khác.

## API

| Hàm | Ý nghĩa |
|---|---|
| `from_torch(module, example_inputs, *, inputs=None, kwargs=None, config=None, name, optimize=True, strict=False, decompositions=None)` | Trả về graph `formers.fir`. `inputs` là danh sách `Input(role, dtype)`, mỗi input của người dùng một phần tử (`role` là `"token"`, `"position"` hoặc `""`). `config` là dict hoặc object có `to_dict()`. |
| `export(module, example_inputs, directory, *, report=False, **options)` | `from_torch` rồi `fir.export`; trả về đường dẫn `model.fir`. |
| `evaluate(graph, module, example_inputs)` | Chạy graph bằng evaluator reference của FIR với `module.state_dict()`. Trả về `(flat_values, shape)`. |
| `coverage(module, example_inputs=None, **options)` | Báo cáo các ATen op trong module và cách lower từng op. |
| `UnsupportedOp` | Ném khi ATen op không có kernel lẫn micro-kernel; thông báo nêu tên op và node. |

## Cách hoạt động

1. `torch.export` bắt module, `run_decompositions()` hạ xuống Core ATen (khoảng 160 op).
2. Mỗi node được replay trên tracer `formers.fir`. ATen op mà FIR đã có thành kernel FIR thường. Op còn
   thiếu thành micro-kernel, hoặc được ghép từ các op có sẵn.
3. `fir.export` ghi `model.fir` và `config.json`. Parameter giữ tên theo `state_dict` và không bao giờ
   bị đọc giá trị; buffer và hằng số được nhúng vào.

## Micro-kernel

Đăng ký qua hook custom-op của `formers.fir` (`formers/fir/custom.py`):

- `prod`, `cumprod`, `scatter` / `scatter_add`, `sort` / `argsort` / `topk`, `trunc`.
- `window`: convolution N chiều (stride, padding, dilation, groups) và max/avg pooling.
- Op ghép: floor, ceil, round, fmod, remainder, floor_divide, erf, norm, pad reflect/replicate,
  slice_scatter, index, index_put, adaptive avg pool, nearest upsample.

### Hook custom-op (`formers.fir.custom`)

| Hàm | Ý nghĩa |
|---|---|
| `register_op(name, *, reference, lower, replace=False)` | Thêm op vào graph. `reference(inputs, shapes, out_shape, attrs)` trả kết quả phẳng (cho `fir.evaluate`); `lower(emitter, node, ctx, kernel_builder)` sinh micro-kernel và trả `(tên_kernel, threads)`. |
| `custom_op(name, inputs, shape, attrs=None, dtype="f32")` | Ghi op vào trace đang chạy và trả tensor kết quả. Parameter trong `inputs` thành weight của checkpoint. |
| `register_tree_op(kind, emit)` | Thêm phép toán scalar cho bộ dựng kernel pointwise. |

## Chiều ngược: model.fir → torch

```python
back = nf2t.to_torch("out_dir/model.fir", weights=dict(model.state_dict()))   # một torch.nn.Module
logits = back(token, position)
```

`to_torch(source, weights=None, config=None)` đọc `model.fir` dạng `runtime graph` (đường dẫn hoặc text)
và trả về `FirTorch`, module có `forward(token, position)` trả logits. `weights` ánh xạ tên tensor
checkpoint sang tensor; chúng thành `nn.Parameter` cùng tên. Chỉ dùng ATen cơ bản, nên file FIR nào
VM Python chạy được thì cũng chạy được trong torch:

- Node `ext`: `fir.embedding` là `index_select`; `fir.linear` / `fir.linear_t` là `torch.matmul`.
- Node `micro`: một trình thông dịch vector hoá chạy bytecode của kernel, mỗi thread là một phần tử
  tensor. `add`/`sub`/`mul`/`div`/`exp`/`sigmoid`/`rsqrt`... là đúng op ATen đó, `uloadf32` là gather,
  `ustoref32` là `index_put_`, `ifbegin` là mask (`where`), `forbegin` là vòng lặp Python quanh các op
  tensor.

Tốc độ thấp hơn runner native rất nhiều (conv tốn một op tensor cho mỗi phép nhân-cộng; ResNet-18 với
8 kênh gốc trên ảnh 32×32 mất khoảng 35 giây trên CPU). Độ phủ đầy đủ với mọi lệnh VM Python thông
dịch được; kernel dùng lệnh thread-group, atomic hoặc barrier bị từ chối như trong VM. Nhiều thread ghi
cùng một địa chỉ thì thứ tự không xác định.

Vòng torch → FIR → torch của ResNet-18 (bọc thành model token → ảnh → logits vì ABI runner chỉ nhận
`token` và `position`) khớp bản gốc tới 4e-7.

## Giới hạn

- Runner FIR chỉ nạp `token` và `position` và đọc `logits`, nên xuất `model.fir` cần graph dạng LM. Graph
  khác (ví dụ bộ phân loại ảnh như ResNet-18) vẫn convert ra graph FIR và chạy trên evaluator reference,
  nhưng chưa xuất được `model.fir` chạy được.
- `sort` / `topk` là O(n²); không dành cho vocab rất lớn.
- Chưa lower: transposed convolution, pooling `ceil_mode`, adaptive pooling không chia hết, bilinear
  upsample, grid_sampler, as_strided, col2im, advanced indexing nhiều chỉ số, trig nghịch đảo,
  `scatter_reduce` khác sum.

## Test

```bash
cd other/nfgraph2torch/tests
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest test_ops.py test_resnet.py test_reverse.py   # evaluator reference
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest test_vm.py                   # model.fir trên VM của FIR
```
