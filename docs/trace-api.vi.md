# Trace API (`nf.trace`)

[English](trace-api.md) | [Tiếng Việt](trace-api.vi.md)

**Trace** ghi lại việc một lần chạy model đã làm trên GPU: các buffer, các chương trình micro-VM, mọi
dispatch và các đầu vào host của từng bước. Bộ lift FIR (`firlift`) biến trace thành `model.fir`, rồi
`formers-cli` chạy file đó mà không cần code gốc. `nf.trace` là API duy nhất để bật và tắt recorder
từ Python, Nim, BybyLang và C.

```text
code model ──nf.trace──▶ run.fprobe (+ run.fprobe.data) ──firlift──▶ model.fir ──formers-cli──▶ token
```

| Ngôn ngữ | Điểm vào | Định nghĩa ở |
| --- | --- | --- |
| Python | `nimformers.trace(...)`, `trace_active()`, `trace_file()`, `trace_stop()` | `nimformers/tracing.py` |
| Nim | `traceStart`, `traceStop`, `withTrace`, `traceStep`, `traceActive`, `traceFile` | `trace.nim` (cũng được `import nimformers` re-export) |
| BybyLang | `import nim "trace"`, rồi dùng các tên Nim ở trên | `trace.nim` |
| C | `nimf_trace_start`, `nimf_trace_stop`, `nimf_trace_active`, `nimf_trace_file`, `nimf_probe_step` | `include/nimformers.h` |

Recorder là của cả tiến trình: mỗi lúc chỉ có một trace. Bật trace mới khi trace cũ còn chạy thì trace
cũ bị dừng trước.

## Tham số

| Tham số (Python / Nim) | Mặc định | Ý nghĩa |
| --- | --- | --- |
| `path` | bắt buộc | File trace. Thư mục chứa nó phải tồn tại. |
| `steps` | `""` (ghi hết) | Những bước host cần ghi: `decode:N` và `prefill:N` cách nhau bằng dấu phẩy, ví dụ `"decode:2"` hoặc `"prefill:8,decode:2"`. `prefill:N` đếm token, `decode:N` đếm bước. Lifter cần **ít nhất hai bước decode** để phân biệt trạng thái lặp lại với trạng thái thay đổi. |
| `data` | `True` | Ghi thêm byte chính xác của mọi upload từ host vào `<path>.data`. Cần cho trọng số, bảng rope và mọi buffer không nằm trong checkpoint. Với `False` chỉ lưu fingerprint nên checkpoint phải cung cấp trọng số. |
| `data_max` / `dataMax` | `0` (không giới hạn) | Với `data`, bỏ qua upload lớn hơn số byte này. Upload bị bỏ vẫn giữ fingerprint nên checkpoint vẫn khớp được. |

`<path>.data` được tạo khi upload đầu tiên được ghi. Model không có upload host nào trong bước được
ghi thì không có file phụ, kể cả khi `data=True`.

> **Cảnh báo: `data=True` có thể làm đầy ổ cứng.** File phụ là bản sao từng byte của mọi upload từ host
> trong các bước được ghi. Nếu driver của model upload trọng số ngay trong bước, **cả model** sẽ bị
> ghi vào đó, đúng dtype đã upload. Với `data_max=0` mặc định (không giới hạn), model 1,1 tỷ tham số
> upload dạng f32 tạo ra `trace.data` 4,5 GB, model 0,5 tỷ tham số tạo ra 2,1 GB. Model 7 B ghi khoảng
> 28 GB, và nếu nhiều bước được ghi đều upload lại dữ liệu thì mỗi bước ghi thêm một lần. File được ghi
> trong lúc model chạy, nên ổ đầy sẽ làm lượt chạy dừng giữa chừng và có thể làm máy không dùng được
> cho đến khi giải phóng dung lượng.
>
> - Luôn đặt `data_max` (ví dụ `4_000_000`) để bỏ qua upload lớn và chỉ giữ bảng nhỏ như cos/sin của
>   rope. Upload bị bỏ vẫn giữ fingerprint, trọng số lấy từ checkpoint.
> - Kiểm tra dung lượng trống trước, và xoá `<path>.data` sau khi lift xong; nó chỉ là file build.
> - Không commit: giữ `trace.data` và `*.fprobe.data` trong `.gitignore`.
> - Biến môi trường `BYBY_FIR_PROBE_DATA=1` có cùng tác dụng, giới hạn là `BYBY_FIR_PROBE_DATA_MAX`.
>   Đặt `BYBY_FIR_PROBE_DATA=1` mà không có giới hạn là tổ hợp nguy hiểm.

Đầu vào sai sẽ báo lỗi (`NimformersError` ở Python, `ValueError`/`IOError` ở Nim, mã khác 0 ở C) và
không để trace nào chạy: thư mục không tồn tại, đường dẫn có xuống dòng, mục `steps` không phải
`decode:N`/`prefill:N`, hoặc số âm.

## Python

```python
import nimformers as nf

with nf.Context() as ctx:
    model = MyModel(ctx)                       # tạo buffer, compile kernel
    with nf.trace("run.fprobe", steps="decode:2"):
        for pos, tok in enumerate([1, 100]):
            model.step(tok)                    # Session.step() đánh dấu mỗi bước
```

| API | Ý nghĩa |
| --- | --- |
| `nf.trace(path, steps="", data=True, data_max=0)` | Bắt đầu ghi và trả về `Trace`. Dùng như context manager hoặc với `.stop()`. |
| `Trace.stop()` | Ghi trace ra file và đóng file phụ. Gọi hai lần cũng không sao. |
| `Trace.active` | `True` khi bản ghi này đang là bản đang chạy. |
| `Trace.path`, `Trace.steps`, `Trace.data`, `Trace.data_max` | Các đối số lúc bắt đầu. |
| `Trace.data_path` | `path + ".data"`, hoặc `None` khi `data=False`. |
| `nf.trace_active()` | Có trace nào đang chạy trong tiến trình. |
| `nf.trace_file()` | Đường dẫn trace của tiến trình, `""` nếu chưa bật. |
| `nf.trace_stop()` | Dừng trace đang chạy. |

**Thứ tự không quan trọng.** Recorder giữ bytecode của mọi kernel ngay khi compile, dù trace có đang
chạy hay không, nên cách dùng quen thuộc (dựng model rồi mới `with nf.trace(...)`) cho trace lift được ở
mọi ngôn ngữ. `nf.trace` còn compile lại các kernel của `Session` còn sống ngay sau khi bắt đầu, như
một lớp bảo vệ thứ hai.

Đánh dấu bước do `Session.step()` và `Session.set_step_inputs()` làm
(xem [Python và C ABI](python-api.vi.md#buffer-resident-và-session)). Mỗi bước phải ghi token và vị trí
bằng `set_step_inputs` để lifter gắn chúng vào đầu vào `token` và `position` của runner.

## Nim

```nim
import nimformers          # re-export trace.nim; `import trace` cũng được

withTrace("run.fprobe", "decode:2"):
  for pos, tok in prompt:
    traceStep(tsDecode, pos, pos, tok):
      runOneToken(tok)
```

| Ký hiệu | Ý nghĩa |
| --- | --- |
| `traceStart(path, steps = "", data = true, dataMax = 0)` | Bắt đầu ghi. Báo `IOError`/`ValueError` khi đầu vào sai. |
| `traceStop()` | Ghi file và đóng. Gọi hai lần an toàn. Khôi phục `BYBY_FIR_PROBE_DATA` và `BYBY_FIR_PROBE_DATA_MAX`. |
| `traceActive(): bool`, `traceFile(): string` | Truy vấn trạng thái. |
| `withTrace(path, body)`, `withTrace(path, steps, body)`, `withTrace(path, steps, data, dataMax, body)` | Bắt đầu, chạy `body`, luôn dừng. |
| `traceBegin(kind, index, position, token, tokens = 1)` / `traceEnd()` | Mở và đóng một bước. `kind` là `tsPrefill` hoặc `tsDecode`. Không làm gì khi không có trace. |
| `traceStep(kind, index, position, token): body` | `traceBegin` + `body` + `traceEnd` (đóng cả khi có exception). |

Có thể gọi `traceStart` trước hoặc sau khi dựng model: kernel compile trước đó đã nằm sẵn trong
registry của recorder.

## BybyLang

```bybylang
import nim "std/os"
import nim "../../trace"

traceStart(getTempDir() / "run.fprobe", "decode:2", false, 0)
# ... chạy model ...
traceStop()
```

BybyLang chuyển thẳng các module `import nim` và lời gọi, nên mọi ký hiệu Nim ở trên đều dùng được,
với cùng sự tự do về thứ tự. `test/bybylang_test/trace_test.bybylang` là ví dụ chạy được
(`make bybylang-test`).

## C

```c
nimf_trace_start("run.fprobe", "decode:2", 1, 0);   /* path, steps (NULL = tất cả), data, data_max */
nimf_probe_step(1, 1, pos, pos, token);              /* begin=1, kind 1 = decode, index, position, token */
/* ... dispatch bước ... */
nimf_probe_step(0, 0, 0, 0, 0);                      /* kết thúc bước */
nimf_trace_stop();
```

| Hàm | Trả về |
| --- | --- |
| `int32_t nimf_trace_start(const char* path, const char* steps, int32_t data, int64_t data_max)` | `0`, nếu không `-1` kèm `nimf_last_error()` |
| `int32_t nimf_trace_stop(void)` | `0` (cả khi không có trace nào) |
| `int32_t nimf_trace_active(void)` | `1` hoặc `0` |
| `const char* nimf_trace_file(void)` | Đường dẫn, hợp lệ đến lần gọi kế tiếp |
| `int32_t nimf_probe_step(int32_t begin, int32_t kind, int64_t index, int64_t position, int64_t token)` | `0`, nếu không `-1`. `kind`: `0` prefill, `1` decode. Không làm gì khi không có trace. |

Giống Nim, C có thể bắt đầu trace trước hoặc sau khi compile kernel.

## Biến môi trường

Đặt các biến này trước khi tiến trình chạy tương đương gọi API lúc khởi động, và là cách trace một
chương trình bạn không điều khiển được (ví dụ vòng decode của Nim):

| Biến | Tương đương |
| --- | --- |
| `BYBY_FIR_PROBE=<path>` | `traceStart(path)` khi thư viện được nạp. |
| `BYBY_FIR_PROBE_STEPS=decode:2` | `steps`. |
| `BYBY_FIR_PROBE_DATA=1` | `data=True`. |
| `BYBY_FIR_PROBE_DATA_MAX=<byte>` | `data_max`. |
| `BYBY_ONE_DISPATCH` | Phải bỏ đặt hoặc `0` khi trace: đường one-dispatch là hộp đen, trace sẽ ghi chú rằng không lift được. |

`traceStart` đặt hai biến `DATA` trong suốt thời gian trace và `traceStop` khôi phục lại.

## Đánh dấu bước

Trace chỉ lift được nếu recorder biết mỗi bước token bắt đầu, kết thúc ở đâu và byte nào của buffer là
token và vị trí của bước. Vòng decode của Nim tự ghi các bản ghi này. Driver Python (hoặc C) ghi qua:

- `Session.step(token, position, kind="decode")` / `nimf_probe_step`: mở và đóng một `step`.
- `Session.set_step_inputs(buffer, token_offset, position_offset, token, position)` /
  `nimf_sess_set_step_inputs`: ghi hai số vô hướng và gán vai trò `token` và `position`.

## File trace

File văn bản (`fprobe 2`) gồm `buf` (buffer), `prog` (chương trình), `step`/`end step`, `op` và `disp`
(dispatch cùng chương trình và các binding), `barrier`, bản ghi `role`/`upload`, và các word của chương
trình ở cuối. File được ghi thành bản chụp đầy đủ qua `<path>.tmp` rồi đổi tên, nên lần chạy bị ngắt
vẫn để lại bản chụp đầy đủ trước đó. Chương trình không được ghi sẽ hiện `note kind=missing_words` và
`prog=none` ở dispatch; trace như vậy không lift được.

## Từ trace đến model chạy được

```sh
# 1. ghi hai bước decode (ở đây bằng biến môi trường; `nf.trace` làm đúng như vậy trong code)
BYBY_FIR_PROBE=run.fprobe BYBY_FIR_PROBE_STEPS=decode:2 BYBY_FIR_PROBE_DATA=1 \
    python examples/python/forge_vm.py models/forge-500m-chat --tokens 1,100,200 --new 0
# 2. lift
other/formers.n/firlift/firlift --trace run.fprobe --config model_config.json \
    --out models/forge-500m-chat-fir --report
# 3. kiểm tra không cần trọng số, rồi chạy
other/formers.n/formers-cli --fir-check models/forge-500m-chat-fir/model.fir
other/formers.n/formers-cli --models models/forge-500m-chat-fir -m model.safetensors \
    --prompt-ids 1,100,200 -n 8 --temp 0
```

Tùy chọn của `firlift` và quy tắc lift nằm ở [FIR script lifting](fir-script-lift.md); định dạng file ở
[FIR](fir.vi.md). Model không có `tokenizer.json` dùng được thì chạy bằng `--prompt-ids` và in ra id token.

## Giới hạn

- Mỗi tiến trình một recorder; thư viện tuần tự hóa các lời gọi vào recorder giữa các luồng.
- `steps` là số bước **giữ lại**; các bước vượt quá vẫn chạy nhưng không được ghi.
- Việc sinh chương trình phụ thuộc vị trí hoặc thay đổi shape theo bước cần một quy tắc lowering; lifter
  báo rõ điều nó không làm được thay vì đoán.
