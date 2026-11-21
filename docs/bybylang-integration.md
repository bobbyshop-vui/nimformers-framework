# Hướng dẫn tích hợp BybyLang và Python với Dynamic Library (`libnimformers.dylib` / `.so`)

Tài liệu này hướng dẫn cách sử dụng thư viện động đã được biên dịch (`libnimformers.dylib` trên macOS, `libnimformers.so` trên Linux, `nimformers.dll` trên Windows) với BybyLang và Python.

---

## 1. Biên dịch Thư viện Động (Shared Library)

Trước khi import hoặc load `dylib`/`so`, bạn cần biên dịch thư viện C ABI từ thư mục gốc của repository:

```bash
make build-for-python
```

Lệnh này sẽ tạo ra file `libnimformers.dylib` (hoặc `libnimformers.so` / `nimformers.dll`) trong thư mục gốc dự án.

---

## 2. Dùng Nim Modules trong BybyLang (`import nim`)

Khi viết script `.bybylang`, bạn có thể import trực tiếp module Nim `lib_c_api` để gọi C ABI. Cách này biên dịch AOT code Nim thẳng vào file thực thi.

Ví dụ (`script.bybylang`):

```bybylang
import nim "../../lib_c_api"

# Khởi tạo context (0 = NIMF_BACKEND_AUTO)
ctx = nimf_context_create(0)
if ctx == nil:
  print $nimf_last_error()
  quit(1)

A = @[1'f32, 2, 3, 4]
B = @[10'f32, 20, 30, 40]
C = newSeq[float32](4)

# Gọi hàm C ABI
if nimf_add(ctx, addr A[0], addr B[0], addr C[0], 4) != 0:
  print $nimf_last_error()
  discard nimf_context_destroy(ctx)
  quit(1)

print C
discard nimf_context_destroy(ctx)
```

---

## 3. Import và Sử dụng `libnimformers.dylib` trong Python

Nếu bạn muốn load file `dylib` / `so` được biên dịch để sử dụng trong Python (thông qua Python binding hoặc `ctypes`), có 2 cách thực hiện:

### Cách 1: Sử dụng package `nimformers` (Khuyên dùng)

Package `nimformers` sẽ tự động tìm kiếm và load `libnimformers.dylib` / `.so`:

```python
import numpy as np
from nimformers import Context

# Tự động load libnimformers.dylib / .so và khởi tạo context
ctx = Context(backend="auto")

a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

# Gọi phép toán qua C ABI được wrap sẵn
c = ctx.add(a, b)
print(c)  # Output: [11. 22. 33. 44.]
```

### Cách 2: Load trực tiếp file `.dylib` / `.so` bằng `ctypes` hoặc `nimformers._lib`

Nếu bạn muốn chỉ định chính xác đường dẫn tới file `.dylib`:

```python
import ctypes as ct
from nimformers._lib import load_library, get_library

# Cách A: Load thư viện bằng helper của nimformers với đường dẫn dylib cụ thể
lib = load_library("./libnimformers.dylib") # Hoặc libnimformers.so trên Linux

# Gọi trực tiếp hàm C ABI
ctx = lib.nimf_context_create(0) # 0 = NIMF_BACKEND_AUTO
if not ctx:
    error_msg = lib.nimf_last_error().decode('utf-8')
    print(f"Error: {error_msg}")
    exit(1)

print("Successfully loaded libnimformers.dylib!")
lib.nimf_context_destroy(ctx)

# Cách B: Load trực tiếp bằng ctypes thuần của Python
dylib_path = "./libnimformers.dylib"
raw_lib = ct.CDLL(dylib_path)

raw_lib.nimf_context_create.argtypes = [ct.c_int]
raw_lib.nimf_context_create.restype = ct.c_void_p

raw_lib.nimf_context_destroy.argtypes = [ct.c_void_p]
raw_lib.nimf_context_destroy.restype = ct.c_int

ctx_raw = raw_lib.nimf_context_create(0)
print("Raw ctypes load success:", ctx_raw)
raw_lib.nimf_context_destroy(ctx_raw)
```

---

## Tham khảo C ABI Signatures

Tất cả các hàm được export trong `libnimformers.dylib` / `.so` tuân theo chuẩn C ABI được định nghĩa tại [`include/nimformers.h`](../include/nimformers.h).
