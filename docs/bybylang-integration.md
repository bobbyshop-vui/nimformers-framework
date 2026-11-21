# BybyLang and Python Integration Guide

This guide describes how to interface BybyLang scripts and Python code with the compiled `libnimformers` shared library (`libnimformers.dylib` on macOS, `libnimformers.so` on Linux, `nimformers.dll` on Windows).

---

## 1. Build the Shared Library

Before importing or loading the library, you must build the C ABI shared library from the repository root:

```bash
make build-for-python
```

This command generates `libnimformers.dylib` (or the equivalent for your platform) in the project root.

---

## 2. Using Nim Modules in BybyLang (`import nim`)

When writing `.bybylang` scripts, you can import Nim modules directly to access the C ABI. This compiles Nim code Ahead-of-Time (AOT) into your final binary.

Example (`script.bybylang`):

```bybylang
import nim "../../lib_c_api"

# Initialize context (0 = NIMF_BACKEND_AUTO)
ctx = nimf_context_create(0)
if ctx == nil:
  print $nimf_last_error()
  quit(1)

A = @[1'f32, 2, 3, 4]
B = @[10'f32, 20, 30, 40]
C = newSeq[float32](4)

# Call C ABI function
if nimf_add(ctx, addr A[0], addr B[0], addr C[0], 4) != 0:
  print $nimf_last_error()
  discard nimf_context_destroy(ctx)
  quit(1)

print C
discard nimf_context_destroy(ctx)
```

---

## 3. Loading and Using `libnimformers.dylib` in Python

You can load the compiled shared library in Python using the `nimformers` package or directly with `ctypes`.

### Method 1: Using the `nimformers` package

The `nimformers` package attempts to automatically locate and load `libnimformers.dylib` / `.so`:

```python
import numpy as np
from nimformers import Context

# Automatically load libnimformers.dylib / .so and initialize context
ctx = Context(backend="auto")

a = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
b = np.array([10.0, 20.0, 30.0, 40.0], dtype=np.float32)

# Call operations via C ABI wrapper
c = ctx.add(a, b)
print(c)  # Output: [11. 22. 33. 44.]
```

### Method 2: Loading the file directly with `ctypes`

You can manually load the specific shared library file:

```python
import ctypes as ct
from nimformers._lib import load_library

# Load library using helper
lib = load_library("./libnimformers.dylib") # Or libnimformers.so on Linux

# Call C ABI function
ctx = lib.nimf_context_create(0) # 0 = NIMF_BACKEND_AUTO
if not ctx:
    error_msg = lib.nimf_last_error().decode('utf-8')
    print(f"Error: {error_msg}")
    exit(1)

print("Successfully loaded libnimformers.dylib")
lib.nimf_context_destroy(ctx)

# Alternative: Load with raw ctypes
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

## C ABI Signatures

All exported functions in `libnimformers.dylib` / `.so` follow the standard C ABI defined in [`include/nimformers.h`](../include/nimformers.h).
