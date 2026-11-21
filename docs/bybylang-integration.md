# BybyLang and Dynamic Library Integration

This guide describes how to interface BybyLang scripts with the compiled `libnimformers` shared library (`libnimformers.dylib` on macOS, `libnimformers.so` on Linux, `nimformers.dll` on Windows).

---

## 1. Build the Shared Library

The C ABI shared library must be built from the repository root:

```bash
make build-for-python
```

This command generates the shared library file in the project root.

---

## 2. Using Nim Modules in BybyLang (`import nim`)

BybyLang scripts can access the C ABI by importing Nim modules. This compiles the Nim code Ahead-of-Time (AOT) into the final binary.

Example of calling C ABI functions (`script.bybylang`):

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

This approach links the symbols directly during the AOT compilation process, rather than using dynamic FFI at runtime.

---

## 3. Dynamic Loading of Shared Libraries in BybyLang

BybyLang does not support dynamic FFI (e.g., runtime `dlopen` of arbitrary `.dylib` or `.so` files) within its DSL. To interface with a specific compiled library file at runtime, the library must be imported via a Nim module (`import nim`) that implements the required FFI wrappers in Nim.

The `lib_c_api.nim` module serves as the primary interface for this purpose. 

If additional functionality from a custom dynamic library is required, create a Nim wrapper module that loads the library and export its procedures, then import that Nim module into BybyLang.

---

## C ABI Signatures

All exported functions in the shared library follow the standard C ABI defined in [`include/nimformers.h`](../include/nimformers.h).
