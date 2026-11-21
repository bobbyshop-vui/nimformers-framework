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

## 2. Using Shared Library in BybyLang

To use the compiled shared library (`libnimformers.dylib` / `.so`) in BybyLang, you should import the `nimformers` module. This module is configured to automatically detect and link the pre-compiled shared library at runtime.

### Example (`script.bybylang`)

```bybylang
import nim "nimformers"

# The module 'nimformers' automatically handles loading the compiled
# shared library (dylib/.so) and provides access to the C ABI.

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

This approach allows BybyLang to dynamically link against the shared library.

---

## 3. Alternative: Source-level Integration

If you prefer to compile the Nim source directly into your AOT binary instead of loading a library file at runtime, you can import the framework's C API implementation directly:

```bybylang
import nim "../../nimformers_lib"
# ... rest of script ...
```

This approach links the symbols statically during the AOT process, without requiring a pre-compiled library file at runtime.

---

## C ABI Signatures

Refer to [`include/nimformers.h`](../include/nimformers.h) for the complete list of function signatures and their parameter types.
