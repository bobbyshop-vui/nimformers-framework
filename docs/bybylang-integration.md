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

## 2. Dynamic Loading via Nim Wrapper

BybyLang DSL does not support direct `dlopen` or library loading. To use a pre-compiled shared library (e.g., `libnimformers.dylib`), you must create a Nim wrapper module that imports the library's symbols and then `import nim` that wrapper into your BybyLang script.

### Step 1: Create a Nim Wrapper (`loader.nim`)

Create a file named `loader.nim` that uses Nim's `importc` and `dynlib` pragmas to link against the compiled library:

```nim
# loader.nim
const libName = "libnimformers.dylib" # or .so / .dll

proc nimf_context_create*(backendKind: int32): pointer {.importc: "nimf_context_create", cdecl, dynlib: libName.}
proc nimf_context_destroy*(handle: pointer): int32 {.importc: "nimf_context_destroy", cdecl, dynlib: libName.}
proc nimf_add*(ctx: pointer, a, b, outData: ptr float32, n: int64): int32 {.importc: "nimf_add", cdecl, dynlib: libName.}
proc nimf_last_error*(): cstring {.importc: "nimf_last_error", cdecl, dynlib: libName.}
```

### Step 2: Import into BybyLang (`script.bybylang`)

Use `import nim` to bring the wrapper's functions into the BybyLang environment:

```bybylang
import nim "./loader"

# Use the functions from the dylib
ctx = nimf_context_create(0)
if ctx == nil:
  print $nimf_last_error()
  quit(1)

A = @[1'f32, 2, 3, 4]
B = @[10'f32, 20, 30, 40]
C = newSeq[float32](4)

if nimf_add(ctx, addr A[0], addr B[0], addr C[0], 4) != 0:
  print $nimf_last_error()
  discard nimf_context_destroy(ctx)
  quit(1)

print C
discard nimf_context_destroy(ctx)
```

---

## 3. Alternative: Source-level Integration

If you prefer to compile the Nim source directly into your AOT binary instead of loading a library file at runtime, you can import the framework's internal API directly:

```bybylang
import nim "../../nimformers_lib"
# ... rest of script ...
```

This approach does not require a pre-compiled `.dylib` file as the code is linked statically during the AOT process.

---

## C ABI Signatures

Refer to [`include/nimformers.h`](../include/nimformers.h) for the complete list of function signatures and their parameter types.
