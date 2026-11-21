# Install Nimformers as a Nim library

[English](nimble-package.md) | [Tiếng Việt](nimble-package.vi.md)

The Nimble package installs the Nim facade and its source dependencies, the
BybyLang runtime and backend shader files, and a target-built C ABI shared
library with its parallel worker. It does not include model weights,
checkpoints, datasets, or build caches.

## Install directly from GitHub

You can install the package from this public repository without waiting for it
to be added to the Nimble package index:

```sh
nimble install https://github.com/bobbyshop-vui/nimformers-framework
```

Nimble accepts Git repository URLs as package sources. Once the package is
registered in the official index, the shorter `nimble install nimformers`
command will also work.

During installation Nimble builds the shared library and worker for the target
machine and installs both under the package's `lib/` directory. There is no
separate build command in the install flow.

## Install from a checkout

If you already cloned the repository, install everything with one Nimble
command. The install hook builds the platform shared library and worker before
Nimble copies the package:

```sh
git clone --recurse-submodules <repository-url> nimformers-framework
cd nimformers-framework
nimble install
```

The direct GitHub command in the previous section does the same end-to-end
installation. The shared library is installed as `lib/libnimformers.dylib`
(macOS), `lib/libnimformers.so` (Linux), or `lib/nimformers.dll` (Windows),
along with `lib/nimformers-worker`. Model files remain separate runtime inputs.
`nimformers.nim` exposes the source-level Nim API; use `nimformers_shared` to
call the installed C shared library without compiling the framework
implementation into the application.

## Use the Nim facade or dynamic bindings

Static Nim API:

```nim
import nimformers

let group = initProcessGroup(["cpu"])
echo group.partition(12)
```

The process-group API is provided by `nimformers.nim`; see the [distributed
execution reference](distributed.md) for device selection and worker behavior.
The runnable minimal example is
[`examples/nim/native_process_group.nim`](../examples/nim/native_process_group.nim).

`nimformers_shared` uses the same C ABI v2 library from Nim without linking its
implementation into the application:

```nim
import nimformers_shared

var ctx = newContext("cpu")
defer: close(ctx)
echo "shared library: ", findLibrary()
echo add(ctx, [1'f32, 2], [10'f32, 20])
```

`findLibrary()` returns the full path from the installed package. Set
`NIMFORMERS_LIB` to override automatic discovery. The wrapper also resolves the
parallel worker beside the library. The complete example is
[`examples/nim/shared_library.nim`](../examples/nim/shared_library.nim).
The shared binding also exposes `newProcessGroup`, `partition`, `vmRun`,
`generateBatch`, and `generateIdsBatch` over the native parallel C ABI. The
batch methods launch the configured `formers-cli` executable once per prompt;
they return text or space-separated token IDs in prompt order.
All high-level model `forward` APIs also support customizable before/after
hooks; see the [forward hook reference](forward-hooks.md).

```nim
var group = newProcessGroup(["cpu", "vulkan:0"])
defer: close(group)
let outputs = generateBatch(group, "/path/to/formers-cli",
  ["--models", "/models"], ["Hello", "Write a short poem"], maxTokens = 16)
```

## Use the bindings from BybyLang

BybyLang imports the Nim facade into its generated AOT program:

```bybylang
import nim "nimformers"

let group = initProcessGroup(["cpu"])
echo group.partition(12)
```

BybyLang can also load the shared-library bindings with `nimformers_shared`:

```bybylang
import nim "nimformers_shared"

ctx = nimformers_shared.newContext("cpu")
group = nimformers_shared.newProcessGroup(@["cpu"])
print nimformers_shared.backendName(ctx)
print nimformers_shared.add(ctx, @[1'f32, 2, 3], @[10'f32, 20, 30])
print nimformers_shared.partition(group, 12)
nimformers_shared.close(group)
nimformers_shared.close(ctx)
```

Build the BybyLang compiler from the initialized submodule, then compile and
run the script:

```sh
make -C vendor/bybylang build
./vendor/bybylang/bybylang examples/bybylang/shared_library.bybylang \
  --aot=/tmp/nimformers-shared-example
/tmp/nimformers-shared-example
```

On Linux use `libnimformers.so`; on Windows use `nimformers.dll`. Make the
installed Nimble package visible to Nim when compiling the generated program,
for example with `BYBYLANG_NIM_FLAGS="--path:/path/to/nimble/package"`.
`NIMFORMERS_LIB` overrides automatic library discovery at runtime.

## Package contents and platform support

`nimble install` installs the allowlisted Nim modules and BybyLang runtime
sources/assets, plus the shared library and worker built for the target OS.
The GPU kernels' source and compiled shader files are runtime dependencies, so
they are included. Checkpoints, tokenizers, model downloads, build caches, and
benchmark artifacts are excluded.
