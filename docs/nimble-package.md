# Install Nimformers as a Nim library

[English](nimble-package.md) | [Tiếng Việt](nimble-package.vi.md)

The Nimble package installs the Nim facade, its source dependencies, the
BybyLang runtime and backend shader files, plus `nimformers_shared` bindings for
loading the C ABI dynamically. It does not include model weights, checkpoints,
datasets, generated binaries, or build caches.

## Install directly from GitHub

You can install the package from this public repository without waiting for it
to be added to the Nimble package index:

```sh
nimble install https://github.com/bobbyshop-vui/nimformers-framework
```

Nimble accepts Git repository URLs as package sources. Once the package is
registered in the official index, the shorter `nimble install nimformers`
command will also work.

## Check out and install

The BybyLang runtime is a Git submodule. Initialize it before building or
installing from a source checkout:

```sh
git clone --recurse-submodules <repository-url> nimformers-framework
cd nimformers-framework
nimble develop
nimble buildShared
```

For an existing checkout that lacks the submodule contents, run
`git submodule update --init --recursive` first. `nimble buildShared` builds
`libnimformers.dylib` (macOS), `libnimformers.so` (Linux), or `nimformers.dll`
(Windows), along with the sibling `nimformers-worker` executable. Model files
remain separate runtime inputs and are not part of the Nimble package.

`nimble develop` makes the checkout's package available to Nimble projects.
After a release is listed in the Nimble package registry, install it in a
consumer project with `nimble install nimformers`. `nimformers.nim` exposes
the native Nim API; the application still needs the C shared library only if
it uses `nimformers_shared`.

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
import bindings/nimformers_shared as nimformers_shared

var ctx = newContext("cpu")
defer: close(ctx)
echo add(ctx, [1'f32, 2], [10'f32, 20])
```

Set `NIMFORMERS_LIB` to the full shared-library path when it is not on the
dynamic loader's search path. The complete example is
[`examples/nim/shared_library.nim`](../examples/nim/shared_library.nim).
The shared binding also exposes `newProcessGroup`, `partition`, `vmRun`,
`generateBatch`, and `generateIdsBatch` over the native parallel C ABI. The
batch methods launch the configured `formers-cli` executable once per prompt;
they return text or space-separated token IDs in prompt order.

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
import nim "bindings/nimformers_shared"

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
export NIMFORMERS_LIB="/absolute/path/to/libnimformers.dylib"
./vendor/bybylang/bybylang examples/bybylang/shared_library.bybylang \
  --aot=/tmp/nimformers-shared-example
/tmp/nimformers-shared-example
```

On Linux use `libnimformers.so`; on Windows use `nimformers.dll`. Make the
installed Nimble package visible to Nim when compiling the generated program,
for example with `BYBYLANG_NIM_FLAGS="--path:/path/to/nimble/package"`. The same
`NIMFORMERS_LIB` environment variable selects the library at runtime.

## Package contents and platform support

`nimble install` installs only the allowlisted Nim modules and BybyLang runtime
sources/assets required by those modules. The GPU kernels' source and compiled
shader files are runtime dependencies, so they are included. Checkpoints,
tokenizers, model downloads, generated executables, caches, and benchmark
artifacts are excluded. Install and build on the target operating system to
produce its native shared library and worker executable.
