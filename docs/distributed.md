# Device-parallel execution

[English](distributed.md) | [Tiếng Việt](distributed.vi.md)

The BybyLang native scheduler assigns work to child processes on one host.
Nim and `.bybylang` programs can import it through `nimformers`. The Python
package calls the same scheduler through the shared library. The scheduler
can run one child per available device at a time on macOS and Linux.

## Pick devices and split the work

Use `cpu`, `vulkan[:ordinal]`, or `metal[:ordinal]`. GPU names without an
ordinal select device 0. Ordinals are backend-specific. Use one entry for
each physical device you intend to use.

```nim
import nimformers

let group = initProcessGroup(["cpu", "vulkan:0"], [1.0, 8.0])
echo group.partition(90)  # @[10, 80]
```

In a BybyLang script, import the same Nim module.

```bybylang
import nim "nimformers"

group = nimformers.initProcessGroup(@["cpu", "vulkan:0"], @[1.0, 8.0])
print nimformers.partition(group, 90)
```

The native `runCommands` API assigns
independent commands to contiguous slices of the input and returns their
stdout and stderr in order. Use `runShards` to build one native command per
nonempty slice. The scheduler sets each child's device environment before
the child initializes its backend.

Configured ratios stay fixed. A call can override them and can set a ratio
to zero to leave a device idle. Automatic mode starts at 8:1 GPU to CPU and
adjusts subsequent calls using per-workload measurements on the group. It
caps the CPU weight at half the smallest GPU weight. `partition` previews
counts using explicit, configured or initial weights. Rounding can leave
some devices idle for short batches.

## Run a micro-kernel batch

```nim
import nimformers

let group = initProcessGroup(["cpu"])
var prog: VmProgram
prog.emitLoadF32(1, 0, 0)
prog.emitAdd(1, 1, 1)
prog.emitStoreF32(1, 0, 0)
prog.emitHalt()
let output = runVmParallel(group, prog.words, @[1'f32, 2, 3, 4],
                           itemSize = 1, threadsPerItem = 1, tgSize = 1)
echo output  # @[2.0, 4.0, 6.0, 8.0]
```

`make build-shared` or `nimble buildShared` builds `nimformers-worker` and
the shared library. Put the worker on PATH, next to the executable or in
the current directory. Set `NIMFORMERS_WORKER` or pass `workerPath` to pick
a specific binary.

The VM sees a separate contiguous buffer on each device. Indices start at
zero within each slice, so use independent items or threadgroups. When
`tgSize > 0`, the number of items must be divisible by
`tgSize / gcd(tgSize, threadsPerItem)`. This preserves explicit threadgroup
boundaries. With `tgSize = 0`, choose programs that are independent of
the backend's threadgroup geometry.

Python exposes the same launch through `group.vm_run(program, memory,
threads_per_item=None, tg_floats=0, tg_size=0, ratios=None)`. The first
axis is the item axis. `threads_per_item` defaults to the number of float32
elements in one item. The method returns a new array in the original order.
Use `init_process_group(devices, ratios=None)` from `nimformers.distributed`.

## Run independent FIR prompts

For Nim or BybyLang, call `runFormersBatch` with a group, the `formers-cli`
executable, CLI model arguments, and prompts. Python's
`formers.Model.generate_batch(..., group=group)` and `generate_ids_batch`
use that same native entry point. See the [formers.n Python API](../other/formers.n/README.md#python-api).

In Nim, `runFormersBatch` returns each prompt's stdout in input order.
`runFormersBatchDetailed` returns `ParallelResult` values with both `stdout`
and `stderr`, also in input order, so a CLI or application can display
per-worker logs and throughput when needed.

The scheduler runs the full inference for each prompt on one device and
returns results in prompt order. Include `--fir model.fir` and the model's
config, weights, and tokenizer options for a FIR graph. Each participating
device needs memory for the model and must support its kernels. Codestral
resident FIR currently needs Vulkan. This path distributes independent
prompts and independent VM data across devices on one machine.

### Submit a batch with `formers-cli`

Repeat `-p` or `-f` to send multiple prompts through the native scheduler:

```sh
other/formers.n/formers-cli --models test/mamba3/models \
  -m test/mamba3/models/mamba3_siso_187m.nimq \
  --config test/mamba3/models/mamba3-siso-187m/config.json \
  --tokenizer test/mamba3/models/mamba3-siso-187m --no-lora \
  --distributed cpu,metal:0 --distributed-ratios 1,1 \
  -p 'Write one sentence about the ocean.' \
  -p 'Explain why the sky appears blue.' \
  -n 24 --temp 0 --repeat-penalty 1.0 --no-progress
```

The CLI prints each worker's prompt count and returns outputs in input order.
For token comparison, run each prompt directly with `--backend cpu` or
`--backend metal`, using the same model and generation settings. Each worker
loads the complete model; this API distributes independent prompt batches.
