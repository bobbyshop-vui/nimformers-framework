
# Micro-VM and JIT API

[English](microkernel-api.md) | [Tiếng Việt](microkernel-api.vi.md)

The micro-VM bytecode API lives in `vendor/bybylang/micro_kernel/session_vm.nim`; GPU encoding is in `session_vm_dispatch.nim`; Metal/Vulkan source lowering and the JIT cache live in `jit_vm.nim`. The root-level `session_vm.nim`, `session_vm_dispatch.nim`, and `jit_vm.nim` are compatibility shims.


## Program and instruction emitters

```nim
type VmProgram* = object
  words*: seq[uint32]

proc emit*(p: var VmProgram, word: uint32)
proc emitNop*(p: var VmProgram)
proc emitHalt*(p: var VmProgram)
proc emitLoadF32*(p: var VmProgram, rd, rb, offset: uint32)
proc emitStoreF32*(p: var VmProgram, rs, rb, offset: uint32)
proc emitLoadU8*(p: var VmProgram, rd, rb, offset: uint32)
proc emitLoadConstF32*(p: var VmProgram, rd: uint32, v: float32)
proc emitAdd*(p: var VmProgram, rd, ra, rb: uint32)
proc emitSub*(p: var VmProgram, rd, ra, rb: uint32)
proc emitMul*(p: var VmProgram, rd, ra, rb: uint32)
proc emitDiv*(p: var VmProgram, rd, ra, rb: uint32)
proc emitFma*(p: var VmProgram, rd, ra, rb, rc: uint32)
proc emitExp*(p: var VmProgram, rd, rb: uint32)
proc emitNeg*(p: var VmProgram, rd, rb: uint32)
proc emitSigmoid*(p: var VmProgram, rd, rb: uint32)
proc emitSoftplus*(p: var VmProgram, rd, rb: uint32)
proc emitRsqrt*(p: var VmProgram, rd, rb: uint32)
proc emitAddConst*(p: var VmProgram, rd, rb: uint32, v: float32)
proc emitMulConst*(p: var VmProgram, rd, rb: uint32, v: float32)
```

Additional control-flow/threadgroup emitters: `emitForBegin`, `emitForEnd`, `emitIfBegin`, `emitIfEnd`, `emitBarrierTg`, `emitTgLoadF32`, `emitTgStoreF32`, `emitLoadLid`, and `emitLoadTgid`.

Integer/address emitters: `emitUTgid`, `emitULid`, `emitUConst`, `emitUAdd`, `emitUMul`, `emitUShr`, `emitUAddK`, `emitUMulK`, `emitUAndK`, `emitUShrK`, `emitUShlK`, `emitULoadU8`, `emitULoadU32`, `emitULoadConstF`, `emitULoadIn`, `emitUStoreF32`, `emitULoadF32`, `emitUAtomicAdd`, `emitUToR`, and `emitRToU`.

Advanced JIT address/virtual-thread emitters include `emitGridBarrier`, `emitGridEpilogue`, `emitBaseF32`, `emitBaseU8`, `emitBaseLut`, `emitBaseIn`, `emitULoadMeta`, `emitVtgBegin`, `emitVtgEnd`, and `emitLoadVact`. Some opcodes are backend-specific; inspect `OP_*` comments in `session_vm.nim` before using them.


## Bytecode persistence and source lowering

```nim
proc saveBvm*(p: VmProgram, path: string)
proc loadBvm*(path: string): VmProgram
proc lowerBytecodeToMetal*(p: VmProgram, kernelName: string): string
proc lowerBytecodeToGlsl*(p: VmProgram, kernelName: string): string
```

`lowerBytecodeToMetal` and `lowerBytecodeToGlsl` produce kernel source; they do not dispatch it. Use the session encoder APIs in [GPU Session API](gpu-session-api.md) to set a pipeline, buffers, and dispatch geometry.


## JIT cache

```nim
type JitConfig* = object
type JitCache* = object
type CompiledEntry* = object

proc defaultJitConfig*(): JitConfig
proc initJitCache*(cfg = defaultJitConfig()): JitCache
proc ensureJitDeviceReady*(): bool
proc programKey*(p: VmProgram, tag = ""): string
proc programKeyNoSession*(p: VmProgram, tag = ""): string
proc maybeJit*(cache: var JitCache, p: VmProgram, key: string): CompiledEntry
proc dispatchProgramTagged*(cache: var JitCache, p: VmProgram, tag: string): string
proc dispatchProgramNoSession*(cache: var JitCache, p: VmProgram, tag = ""): string
proc dispatchProgram*(cache: var JitCache, p: VmProgram, tag = ""): string
proc kernelNameFor*(cache: var JitCache, p: VmProgram, tag: string): string
proc glslSourceFor*(cache: var JitCache, p: VmProgram, tag: string): string
proc cachedGlslSource*(cache: var JitCache, kernelName: string): string
proc mergePrograms*(parts: openArray[VmProgram]): VmProgram
```


## DSL support

For a kernel declared in `.bybylang`, use a `micro kernel NAME INPUT -> OUTPUT threads N tg N out N:` block with one `emit...` instruction per indented line. The compiler constructs a `VmProgram`, lowers it to the selected backend, dispatches inside a session, and returns the output vector. Full syntax and a working example are in [BybyLang GPU API and DSL](bybylang-api.md) and `vendor/bybylang/demo/demo_gpu.bybylang`.


## Execution limits

- A Metal or Vulkan device and corresponding session support are required for GPU execution.
- Threadgroup size must be positive and no larger than the total thread count passed to the current DSL helper.
- Grid-wide synchronization is not a normal workgroup barrier. `emitGridBarrier` uses a spin protocol and must only be used when the launch geometry guarantees all participating groups can be resident; prefer a second dispatch for a global dependency.
- Not every VM opcode is supported by every backend. Validate generated MSL/SPIR-V and run it on the actual backend before relying on it.
