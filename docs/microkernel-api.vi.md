
# API micro-VM và JIT

[English](microkernel-api.md) | [Tiếng Việt](microkernel-api.vi.md)

Bytecode micro-VM nằm trong `vendor/bybylang/micro_kernel/session_vm.nim`;
encode GPU ở `session_vm_dispatch.nim`; lower sang Metal/Vulkan và cache
JIT ở `jit_vm.nim`. Các file cùng tên ở thư mục gốc là lớp tương thích.


## Chương trình và hàm phát instruction

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

Các hàm điều khiển luồng/threadgroup bổ sung: `emitForBegin`, `emitForEnd`,
`emitIfBegin`, `emitIfEnd`, `emitBarrierTg`, `emitTgLoadF32`, `emitTgStoreF32`,
`emitLoadLid`, `emitLoadTgid`.

Các hàm số nguyên/địa chỉ: `emitUTgid`, `emitULid`, `emitUConst`, `emitUAdd`,
`emitUMul`, `emitUShr`, `emitUAddK`, `emitUMulK`, `emitUAndK`, `emitUShrK`,
`emitUShlK`, `emitULoadU8`, `emitULoadU32`, `emitULoadConstF`, `emitULoadIn`,
`emitUStoreF32`, `emitULoadF32`, `emitUAtomicAdd`, `emitUToR`, `emitRToU`.

Nhóm JIT địa chỉ/thread ảo nâng cao gồm `emitGridBarrier`, `emitGridEpilogue`,
`emitBaseF32`, `emitBaseU8`, `emitBaseLut`, `emitBaseIn`, `emitULoadMeta`,
`emitVtgBegin`, `emitVtgEnd`, `emitLoadVact`. Một số opcode chỉ có ở một số
backend; đọc chú thích `OP_*` trong `session_vm.nim` trước khi dùng.


## Lưu bytecode và lower source

```nim
proc saveBvm*(p: VmProgram, path: string)
proc loadBvm*(path: string): VmProgram
proc lowerBytecodeToMetal*(p: VmProgram, kernelName: string): string
proc lowerBytecodeToGlsl*(p: VmProgram, kernelName: string): string
```

`lowerBytecodeToMetal` và `lowerBytecodeToGlsl` tạo source kernel, chưa chạy
kernel. Dùng encoder trong [API session GPU](gpu-session-api.vi.md) để đặt
pipeline, buffer và hình học dispatch.


## Cache JIT

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


## Hỗ trợ DSL

Trong `.bybylang`, dùng block
`micro kernel NAME INPUT -> OUTPUT threads N tg N out N:` với mỗi dòng
thụt vào là một lệnh `emit...`. Compiler tạo `VmProgram`, lower sang backend
đã chọn, dispatch trong session và trả vector output. Cú pháp cùng ví dụ ở
[API BybyLang GPU và DSL](bybylang-api.vi.md) và
`vendor/bybylang/demo/demo_gpu.bybylang`.


## Giới hạn thực thi

- GPU execution cần thiết bị Metal/Vulkan và hỗ trợ session tương ứng.
- Kích thước threadgroup phải dương và không lớn hơn tổng số thread truyền
  cho helper DSL hiện tại.
- Đồng bộ toàn grid khác barrier workgroup thông thường. `emitGridBarrier`
  dùng giao thức spin; chỉ dùng khi geometry bảo đảm mọi group tham gia có
  thể cùng resident. Với phụ thuộc toàn cục, ưu tiên dispatch thứ hai.
- Không phải backend nào cũng hỗ trợ mọi opcode. Kiểm tra MSL/SPIR-V sinh
  ra và chạy trên backend thực tế trước khi dùng chính thức.
