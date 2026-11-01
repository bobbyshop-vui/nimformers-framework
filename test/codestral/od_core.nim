## od_core.nim — minimal core of "1 dispatch / token" (model independent).
##
## * odCompile  : JITs one VmProgram (may use OP_GRID_BARRIER / OP_BASE_* / OP_VTG_*) -> a Metal kernel name.
## * odDispatch : binds 4 buffers (memF32 | memU8 | lut+meta | inF32 = memF32) + dispatches ntg threadgroups x 256.
##
## Metal only. Shared by onedispatch.nim and t_grid_barrier.nim.

import vendor/bybylang/gpubackend as byby
import session_vm
import session_vm_dispatch
import jit_vm

const
  OdTg* = 256            ## threads per PHYSICAL threadgroup of the persistent kernel (safe on Intel iGPUs)
  OdCtrlFloats* = 64     ## khoi ctrl dau memF32: [0]=cnt [1]=done [2]=err [3]=lastErr (uint32 bit-cast)

var gOdJit* = jit_vm.initJitCache(jit_vm.JitConfig(
  hotThreshold: 0, enableMetalJit: true, enableVulkanJit: false,
  enableCpuJit: false, maxCacheEntries: 16))

proc odCompile*(p: VmProgram; tag: string): string =
  ## Compiles OUTSIDE a session (JIT needs no session). Raises if no real kernel comes out.
  result = jit_vm.dispatchProgramNoSession(gOdJit, p, tag)
  if result.len == 0 or result == "micro_vm_kernel":
    raise newException(CatchableError,
      "odCompile[" & tag & "]: Metal JIT FAILED. Set BYBY_JIT_DUMP_SOURCE=1 to see the source + compile error.")

proc odDispatch*(kname: string; hMem, hW, hLut: byby.SessionHandle;
                 ntg, tgBytes: int) =
  ## Encodes EXACTLY one compute dispatch into the open session.
  ## Metal slots: 1 = memF32 (rw), 2 = memU8 (fp4 weights), 3 = lut/meta/lora (ro), 4 = inF32 (alias of memF32).
  if hMem.kind == byby.gbVulkan:
    raise newException(CatchableError, "odDispatch: Metal only (no grid barrier for Vulkan yet)")
  if not sessionEncPipeline(false, kname, ""):
    raise newException(CatchableError, "odDispatch: cannot bind pipeline " & kname)
  discard sessionEncBindVkOrMetal(false, 1, hMem.metalHandle, hMem.vulkanHandle, 0)
  discard sessionEncBindVkOrMetal(false, 2, hW.metalHandle,   hW.vulkanHandle,   0)
  discard sessionEncBindVkOrMetal(false, 3, hLut.metalHandle, hLut.vulkanHandle, 0)
  discard sessionEncBindVkOrMetal(false, 4, hMem.metalHandle, hMem.vulkanHandle, 0)
  discard sessionEncTgBytes(false, 0, max(tgBytes, 16))
  let maxTG = sessionEncMaxTg(false)
  if maxTG > 0 and maxTG < OdTg:
    raise newException(CatchableError,
      "odDispatch: the pipeline allows only " & $maxTG & " threads/threadgroup < " & $OdTg)
  if not sessionEncDispatch(false, ntg * OdTg, OdTg, tgBytes):
    raise newException(CatchableError, "odDispatch: sessionEncDispatch failed")

proc odStage*(p: var VmProgram; body: VmProgram; mode, nVirt, param: uint32) =
  ## Wraps `body` (one VmProgram written for its own kernel, ending with OP_HALT) in an
  ## OP_VTG_BEGIN/END block. mode 0: V=256 (virtual tgid = vIt); mode 1: V=32 (8 virtual TGs / physical TG,
  ## param = tgF32 stride per sub, in floats); mode 2: V=256 + guard tid < param.
  ## The body must NOT have a HALT in the middle; the final HALT is dropped.
  if body.words.len == 0 or body.words[body.words.len - 1] != OP_HALT:
    raise newException(ValueError, "odStage: the program must end with OP_HALT")
  p.emitVtgBegin(mode, nVirt, param)
  for i in 0 ..< body.words.len - 1:
    p.emit body.words[i]
  p.emitVtgEnd()
