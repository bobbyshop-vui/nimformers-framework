## od_core.nim — loi toi thieu cua "1 dispatch / token" (khong phu thuoc model).
##
## * odCompile  : JIT 1 VmProgram (co the dung OP_GRID_BARRIER / OP_BASE_* / OP_VTG_*) -> ten kernel Metal.
## * odDispatch : bind 4 buffer (memF32 | memU8 | lut+meta | inF32 = memF32) + dispatch ntg threadgroup x 256.
##
## Chi Metal. Dung chung boi onedispatch.nim va t_grid_barrier.nim.

import vendor/bybylang/gpubackend as byby
import session_vm
import session_vm_dispatch
import jit_vm

const
  OdTg* = 256            ## so thread / threadgroup VAT LY cua kernel persistent (an toan tren Intel iGPU)
  OdCtrlFloats* = 64     ## khoi ctrl dau memF32: [0]=cnt [1]=done [2]=err [3]=lastErr (uint32 bit-cast)

var gOdJit* = jit_vm.initJitCache(jit_vm.JitConfig(
  hotThreshold: 0, enableMetalJit: true, enableVulkanJit: false,
  enableCpuJit: false, maxCacheEntries: 16))

proc odCompile*(p: VmProgram; tag: string): string =
  ## Compile NGOAI session (JIT khong can session). Raise neu khong ra duoc kernel that.
  result = jit_vm.dispatchProgramNoSession(gOdJit, p, tag)
  if result.len == 0 or result == "micro_vm_kernel":
    raise newException(CatchableError,
      "odCompile[" & tag & "]: JIT Metal THAT BAI. Bat BYBY_JIT_DUMP_SOURCE=1 de xem source + loi compile.")

proc odDispatch*(kname: string; hMem, hW, hLut: byby.SessionHandle;
                 ntg, tgBytes: int) =
  ## Encode DUNG 1 compute dispatch vao session dang mo.
  ## Slot Metal: 1 = memF32 (rw), 2 = memU8 (weight fp4), 3 = lut/meta/lora (ro), 4 = inF32 (alias memF32).
  if hMem.kind == byby.gbVulkan:
    raise newException(CatchableError, "odDispatch: chi ho tro Metal (grid barrier chua co cho Vulkan)")
  if not sessionEncPipeline(false, kname, ""):
    raise newException(CatchableError, "odDispatch: khong bind duoc pipeline " & kname)
  discard sessionEncBindVkOrMetal(false, 1, hMem.metalHandle, hMem.vulkanHandle, 0)
  discard sessionEncBindVkOrMetal(false, 2, hW.metalHandle,   hW.vulkanHandle,   0)
  discard sessionEncBindVkOrMetal(false, 3, hLut.metalHandle, hLut.vulkanHandle, 0)
  discard sessionEncBindVkOrMetal(false, 4, hMem.metalHandle, hMem.vulkanHandle, 0)
  discard sessionEncTgBytes(false, 0, max(tgBytes, 16))
  let maxTG = sessionEncMaxTg(false)
  if maxTG > 0 and maxTG < OdTg:
    raise newException(CatchableError,
      "odDispatch: pipeline chi cho " & $maxTG & " thread/threadgroup < " & $OdTg)
  if not sessionEncDispatch(false, ntg * OdTg, OdTg, tgBytes):
    raise newException(CatchableError, "odDispatch: sessionEncDispatch that bai")

proc odStage*(p: var VmProgram; body: VmProgram; mode, nVirt, param: uint32) =
  ## Boc `body` (1 VmProgram viet cho kernel rieng, ket thuc bang OP_HALT) vao khoi
  ## OP_VTG_BEGIN/END. mode 0: V=256 (tgid ao = vIt); mode 1: V=32 (8 TG ao / TG vat ly,
  ## param = stride tgF32 moi sub, floats); mode 2: V=256 + guard tid < param.
  ## Than KHONG duoc co HALT giua chung; HALT cuoi bi bo.
  if body.words.len == 0 or body.words[body.words.len - 1] != OP_HALT:
    raise newException(ValueError, "odStage: program phai ket thuc bang OP_HALT")
  p.emitVtgBegin(mode, nVirt, param)
  for i in 0 ..< body.words.len - 1:
    p.emit body.words[i]
  p.emitVtgEnd()
