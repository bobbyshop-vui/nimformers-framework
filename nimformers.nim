## nimformers.nim - THE SINGLE ENTRY POINT of the framework.
##
## BEFORE: every file using the framework had to list all 5 separate imports
## (the order MATTERS because of cross dependencies: backend <- bybylang,
## nimformer <- backend + quant + customfloat, tensor_ops <- backend + nimformer):
##
##   import nimformer
##   import backend
##   import customfloat
##   import quant
##   import tensor_ops
##
## Forgetting one of the 5 (or the right `export`) gave an obscure "undefined identifier"
## error, hard to track down without knowing the code. NOW: just
##
##   import nimformers
##
## gives all the Tensor/Backend/QuantTensor/BybyLang exports (sessionMatmulQ4,
## GpuBackend, etc. - the whole export list already in backend.nim),
## every model-building function of nimformer.nim/customfloat.nim/quant.nim, AND
## the extra tensor ops (view/permute/reduction/concat/split/stack/
## dropout/comparison/cumsum/cumprod/scan - tensor_ops.nim, used a lot for
## Mamba/SSM).
##
## The 5 original files are NOT changed/removed (nimformer.nim/backend.nim/customfloat.nim/
## quant.nim/tensor_ops.nim) - the old files importing them separately
## (nim_inference.nim, finetune_lora.nim, graft_attention.nim, etc.) STILL
## work exactly the same, switching to "import nimformers" is not required.
## This is an ADDITIONAL facade, not a mandatory replacement.
##
## The "if/else backend choice" already lives in backend.nim (resolveBackendKind,
## detectBackend through BybyLang) - nimformers.nim does NOT repeat that logic,
## it only re-exports it so users can call newBackend() right after
## `import nimformers` without knowing backend.nim exists separately.
##
## NOTE ON THE KERNEL FUNCTIONS (beMatmul/beMatmulQ4/beAdd/beExp/beSoftmax/
## beAttentionFused/beResidentUploadQ4/... - 43 public procs at the time
## of writing): they need NOT be listed by hand in this file. `export backend`
## (the line below) re-exports the WHOLE public API of backend.nim
## following Nim's standard `export <imported module>` semantics - including
## the `export byby.X, byby.Y, ...` block (re-exported BybyLang API) AND
## every `beXxx*` proc defined in backend.nim itself, not just
## some subset. A new kernel function added to backend.nim (marked `*`)
## is exposed through `import nimformers` automatically, no change needed here.

import nimformer
import backend
import customfloat
import quant
import tensor_ops
import trace

export trace
export nimformer
export backend
export customfloat
export quant
export tensor_ops

# backend.nim re-exports part of the API of vendor/bybylang/gpubackend.
# Nim does not pass those imported-and-reexported symbols on through
# `export backend`, so the facade must expose them explicitly.
export backend.GpuBackend, backend.SessionHandle, backend.ResidentWeightHandle
export backend.sessionBegin, backend.sessionUpload, backend.sessionUploadIndices,
       backend.sessionRope, backend.sessionAllocScratch, backend.sessionMatmul,
       backend.sessionMatmulNT, backend.sessionSoftmax, backend.sessionVecOp,
       backend.sessionActivation, backend.sessionEnd, backend.sessionAllocHeap,
       backend.sessionAllocFromHeap, backend.sessionFreeHeap, backend.sessionRead,
       backend.sessionFree, backend.sessionCopyInto, backend.sessionScale, backend.residentUploadQ4,
       backend.sessionMatmulQ4, backend.residentFreeQ4, backend.sessionLayernormEnc,
       backend.sessionRmsnormEnc, backend.sessionAttentionFusedEnc,
       backend.sessionAttentionDecodeGqaEnc, backend.cpuAttentionDecodeGqa,
       backend.gpuGather, backend.gpuArgmaxIndex, backend.gpuCompare,
       backend.gpuDropoutMask, backend.gpuPermute, backend.gpuSplitSlice,
       backend.gpuConcat

# ── micro-VM / JIT (vendor/bybylang/micro_kernel) ──────────────────────────
# session_vm + session_vm_dispatch + jit_vm moved from the root into
# vendor/bybylang/micro_kernel/. At app level `import nimformers` is enough to
# get VmProgram / encodeVm / the JIT cache; the 3 old files are not imported directly.
# (The root session_vm.nim / jit_vm.nim / session_vm_dispatch.nim stay as
# backward-compatible shims if old code still does `import session_vm`.)
import vendor/bybylang/micro_kernel
export micro_kernel
