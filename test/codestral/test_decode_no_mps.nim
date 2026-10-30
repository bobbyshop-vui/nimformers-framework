## MPS must reject M=1 at both Nim and native entry points; M>1 still works.
import std/math
import backend
import vendor/bybylang/backends/metal/metal_backend as metal

proc mpsDirect(a,b,c:ptr float32; m,k,n:cint; error:ptr cstring):cint
  {.importc:"metal_matmul_mps",header:"metal_shim.h".}
proc mpsNtDirect(a,b,c:pointer; m,k,n,stride,offset:cint):cint
  {.importc:"metal_matmul_mps_nt_enc",header:"metal_shim.h".}

proc main() =
  # Rejection works before GPU initialization, including callers bypassing Nim.
  var error:cstring
  var scalar=1'f32
  doAssert mpsDirect(addr scalar,addr scalar,addr scalar,1,1,1,addr error)==0
  doAssert mpsNtDirect(nil,nil,nil,1,1,1,1,0)==0
  doAssert not metal.metalSessionMatmulMpsNTEnc(nil,nil,nil,1,1,1,1,0)
  let ctx=newBackend("metal")
  let be=ctx.kind.toByby()
  var a=sessionUpload(be,@[1'f32,2,3,4, 5,6,7,8])
  var b=sessionUpload(be,@[1'f32,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1])
  var c=sessionAllocScratch(be,8)
  defer:
    sessionFree(be,a);sessionFree(be,b);sessionFree(be,c)
  doAssert sessionBegin(be)
  doAssert not metal.metalSessionMatmulMpsNTEnc(a.metalHandle,b.metalHandle,c.metalHandle,1,4,4,4,0)
  doAssert mpsNtDirect(a.metalHandle,b.metalHandle,c.metalHandle,1,4,4,4,0)==0
  # Decode fallback retains native, exact FP32 matrix/vector behavior.
  doAssert metal.metalSessionMatmulNTEnc(a.metalHandle,b.metalHandle,c.metalHandle,1,4,4)
  doAssert sessionEnd(be)
  let decoded=sessionRead(be,c,2)
  doAssert abs(decoded[0]-1)<1e-6 and abs(decoded[1]-2)<1e-6
  doAssert sessionBegin(be)
  doAssert metal.metalSessionMatmulMpsNTEnc(a.metalHandle,b.metalHandle,c.metalHandle,2,4,4,4,0)
  doAssert sessionEnd(be)
  let prefilled=sessionRead(be,c,8)
  for i,v in @[1'f32,2,3,4,5,6,7,8]:doAssert abs(prefilled[i]-v)<1e-6
  echo "PASS: decode M=1 rejected by both MPS APIs; native decode and MPS prefill M=2 match reference"

main()
