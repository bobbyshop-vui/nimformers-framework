## Independent reference recurrence for inference layout and nonzero chunk carry.
import std/[math,strformat]
import mamba2_full_vm, prefill_inference_kernels, vm_interp

for T in [2,3,7,8,31,64,128]:
  for carry in [false,true]:
    let referenceLayout=m2Layout(2,T,4,2,3,2,4,8,1e-5'f32,2)
    var reference=newSeq[float32](referenceLayout.fwdEnd)
    for i in 0..<2*T*referenceLayout.cDim:
      reference[referenceLayout.conv+i]=sin(float32(i)*0.13'f32)*0.4'f32
    for i in 0..<2*T*referenceLayout.H:reference[referenceLayout.dt+i]=0.07'f32
    for h in 0..<referenceLayout.H:
      reference[referenceLayout.aLog+h]=float32(h)*0.1'f32
      reference[referenceLayout.dSkip+h]=0.5'f32
    if carry:
      for b in 0..<2:
        for i in 0..<referenceLayout.HD*referenceLayout.N:
          reference[referenceLayout.states+b*(T+1)*referenceLayout.HD*referenceLayout.N+i]=cos(float32(i+b)*0.1'f32)
    let initial=reference
    runGrid(buildM2ScanProgram(referenceLayout),reference,2*referenceLayout.HD,tgSize=4)
    for mode in ["parallel","serial","coalesced"]:
      let compact=mode!="parallel"
      let L=prefixLayout(2,T,4,2,3,2,4,8,1e-5'f32,2,compact)
      var actual=newSeq[float32](L.fwdEnd)
      for i in 0..<L.states:actual[i]=initial[i]
      let stride=(if compact: 2 else: T+1)*L.HD*L.N
      for b in 0..<2:
        for i in 0..<L.HD*L.N:
          actual[L.states+b*stride+i]=initial[referenceLayout.states+b*(T+1)*L.HD*L.N+i]
      if mode=="coalesced":
        runGrid(buildPrefixStateTransposeProgram(L,true),actual,2*L.HD*L.N,tgSize=4)
        runGrid(buildCompactPrefixScanProgram(L,true),actual,2*L.HD,tgSize=4)
        runGrid(buildPrefixStateTransposeProgram(L,false),actual,2*L.HD*L.N,tgSize=4)
      elif compact:runGrid(buildCompactPrefixScanProgram(L),actual,2*L.HD,tgSize=4)
      else:
        let tg=m2ScanTgSize(T)
        runGrid(buildPrefixParallelScanProgram(L,tg,carry),actual,2*L.HD*tg,tgSize=tg,tgFloats=4*tg)
      var error=0'f32
      for i in 0..<2*T*L.C:error=max(error,abs(actual[L.yPre+i]-reference[referenceLayout.yPre+i]))
      for b in 0..<2:
        let terminal=L.states+b*stride+(if mode=="coalesced": 0 elif compact: 1 else: T)*L.HD*L.N
        for i in 0..<L.HD*L.N:
          error=max(error,abs(actual[terminal+i]-reference[referenceLayout.states+(b*(T+1)+T)*L.HD*L.N+i]))
      doAssert error<2e-5'f32, &"T={T} carry={carry} compact={compact} error={error}"
    echo &"PASS inference scan T={T} carry={carry}: serial compact + parallel match original recurrence"
