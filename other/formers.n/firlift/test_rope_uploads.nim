## Regression: hidden-sized cos/sin uploads are not token embeddings.
## nim c -r --path:vendor/bybylang other/formers.n/firlift/test_rope_uploads.nim
import ./firlift
import ./steps

proc p(name: string; buffer: int): PortRef =
  PortRef(name: name, buffer: buffer, bytes: 4096*4)

proc add(t: var Trace; impl: string; ins: seq[PortRef] = @[];
         outs: seq[PortRef] = @[]): Op {.discardable.} =
  result = Op(id: t.ops.len, order: t.ops.len, impl: "session:" & impl,
    parent: -1, ins: ins, outs: outs)
  t.ops.add result

var t: Trace
let embedding = t.add("sessionUpload", outs = @[p("dst", 0)])
t.add("sessionRmsnormEnc", ins = @[p("x", 0)], outs = @[p("y", 1)])
let cosUpload = t.add("sessionUpload", outs = @[p("dst", 2)])
let sinUpload = t.add("sessionUpload", outs = @[p("dst", 3)])
let rope = t.add("sessionRope", ins = @[p("x", 1), p("cosT", 2), p("sinT", 3)],
  outs = @[p("x", 1)])
doAssert not t.isRopeTableUpload(embedding, 0)
doAssert t.isRopeTableUpload(cosUpload, 2)
doAssert t.isRopeTableUpload(sinUpload, 3)
doAssert t.isUploadedRopeTable(rope, "cosT")
doAssert t.isUploadedRopeTable(rope, "sinT")
# Reused buffer IDs must not retain their old role.
t.add("sessionFree", outs = @[p("buffer", 2)])
let reused = t.add("sessionUpload", outs = @[p("dst", 2)])
t.add("sessionRmsnormEnc", ins = @[p("x", 2)], outs = @[p("y", 4)])
doAssert not t.isRopeTableUpload(reused, 2)
doAssert t.isUploadedRopeTable(rope, "cosT")
# Graph-computed tables retain their producer and must not become position-derived.
let overwritten = t.add("sessionUpload", outs = @[p("dst", 5)])
t.add("sessionVecOp", outs = @[p("c", 5)])
let computedRope = t.add("sessionRope", ins = @[p("cosT", 5), p("sinT", 3)])
doAssert not t.isRopeTableUpload(overwritten, 5)
doAssert not t.isUploadedRopeTable(computedRope, "cosT")
# A replacement upload terminates the preceding upload's lifetime.
let old = t.add("sessionUpload", outs = @[p("dst", 6)])
let replacement = t.add("sessionUpload", outs = @[p("dst", 6)])
t.add("sessionRopeKvEnc", ins = @[p("cosT", 6)])
doAssert not t.isRopeTableUpload(old, 6)
doAssert t.isRopeTableUpload(replacement, 6)
echo "PASS RoPE upload classification, computed tables and reused buffer lifetimes"
