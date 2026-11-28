## Included by packed_training. The records preserve genuine document endings.

type
  EosRecord = tuple[offset, tokens: int]
  EosDocuments = object
    bytes: MappedView[uint8]
    records: seq[EosRecord]
    bits, vocab: int
  EosCorpus = object
    train, validation: array[PackedDomain, EosDocuments]

proc loadEosCorpus(path: string, corpus: PackedCorpus): EosCorpus =
  let manifest = parseFile(path / "manifest.json")
  let bits = manifest["bits_per_token"].getInt()
  if manifest["format"].getStr() != "spectron-eos-v1" or
      manifest["tokenizer_md5"].getStr() != corpus.tokenizerHash or
      bits notin 8 .. 16 or corpus.vocab > 1 shl bits:
    raise newException(ValueError, "EOS examples and training data use different tokenizers or formats")
  for (split, records) in [("train", addr result.train), ("validation", addr result.validation)]:
    for domain in PackedDomain:
      let prefix = if domain == packedCode: "code" else: "english"
      let name = prefix & "-" & split & ".bin"
      let file = path / name
      let size = getFileSize(file)
      if size < 5 or size > int.high:
        raise newException(ValueError, "Invalid EOS data file " & file)
      var docs = EosDocuments(bits: bits, vocab: corpus.vocab,
        bytes: mappedView[uint8](mapWeights(file), 0, int(size)))
      let bytes = docs.bytes.dataPtr
      var offset = 0
      while offset < size:
        if size - offset < 4: raise newException(ValueError, "Truncated EOS record in " & file)
        let tokens = int(bytes[offset]) or (int(bytes[offset + 1]) shl 8) or
                     (int(bytes[offset + 2]) shl 16) or (int(bytes[offset + 3]) shl 24)
        offset += 4
        let payload = (int64(tokens) * bits + 7) div 8
        if tokens <= 0 or payload > size - offset:
          raise newException(ValueError, "Invalid EOS record size in " & file)
        docs.records.add((offset, tokens))
        offset += int(payload)
      if docs.records.len == 0 or docs.records.len != manifest["documents"][name].getInt():
        raise newException(ValueError, "EOS record count disagrees with manifest for " & file)
      records[][domain] = move(docs)

proc eosWindow(docs: EosDocuments, context: int, rng: var Rand,
               ids, targets: var seq[int]): int =
  ## Copy a variable-length suffix up to a true boundary. Continue with the
  ## head of another document to train EOS as a boundary inside the window.
  var doc = docs.records[rng.rand(docs.records.high)]
  let tail = min(doc.tokens, rng.rand(max(1, context div 2) .. context))
  var index = doc.tokens - tail
  let bytes = docs.bytes.dataPtr
  let mask = (1'u32 shl docs.bits) - 1
  ids.setLen(context)
  targets.setLen(context)
  for i in 0 .. context:
    var id: int
    if index == doc.tokens:
      id = docs.vocab
      inc result
      doc = docs.records[rng.rand(docs.records.high)]
      index = 0
    else:
      let bit = int64(index) * docs.bits
      let offset = doc.offset + int(bit shr 3)
      let shift = int(bit and 7)
      var raw = uint32(bytes[offset])
      if shift + docs.bits > 8: raw = raw or (uint32(bytes[offset + 1]) shl 8)
      if shift + docs.bits > 16: raw = raw or (uint32(bytes[offset + 2]) shl 16)
      id = int((raw shr shift) and mask)
      if id >= docs.vocab: raise newException(ValueError, "Invalid BPE ID in EOS record")
      inc index
    if i < context: ids[i] = id
    if i > 0: targets[i - 1] = id

proc extendWithEos(model: var NimformerModelSpectron, adam: var ModelAdamState) =
  ## Keep every existing parameter and Adam moment. The new output row starts
  ## at zero and leaves the old token logits intact.
  inc model.vocab
  model.hasEos = true
  model.embed.vocab = model.vocab
  model.outProj.outF = model.vocab
  template grow(weights, moments: untyped) =
    weights.shape[0] = model.vocab
    let size = model.vocab * (if weights.shape.len == 1: 1 else: model.embed.dim)
    weights.data.setLen(size)
    moments.m.setLen(size)
    moments.v.setLen(size)
  grow(model.embed.weight, adam.embed)
  grow(model.outProj.weight, adam.outProjW)
  grow(model.outProj.bias, adam.outProjB)
  let dim = model.embed.dim
  let boundary = randnTensor(@[1, dim], sqrt(2'f32 / float32(model.vocab)))
  for i in 0 ..< dim:
    model.embed.weight.data[model.eosTokenId * dim + i] = boundary.data[i]

proc evalEos(model: NimformerModelSpectron, docs: EosCorpus, options: PackedTrainingOptions,
             step: int, ctx: Backend) =
  let eos = model.eosTokenId
  var rng = initRand(230171)
  var ids, targets = newSeq[seq[int]](options.batch)
  for row in 0 ..< options.batch:
    let domain = PackedDomain(row mod 2)
    discard docs.validation[domain].eosWindow(options.context, rng, ids[row], targets[row])
  let logits = model.forwardBatch(ids, ctx)
  var correct, early, before = 0
  var probability = 0.0
  for row, labels in targets:
    for position, target in labels:
      let offset = (row * options.context + position) * model.vocab
      var best = 0
      for token in 0 ..< model.vocab:
        if not finite32(logits.data[offset + token]):
          raise newException(SpectronNumericalError, "Non-finite EOS validation logit")
        if logits.data[offset + token] > logits.data[offset + best]: best = token
      if target == eos:
        if best == eos: inc correct
        let peak = float64(logits.data[offset + best])
        var total = 0.0
        for token in 0 ..< model.vocab:
          total += exp(float64(logits.data[offset + token]) - peak)
        probability += exp(float64(logits.data[offset + eos]) - peak) / total
        break
      inc before
      if best == eos: inc early
  echo &"[eos-val] step={step} stop_top1={correct}/{options.batch} " &
       &"early_stop_top1={early}/{before} mean_stop_p={probability / float64(options.batch):.6f}"
