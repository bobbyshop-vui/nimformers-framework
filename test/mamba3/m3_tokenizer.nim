## Native Llama-3 tokenizer.json adapter (same tokenizer in training and inference).
import std/[os, json, strutils]
import native_hf_tokenizer as native

type HfTokenizer* = object
  model: native.HFTokenizer
  vocab*, bosId*, eosId*, unkId*: int
  backend*: string
  special*, banned*, garbageSpace*: seq[bool]
  stopIds*: seq[int]

proc newHfTokenizer*(hfDir:string): HfTokenizer =
  result.model=native.newHFTokenizer(hfDir)
  result.vocab=result.model.totalVocabSize
  result.bosId=result.model.bosId; result.eosId=result.model.eosId; result.unkId= -1
  result.backend="native-llama3-bpe"
  if result.vocab!=128256: raise newException(ValueError,"Mamba-3 187M requires the official Llama-3 tokenizer, vocab=128256")
  result.special=newSeq[bool](result.vocab)
  result.banned=newSeq[bool](result.vocab)
  result.garbageSpace=newSeq[bool](result.vocab)
  let doc=parseFile(hfDir / "tokenizer.json")
  for token in doc["added_tokens"]:
    let id=token["id"].getInt
    if id<0 or id>=result.vocab: raise newException(ValueError,"invalid added token id")
    if token.hasKey("special") and token["special"].getBool:
      result.special[id]=true; result.banned[id]=true
    if token["content"].getStr in ["<|end_of_text|>","<|eot_id|>","<|end_of_message|>"]:
      result.stopIds.add(id); result.banned[id]=false
  if result.eosId>=0 and result.eosId notin result.stopIds:
    result.stopIds.add(result.eosId); result.banned[result.eosId]=false
proc encodeHf*(tok:HfTokenizer; text:string; addEos=false; addBos=true): seq[int] =
  result=native.encode(tok.model,text,addSpecialTokens=false)
  if addBos: result.insert(tok.bosId,0)
  if addEos: result.add(tok.eosId)
proc decodeHf*(tok:HfTokenizer; ids:seq[int]; skipSpecial=true):string =
  native.decode(tok.model,ids,skipSpecialTokens=skipSpecial)
proc vocabSizeHf*(tok:HfTokenizer):int = tok.vocab
proc isStopIdHf*(tok:HfTokenizer; id:int):bool = id in tok.stopIds
proc pieceOfHf*(tok:HfTokenizer; id:int):string = native.decode(tok.model,@[id],skipSpecialTokens=false)
