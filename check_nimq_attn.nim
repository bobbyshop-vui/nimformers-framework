## check_nimq_attn.nim
## Tiện ích nhỏ: liệt kê tên tensor bên trong 1 file .nimq (quant state dict)
## và báo xem có tensor nào thuộc Attention/LinearPlus không - dùng để xác
## nhận 1 file .nimq gốc (trước graft) chắc chắn CHƯA có Attention.
##
## Cách build (trên máy đã có project, cùng chỗ với các .nim khác):
##   nim c -d:release -o:check_nimq_attn check_nimq_attn.nim
## Cách chạy:
##   ./check_nimq_attn codestral_mamba_7b_fp4.nimq

import std/[os, strutils]
import quant   # loadQuantStateDict

when isMainModule:
  if paramCount() < 1:
    echo "usage: check_nimq_attn <file.nimq>"
    quit(1)
  let path = paramStr(1)
  let (arch, sd) = loadQuantStateDict(path)
  echo "arch = ", arch
  echo "So tensor: ", sd.len

  var attnNames: seq[string] = @[]
  for (name, qt) in sd:
    let low = name.toLowerAscii()
    if "attn" in low or "attention" in low or "linearplus" in low or "q_proj" in low or "k_proj" in low or "v_proj" in low:
      attnNames.add(name)

  if attnNames.len == 0:
    echo "-> KHONG tim thay tensor Attention nao. File nay la Mamba thuan, CHUA graft."
  else:
    echo "-> Tim thay ", attnNames.len, " tensor lien quan Attention:"
    for n in attnNames: echo "   ", n