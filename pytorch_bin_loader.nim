## pytorch_bin_loader.nim
## Load pytorch_model.bin của ayoubkirouane/Mamba-Chat-2.8B (dựa trên
## mamba_ssm gốc - MambaLMHeadModel, KHÔNG phải transformers.AutoModel,
## dim=2560, 64 layer mamba thuần). Model gốc lưu ở FP16
## (`MambaLMHeadModel.from_pretrained(..., dtype=torch.float16)`), nên xử
## lý cẩn thận:
##   - torch.load trả tensor fp16 -> BẮT BUỘC ép fp32 trước khi tobytes,
##     nếu không buffer sẽ là 2 byte/phần tử (fp16) trong khi code Nim bên
##     dưới giả định 4 byte/phần tử (fp32) -> đọc sai hoàn toàn hoặc crash.
##     pyTensorToSeq() dưới đây ép fp32 ở PHÍA NUMPY (arr.astype("float32"))
##     - KHÔNG dùng tensor.to(torch.float32) (xem lý do kỹ thuật + lỗi nimpy
##     cụ thể đã gặp trong comment ngay tại pyTensorToSeq) - đừng bỏ bước ép
##     fp32 này khi sửa code.
##   - fp16 có dải giá trị hẹp (~[-65504, 65504], eps lớn) - vài tensor như
##     A_log, dt_proj.bias của Mamba SSM có thể có giá trị cực nhỏ/cực lớn
##     sau khi exp/softplus; ép fp32 xong vẫn giữ nguyên giá trị (chỉ đổi
##     cách biểu diễn bit, không tăng precision đã mất) - không có cách nào
##     phục hồi precision đã mất khi model gốc lưu fp16, chỉ là tránh MẤT
##     THÊM precision trong lúc convert.
##
## Sau bước này, xem thêm ghi chú "fp16" trong lora.nim (LoRA A/B init +
## Adam eps phải đủ lớn để không underflow khi base đến từ fp16).
##
## Chạy trước:
##   1. Tải file: huggingface-cli download ayoubkirouane/Mamba-Chat-2.8B --local-dir ./mamba-chat-2.8b
##   2. ./pytorch_bin_loader ./mamba-chat-2.8b/pytorch_model.bin
##      -> chỉ ghi manifest (.manifest.json, rất nhẹ) để xem ten tensor thật
##         (backbone.embedding.weight, backbone.layers.N.mixer.*, ...)
##   3. Pipeline fine-tune (graft_attention.nim --from-hf, jamba_hybrid.
##      loadMambaOnlyFromHFBin) đọc THẲNG file .bin gốc này qua
##      loadAllHFTensors() - KHÔNG cần xuất thêm 1 file .bin nào khác,
##      tránh nhân đôi dung lượng đĩa (checkpoint HF gốc có thể vài GB).
##      Cờ `--export out.bin` bên dưới CHỈ là tiện ích tùy chọn (archive/
##      inspect offline), KHÔNG phải bước bắt buộc - xem cảnh báo trong
##      exportBinToRawBin().
##
## YÊU CẦU: `nimble install nimpy`, và python env đã có torch (bạn xác nhận
## đã cài). Biến môi trường NIMPY_PYTHON hoặc PYTHONHOME có thể cần set nếu
## nimpy không tự tìm ra đúng python có torch.

import std/[json, os, strformat, tables, strutils]
import nimpy
import rawbin   # xuat sang .bin THO (khong nen) - KHONG dung .nimq (quant.nim)
                # o buoc nay, vi day la mot phan cua pipeline fine-tune

let pyBuiltins = pyImport("builtins")
proc pyLen(o: PyObject): int =
  ## LUU Y: KHONG dung o.len() - "len" trong Python la HAM BUILT-IN
  ## (len(x)), KHONG PHAI method cua object (x.len()) - torch.Size (tuple)
  ## va nhieu object khac KHONG co attribute "len" -> o.len() se bi nimpy
  ## co goi dong (dynamic dispatch) sang Python va bao loi runtime "No
  ## callable attribute: len". Dung ham nay (goi qua builtins.len(x)) o MOI
  ## noi thay vi o.len().to(int).
  pyBuiltins.len(o).to(int)

type
  HFTensorInfo* = object
    name*: string
    shape*: seq[int]
    dtype*: string

# ─────────────────────────────────────────────────────────────────
# Gọi Python: torch.load(path, map_location="cpu") -> dict[str, Tensor]
# Hỗ trợ cả .bin (pickle) lẫn .safetensors (qua thư viện safetensors nếu có).
# ─────────────────────────────────────────────────────────────────
proc pyLoadStateDict(path: string): PyObject =
  let torch = pyImport("torch")
  if path.endsWith(".safetensors"):
    let st = pyImport("safetensors.torch")
    result = st.load_file(path)
  else:
    result = torch.load(path, map_location = "cpu")

proc pyTensorToSeq(t: PyObject): tuple[data: seq[float32], shape: seq[int]] =
  ## .detach().contiguous().float().flatten() -> numpy -> tobytes() -> seq[float32].
  ## SUA (Codestral-7B / bf16): checkpoint cua Mistral luu dtype bfloat16 -
  ## numpy KHONG HO TRO bfloat16 (khong co kieu nay trong numpy) nen
  ## tensor.numpy() RAISE NGAY LAP TUC ("Got unsupported ScalarType
  ## BFloat16"), truoc ca khi kip chay .astype("float32") o buoc sau (buoc
  ## do gia dinh sai la .numpy() luon thanh cong roi moi can doi dtype).
  ## BAT BUOC ep ve float32 O PHIA TORCH (torch CO ho tro bfloat16 that,
  ## chi numpy la khong) TRUOC khi goi .numpy().
  ## LUU Y QUAN TRONG (van dung nhu ghi chu cu): KHONG dung tensor.to(torch.float32)
  ## - nimpy TU DINH NGHIA san 1 proc rieng ten "to" (dung de convert
  ## PyObject -> kieu Nim, vd `.to(int)`), trung ten voi method ".to(dtype)"
  ## cua PyTorch -> bi Nim/nimpy resolve NHAM. Dung ".float()" thay the -
  ## day la shorthand CHINH THUC cua PyTorch cho ".to(torch.float32)", TEN
  ## KHAC hoan toan nen khong dung ham "to" cua nimpy, khong bi nham lan.
  ## LUU Y THEM (phat hien them 1 truong hop nham lan y het ".to"): goi truc
  ## tiep ".float()" tren PyObject cung bi Nim/nimpy resolve NHAM thanh ham
  ## convert builtin `float()` cua chinh Nim (ep kieu ve kieu so `float` cua
  ## Nim, khong phai goi method Python) -> "type mismatch: expected float".
  ## Giai phap: dung callMethod("float") TUONG MINH - ep nimpy phai dispatch
  ## dong sang Python thay vi de Nim tu resolve ten truoc.
  let flat = t.detach().contiguous().callMethod("float").flatten()
  let shapeObj = t.shape
  var shape: seq[int] = @[]
  for i in 0 ..< pyLen(shapeObj):
    shape.add(shapeObj[i].to(int))
  let n = pyLen(flat)
  var data = newSeq[float32](n)
  # Vòng lặp Python->Nim từng phần tử chậm với model 1B (>1e9 tham số cho
  # toàn bộ checkpoint) nếu làm 100% qua nimpy indexing, nên chuyển qua
  # numpy rồi lấy buffer thẳng cho nhanh.
  let arrRaw = flat.numpy()             # gio da CHAC CHAN la float32 (da .float() o tren)
  let arr = arrRaw.astype("float32")    # giu lai (vo hai, phong truong hop dtype khac lot qua)
  let buf = arr.tobytes()
  let raw = buf.to(string)
  assert raw.len == n * 4
  copyMem(addr data[0], unsafeAddr raw[0], raw.len)
  result = (data, shape)

proc listStateDictInfo*(path: string): seq[HFTensorInfo] =
  ## Chỉ đọc metadata (tên + shape + dtype), KHÔNG kéo hết data vào RAM.
  ## Dùng cái này trước để xem manifest rồi mới quyết định map tên ra sao.
  let sd = pyLoadStateDict(path)
  result = @[]
  for key in sd.keys():
    let t = sd[key]
    let name = key.to(string)
    var shape: seq[int] = @[]
    let shapeObj = t.shape
    for i in 0 ..< pyLen(shapeObj):
      shape.add(shapeObj[i].to(int))
    # LUU Y: t.dtype la 1 object Python kieu 'torch.dtype' (vd
    # torch.float16), KHONG PHAI PyObject kieu str -> goi thang
    # ".to(string)" tren no se bi nimpy tu choi voi loi runtime "Can't
    # convert python obj of type 'torch.dtype' to string" (.to(string) cua
    # nimpy chi chap nhan object THAT SU la Python str, khong tu dong goi
    # str() ho). Phai goi ham built-in str() cua Python (qua pyBuiltins)
    # TRUOC de bien no thanh 1 PyObject str that, roi moi .to(string) duoc.
    let dtype = pyBuiltins.str(t.dtype).to(string)
    result.add(HFTensorInfo(name: name, shape: shape, dtype: dtype))

proc writeManifest*(infos: seq[HFTensorInfo], outPath: string) =
  var arr = newJArray()
  for info in infos:
    var shapeArr = newJArray()
    for s in info.shape: shapeArr.add(%s)
    arr.add(%*{"name": info.name, "shape": shapeArr, "dtype": info.dtype})
  writeFile(outPath, pretty(arr))
  echo &"[loader] Da ghi manifest {infos.len} tensor -> {outPath}"

proc loadAllHFTensors*(path: string): seq[RawBinTensor] =
  ## Đọc TOÀN BỘ checkpoint HF (.bin/.safetensors) thẳng vào RAM (seq
  ## RawBinTensor), KHÔNG ghi ra bất kỳ file nào trên đĩa. Dùng hàm này khi
  ## bạn cần build model Nim trực tiếp từ .bin gốc (vd
  ## jamba_hybrid.loadMambaOnlyFromHFBin) mà không muốn nhân đôi dung lượng
  ## đĩa bằng 1 file .bin trung gian.
  let sd = pyLoadStateDict(path)
  result = @[]
  var n = 0
  for key in sd.keys():
    let name = key.to(string)
    let t = sd[key]
    let (data, shape) = pyTensorToSeq(t)
    result.add(RawBinTensor(name: name, shape: shape, data: data))
    inc n
    if n mod 20 == 0:
      echo &"[loader] ... {n} tensor da doc (RAM, khong ghi dia)"

proc exportBinToRawBin*(binPath, outBinPath: string, arch: array[5, int] = [0, 0, 0, 0, 0]) =
  ## TIỆN ÍCH TÙY CHỌN, KHÔNG PHẢI BƯỚC BẮT BUỘC của pipeline fine-tune.
  ## Ghi toàn bộ checkpoint .bin/.safetensors ra 1 file .bin thô KHÁC (rawbin.nim)
  ## - nghĩa là NHÂN ĐÔI dung lượng đĩa (vd checkpoint 5.5GB -> thêm 5.5GB nữa).
  ## CHỈ dùng khi bạn thật sự cần 1 bản sao đọc được bằng Nim thuần không cần
  ## Python nữa (vd để archive/inspect offline). Cho pipeline fine-tune bình
  ## thường (graft_attention --from-hf, jamba_hybrid.loadMambaOnlyFromHFBin),
  ## KHÔNG gọi hàm này - dùng loadAllHFTensors (chỉ ở RAM) thay vào đó.
  let tensors = loadAllHFTensors(binPath)
  var meta = initTable[string, string]()
  meta["vocab"] = $arch[0]
  meta["embedDim"] = $arch[1]
  meta["nHeads"] = $arch[2]
  meta["nLayers"] = $arch[3]
  meta["ffMult"] = $arch[4]
  saveRawBin(outBinPath, meta, tensors)
  echo &"[loader] Xong: {tensors.len} tensor -> {outBinPath} (CANH BAO: file nay la BAN SAO THEM, chiem them dia)"

proc loadTensorByName*(path, tensorName: string): tuple[data: seq[float32], shape: seq[int]] =
  ## Đọc đúng 1 tensor theo tên (đỡ tốn RAM khi chỉ cần vài tensor, vd
  ## embedding hoặc 1 layer cụ thể để so sánh/khởi tạo).
  let sd = pyLoadStateDict(path)
  # LUU Y: KHONG dung sd.contains(tensorName) - dict Python KHONG co method
  # ".contains()" (chi co toan tu "in"/__contains__, khong phai method) ->
  # nimpy se co gang goi dong (dynamic dispatch sang Python) va bao loi
  # runtime "No callable attribute: contains". Duyet qua sd.keys() (method
  # that su, an toan) roi so sanh ten thay vi dung .contains().
  var found = false
  for k in sd.keys():
    if k.to(string) == tensorName:
      found = true
      break
  if not found:
    raise newException(KeyError, "Khong thay tensor: " & tensorName)
  result = pyTensorToSeq(sd[tensorName])

when isMainModule:
  # cách dùng: ./pytorch_bin_loader <path/pytorch_model.bin> [--export out.bin]
  # (--export la TUY CHON, ghi them 1 ban sao .bin - xem canh bao trong
  #  exportBinToRawBin(). Pipeline fine-tune binh thuong KHONG can co nay,
  #  no doc thang tu file .bin goc qua loadAllHFTensors/graft_attention --from-hf)
  let args = commandLineParams()
  if args.len < 1:
    echo "usage: pytorch_bin_loader <model.bin|model.safetensors> [--export out.bin]"
    quit(1)
  let path = args[0]
  echo &"[loader] Doc manifest tu {path} ..."
  let infos = listStateDictInfo(path)
  writeManifest(infos, path & ".manifest.json")
  for info in infos[0 ..< min(20, infos.len)]:
    echo &"  {info.name}  shape={info.shape}  dtype={info.dtype}"
  if infos.len > 20:
    echo &"  ... con {infos.len - 20} tensor nua, xem full trong {path}.manifest.json"
  if args.len >= 3 and args[1] == "--export":
    exportBinToRawBin(path, args[2], [0, 0, 0, 0, 0])