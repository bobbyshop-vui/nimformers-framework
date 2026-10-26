## train_gpu_ops.nim — host wrapper chạy kernel train_gpu_vm trên GPU (Metal/CUDA/Vulkan).
## Mỗi hàm = 1 session: upload 1 lần, chạy chuỗi kernel, đọc 1 lần (giống mamba2_real.nim).
import backend
import nimformer
import customfloat
import mamba2_ssm_gpu    # runMicroVmChain, MvmStep, blit
import std/math
import train_gpu_vm

proc isGpuCtx*(ctx: Backend): bool =
  ctx.kind in {bkMetal, bkCuda, bkVulkan}

proc rmsNormForwardGpu*(ctx: Backend; x, gamma: seq[float32]; rows, C: int;
                        eps: float32): seq[float32] =
  let L = rmsLayout(rows, C, eps)
  var buf = newSeq[float32](L.total)
  buf.blit(L.x, x, 0, rows * C); buf.blit(L.gamma, gamma, 0, C)
  let tg = rmsRowTg(C)
  discard runMicroVmChain(ctx, [
    (buildRmsFwdProgram(L), rows * tg, tg * 4, tg, "rms_fwd")], buf)
  result = buf[L.y ..< L.y + rows * C]

proc rmsNormBackwardGpu*(ctx: Backend; x, gamma, dOut: seq[float32]; rows, C: int;
                         eps: float32): tuple[dX, dGamma: seq[float32]] =
  let L = rmsLayout(rows, C, eps)
  var buf = newSeq[float32](L.total)          # vùng dGamma = 0 cho atomic
  buf.blit(L.x, x, 0, rows * C); buf.blit(L.gamma, gamma, 0, C)
  buf.blit(L.dy, dOut, 0, rows * C)
  let tg = rmsRowTg(C)
  discard runMicroVmChain(ctx, [
    (buildRmsBwdProgram(L), rows * tg, tg * 4, tg, "rms_bwd")], buf)
  result = (buf[L.dx ..< L.dx + rows * C], buf[L.dGamma ..< L.dGamma + C])

proc linearPlusForwardGpu*(ctx: Backend; qkv: seq[float32]; B, T, H, D: int): seq[float32] =
  ## qkv: [B*T, 3C] (q|k|v mỗi token). Trả y [B*T, C].
  let L = lpLayout(B, T, H, D)
  var buf = newSeq[float32](L.total)
  buf.blit(L.qkv, qkv, 0, L.BT * 3 * L.C)
  let tg = lpPhiTg(L)
  discard runMicroVmChain(ctx, [
    (buildLpPhiProgram(L), L.BT * 2 * L.C, tg * 4, tg, "lp_phi"),
    (buildLpDenProgram(L), B * H, 4, 0, "lp_den"),
    (buildLpFwdProgram(L), B * L.C, 4, 0, "lp_fwd")], buf)
  result = buf[L.y ..< L.y + L.BT * L.C]

proc linearPlusBackwardGpu*(ctx: Backend; qkv, dY: seq[float32]; B, T, H, D: int): seq[float32] =
  ## Dựng lại forward trong cùng session rồi backward 2 pass. Trả dQKV [B*T, 3C]
  ## (đúng layout dProj cho qkvProj.backward).
  let L = lpLayout(B, T, H, D)
  var buf = newSeq[float32](L.total)          # vùng trạng thái scan = 0; dQKV được ghi đè
  buf.blit(L.qkv, qkv, 0, L.BT * 3 * L.C)
  buf.blit(L.dY, dY, 0, L.BT * L.C)
  let tg = lpPhiTg(L)
  discard runMicroVmChain(ctx, [
    (buildLpPhiProgram(L), L.BT * 2 * L.C, tg * 4, tg, "lp_phi"),
    (buildLpDenProgram(L), B * H, 4, 0, "lp_den"),
    (buildLpFwdProgram(L), B * L.C, 4, 0, "lp_fwd"),
    (buildLpDDenProgram(L), L.BT * H, 4, 0, "lp_dden"),
    (buildLpBwdQProgram(L), B * L.C, 4, 0, "lp_bwd_q"),
    (buildLpBwdKProgram(L), B * L.C, 4, 0, "lp_bwd_k"),
    (buildLpBwdVProgram(L), B * L.C, 4, 0, "lp_bwd_v")], buf)
  result = buf[L.dQKV ..< L.dQKV + L.BT * 3 * L.C]

proc softmaxCrossEntropyGpu*(ctx: Backend; logits: seq[float32]; targets: seq[float32];
                             rows, V, nValid: int): tuple[loss: float32, dLogits: seq[float32]] =
  ## targets[r] < 0 => bỏ qua. nValid = số hàng hợp lệ (chia loss/gradient cho nValid).
  if nValid == 0: return (0'f32, newSeq[float32](rows * V))
  let L = ceLayout(rows, V)
  var buf = newSeq[float32](L.total)
  buf.blit(L.logits, logits, 0, rows * V); buf.blit(L.tgt, targets, 0, rows)
  buf[L.scal] = 1'f32 / float32(nValid)
  let tg = rmsRowTg(V)
  discard runMicroVmChain(ctx, [
    (buildCeProgram(L), rows * tg, tg * 4, tg, "ce"),
    (buildCeSumProgram(L), 1, 4, 1, "ce_sum")], buf)
  result = (buf[L.sum], buf[L.dLogits ..< L.dLogits + rows * V])

const AdamChunk = 8 * 1024 * 1024     ## phần tử / lần dispatch (chia hết AdamTg)

proc adamStepFusedGpu*(ctx: Backend; param: var seq[float32]; grad: seq[float32];
                       m, v: var seq[float32]; step: int; lr: float32;
                       b1: float32 = 0.9'f32; b2: float32 = 0.999'f32;
                       eps: float32 = 1e-8'f32) =
  ## Adam 1 kernel / chunk (thay ~14 phép be* rời). `step` đã tăng (>=1) trước khi gọi.
  let n = param.len
  let ibc1 = 1'f32 / (1'f32 - pow(b1, float32(step)))
  let ibc2 = 1'f32 / (1'f32 - pow(b2, float32(step)))
  var off = 0
  while off < n:
    let cnt = min(AdamChunk, n - off)
    let pad = ((cnt + AdamTg - 1) div AdamTg) * AdamTg
    let L = adamLayout(pad)
    var buf = newSeq[float32](L.total)
    buf.blit(L.p, param, off, cnt); buf.blit(L.g, grad, off, cnt)
    buf.blit(L.m, m, off, cnt); buf.blit(L.v, v, off, cnt)
    buf[L.scal] = lr; buf[L.scal + 1] = ibc1; buf[L.scal + 2] = ibc2
    discard runMicroVmChain(ctx, [
      (buildAdamProgram(L, b1, b2, eps), pad, 4, AdamTg, "adam")], buf)
    for i in 0 ..< cnt:
      param[off + i] = buf[L.p + i]; m[off + i] = buf[L.m + i]; v[off + i] = buf[L.v + i]
    off += cnt

proc apfAdamStepFused*(ctx: Backend; param: var Tensor; grad: Tensor; state: var ApfAdamState;
                       lr: float32 = 1e-3'f32; b1: float32 = 0.9'f32; b2: float32 = 0.999'f32;
                       eps: float32 = 1e-8'f32; requantizeEvery: int = 1): CustomFloat =
  ## Drop-in cho apfAdamStepGpu: Adam 1 micro-kernel. Nếu requantizeEvery hữu hạn (APF
  ## requantize CustomFloat) thì giữ đường cũ vì bước requantize nằm ngoài kernel này.
  if requantizeEvery > 0 and requantizeEvery != int.high:
    return apfAdamStepGpu(ctx, param, grad, state, lr, b1, b2, eps, requantizeEvery)
  inc state.step
  adamStepFusedGpu(ctx, param.data, grad.data, state.m, state.v, state.step, lr, b1, b2, eps)
  result = buildCustomDtypeForTensor(param.data)
