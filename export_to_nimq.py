#!/usr/bin/env python3
"""
export_to_nimq.py - Hỗ trợ asymmetric int4 (Q4_K_M) với MPS acceleration

FIX (2026-07-17): Bỏ dòng transpose thừa (weight_f32.T) sau khi gọi
dequantize_gptq_linear() trong export_to_nimq(). Hàm dequantize_gptq_linear
đã tự transpose 1 lần để trả về đúng layout [out_features, in_features]
(convention của nn.Linear/HF). Transpose thêm lần nữa ở nơi gọi làm layout
bị lật ngược thành [in_features, out_features], khiến mọi Linear layer nạp
từ GPTQ (q/k/v/o_proj, gate/up/down_proj) bị load sai hoàn toàn phía Nim,
vì loadLinear() bên generate.nim gán thẳng .data mà không transpose (đúng
theo giả định layout [outF, inF]).
"""

import os
import sys
import argparse
import struct
import json
from pathlib import Path
from typing import Dict, Tuple, Any, Set
import numpy as np
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ============================================================
# Kiểm tra MPS - FIX: force FP16
# ============================================================
USE_MPS = False
if torch.backends.mps.is_available():
    USE_MPS = True
    # === FIX: force FP16 để tránh lỗi BFloat16 ===
    torch.set_default_dtype(torch.float16)
    print("🔹 MPS (Metal) available - sẽ dùng GPU cho quantize (FP16)")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
else:
    print("🔹 MPS not available - dùng CPU")

# ============================================================
# QuantKind
# ============================================================
QK_FP32_RAW = 0
QK_INT8 = 1
QK_INT4 = 2
QK_INT4_ASYMMETRIC = 3


# ============================================================
# 1. Load model - FIX: force dtype cho MPS
# ============================================================

def load_model(model_name: str, is_gptq: bool = False) -> Tuple[Any, Any, Dict]:
    print(f"Loading model: {model_name}")

    # FIX: LUÔN load lên CPU dù USE_MPS=True. accelerate.load_checkpoint_in_model
    # cố set_module_tensor_to_device thẳng lên MPS với dtype gốc trong
    # checkpoint (thường là bfloat16) TRƯỚC KHI kịp ép sang float16 ->
    # "TypeError: Trying to convert BFloat16 to the MPS backend" vì MPS
    # không hỗ trợ bfloat16. Load CPU trước (ổn định, đã test nhiều lần),
    # sau đó dequantize_gptq_module() sẽ tự .to("mps") RIÊNG từng module 1
    # (lúc đó tensor đã là float16 rồi, MPS nhận bình thường).
    device = "cpu"
    print(f"   Using device: {device} (module GPTQ sẽ tự chuyển MPS lúc dequantize nếu USE_MPS)")

    # === FIX: force FP16 cho MPS ===
    torch_dtype = torch.float16

    if is_gptq:
        try:
            from auto_gptq import AutoGPTQForCausalLM
            # === FIX: thêm torch_dtype vào from_quantized ===
            model = AutoGPTQForCausalLM.from_quantized(
                model_name,
                device=device,
                use_triton=False,
                use_safetensors=True,
                trust_remote_code=True,
                disable_exllama=True,
                use_exllama=False,
                use_exllamav2=False,
                use_qigen=False,
                torch_dtype=torch_dtype,  # THÊM: force dtype
            )
        except ImportError:
            print("auto-gptq not installed. Falling back to normal transformers.")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch_dtype, device_map=device,
                trust_remote_code=True
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch_dtype, device_map=device,
            trust_remote_code=True
        )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)

    config_dict = {
        "vocab_size": config.vocab_size,
        "hidden_size": config.hidden_size,
        "intermediate_size": getattr(config, "intermediate_size", config.hidden_size * 4),
        "num_hidden_layers": config.num_hidden_layers,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": getattr(config, "num_key_value_heads", config.num_attention_heads),
        "max_position_embeddings": config.max_position_embeddings,
        "rms_norm_eps": getattr(config, "rms_norm_eps", 1e-5),
        "rope_theta": getattr(config, "rope_theta", 10000.0),
        "tie_word_embeddings": getattr(config, "tie_word_embeddings", False),
    }

    return model, tokenizer, config_dict


# ============================================================
# 2. GPTQ Dequantization
# ============================================================

def dequantize_gptq_torch_mps(qweight_t: torch.Tensor, qzeros_t: torch.Tensor,
                               scales_t: torch.Tensor, g_idx_t: torch.Tensor,
                               device: str, bits: int = 4) -> torch.Tensor:
    """
    NGHIÊM CẤM numpy - toàn bộ bằng torch theo yêu cầu. Dùng MPS thật cho
    phần compute nặng.

    Lý do KHÔNG gọi module.forward(eye) trực tiếp trên MPS (đã thử, luôn
    hỏng): auto_gptq's QuantLinear.forward() tự gọi `self.g_idx.long()` và
    bitwise_right_shift NGAY TRÊN MPS device - 2 op này PyTorch MPS backend
    (đặc biệt trên GPU Intel tích hợp, không phải Apple Silicon) xử lý sai,
    làm hỏng dữ liệu int64 ngay cả sau khi đã ép int32 trước đó (vì
    forward() tự ép lại .long() = int64 bên trong, không kiểm soát được từ
    ngoài nếu gọi qua module.forward()).

    Fix: KHÔNG gọi module.forward(). Tự làm 2 bước bằng torch trần:
    1) Unpack bit (bitwise_right_shift, g_idx indexing) - làm trên CPU vì
       đây là 2 op MPS không hỗ trợ đúng, nhưng cũng RẺ (không phải phần
       tốn thời gian).
    2) Phép tính nặng thật sự - (weight - zero) * scale trên toàn ma trận
       hàng triệu phần tử - chuyển sang MPS chạy thật bằng torch (đây là
       chỗ MPS giúp ích thật, không phải bitwise/indexing).
    """
    assert bits == 4, "Chỉ hỗ trợ GPTQ 4-bit."
    qweight_t = qweight_t.cpu().to(torch.int64)   # bitwise + shift cần CPU
    qzeros_t = qzeros_t.cpu().to(torch.int64)
    g_idx_t = g_idx_t.cpu().to(torch.int64)
    scales_t = scales_t.cpu().to(torch.float32)

    in_features = qweight_t.shape[0] * 8
    out_features = qweight_t.shape[1]
    n_groups = scales_t.shape[0]

    w_int4 = torch.zeros((in_features, out_features), dtype=torch.int64)
    for k in range(8):
        w_int4[k::8, :] = (qweight_t >> (4 * k)) & 0xF

    zeros_int4 = torch.zeros((n_groups, out_features), dtype=torch.int64)
    for k in range(8):
        zeros_int4[:, k::8] = (qzeros_t >> (4 * k)) & 0xF
    # Khớp đúng công thức thật auto_gptq: +1 rồi mask lại mod 16 (wraparound)
    zeros_int4 = (zeros_int4 + 1) & 0xF

    zeros_per_row = zeros_int4[g_idx_t, :]   # index trên CPU - an toàn
    scales_per_row = scales_t[g_idx_t, :]

    # === Phần compute nặng: chuyển sang MPS chạy thật bằng torch ===
    w_f = w_int4.to(torch.float32)
    z_f = zeros_per_row.to(torch.float32)
    if device != "cpu":
        try:
            w_f = w_f.to(device)
            z_f = z_f.to(device)
            scales_per_row = scales_per_row.to(device)
        except Exception as e:
            print(f"  ⚠️ Không chuyển được sang {device} ({e}), tính trên CPU")

    weight = (w_f - z_f) * scales_per_row   # torch thật trên MPS (nếu device=mps)
    weight = weight.t().contiguous().to(torch.float32).cpu()  # [out_features, in_features]
    return weight


def dequantize_gptq_linear(qweight: np.ndarray, qzeros: np.ndarray,
                            scales: np.ndarray, g_idx: np.ndarray,
                            bits: int = 4) -> np.ndarray:
    """
    FIX (v2): khớp CHÍNH XÁC theo source thật của auto_gptq
    QuantLinear.forward (qlinear_cuda_old.py, nhánh bits=4):

        zeros = ((qzeros >> shift) & 0xF)
        zeros = zeros + 1
        zeros = zeros & 0xF        # <-- QUAN TRỌNG: mask LẠI sau khi +1

    Bug ở bản trước: cộng 1 xong KHÔNG mask lại. Khi zero_point gốc = 15
    (0b1111), +1 phải wrap về 0 (mod 16) nhưng bản cũ giữ nguyên 16 -> toàn
    bộ group rơi vào case này bị lệch offset -> weight sai -> model sinh
    token loạn xạ dù mọi thứ khác (loading, RoPE, attention...) đều đúng.
    """
    assert bits == 4, "Chỉ hỗ trợ GPTQ 4-bit."

    in_features = qweight.shape[0] * 8
    out_features = qweight.shape[1]
    n_groups = scales.shape[0]

    qweight_u32 = qweight.astype(np.uint32)
    w_int4 = np.zeros((in_features, out_features), dtype=np.int32)
    for k in range(8):
        w_int4[k::8, :] = (qweight_u32 >> (4 * k)) & 0xF

    qzeros_u32 = qzeros.astype(np.uint32)
    zeros_int4 = np.zeros((n_groups, out_features), dtype=np.int32)
    for k in range(8):
        zeros_int4[:, k::8] = (qzeros_u32 >> (4 * k)) & 0xF

    # FIX: +1 rồi mask lại mod 16 (đúng theo auto_gptq thật), KHÔNG để tràn thành 16
    zeros_int4 = (zeros_int4 + 1) & 0xF

    scales_f32 = scales.astype(np.float32)
    group_for_row = g_idx.astype(np.int64)
    zeros_per_row = zeros_int4[group_for_row, :]
    scales_per_row = scales_f32[group_for_row, :]

    weight = (w_int4.astype(np.float32) - zeros_per_row.astype(np.float32)) * scales_per_row
    # weight hiện có shape [in_features, out_features]; transpose 1 lần duy nhất
    # ở đây để trả về đúng convention nn.Linear: [out_features, in_features].
    return weight.T.astype(np.float32)


def dequantize_gptq_module(module, device: str) -> np.ndarray:
    """
    Dùng ĐÚNG module QuantLinear thật (auto_gptq) để lấy dequantized weight,
    thay vì tự giải mã bit-unpacking - đảm bảo khớp 100% với phép tính
    auto_gptq dùng khi infer thật, tránh mọi rủi ro sai order/convention.
    Module hiện đang ở CPU (float16, do load_model() luôn load CPU để tránh
    lỗi accelerate+MPS+BFloat16 lúc load checkpoint) - tự chuyển RIÊNG module
    này sang device (mps) ngay trước khi tính (lúc này đã là float16 nên MPS
    nhận được), rồi chuyển về lại CPU sau để giải phóng bộ nhớ MPS dần thay
    vì giữ nguyên 224 module cùng lúc trên GPU.
    """
    in_features = getattr(module, "infeatures", None) or module.in_features
    out_features = getattr(module, "outfeatures", None) or module.out_features

    moved = False
    if device != "cpu":
        try:
            # FIX: PyTorch MPS có bug đã biết với tensor int64 - .to("mps")
            # có thể làm hỏng dữ liệu (đây là nguyên nhân g_idx bị hỏng
            # thành giá trị rác như 1087250403 quan sát được). Ép về int32
            # TRƯỚC khi chuyển device - int32 được MPS hỗ trợ ổn định, và
            # giá trị thật của g_idx/qweight/qzeros đều nằm gọn trong phạm
            # vi int32 (g_idx: 0..n_groups~86, qweight/qzeros: packed 4-bit
            # trong uint32 nhưng giá trị thực tế luôn < 2^31) nên không mất
            # thông tin khi ép kiểu.
            for attr_name in ("g_idx", "qweight", "qzeros"):
                if hasattr(module, attr_name):
                    buf = getattr(module, attr_name)
                    if buf is not None and buf.dtype == torch.int64:
                        buf.data = buf.data.to(torch.int32)
            module.to(device)
            moved = True
        except Exception as e:
            print(f"  ⚠️ Không chuyển được module sang {device} ({e}), tính trên CPU")
            device = "cpu"

    eye = torch.eye(in_features, dtype=torch.float16, device=device)

    had_bias = getattr(module, "bias", None) is not None
    saved_bias = None
    if had_bias:
        saved_bias = module.bias.data.clone()
        module.bias.data.zero_()

    with torch.no_grad():
        out = module(eye)  # [in_features, out_features] = W^T (vì y = x @ W^T, x=I)

    if had_bias:
        module.bias.data = saved_bias

    weight = out.t().contiguous().to(torch.float32).cpu().numpy()  # -> [out_features, in_features]
    assert weight.shape == (out_features, in_features), f"shape lệch: {weight.shape} != {(out_features, in_features)}"

    # FIX: nếu MPS command buffer bị driver âm thầm "ignore" (như log lỗi
    # "command buffer exited with error status" ở layer cuối), kết quả đọc
    # về thường là NaN hoặc toàn 0 (buffer chưa từng được ghi). Phát hiện
    # -> tính LẠI layer này bằng CPU (chậm hơn nhưng chắc chắn đúng) thay vì
    # âm thầm ghi weight rác vào file.
    bad = (not np.isfinite(weight).all()) or (np.abs(weight).max() < 1e-12)
    if bad and moved:
        print(f"  ⚠️ Nghi ngờ MPS command buffer lỗi (NaN/toàn-0) - tính lại CPU cho layer này")
        module.to("cpu")
        moved = False
        eye_cpu = torch.eye(in_features, dtype=torch.float16, device="cpu")
        had_bias2 = getattr(module, "bias", None) is not None
        saved_bias2 = None
        if had_bias2:
            saved_bias2 = module.bias.data.clone()
            module.bias.data.zero_()
        with torch.no_grad():
            out = module(eye_cpu)
        if had_bias2:
            module.bias.data = saved_bias2
        weight = out.t().contiguous().to(torch.float32).cpu().numpy()
        del eye_cpu

    if moved:
        # FIX: sync + giải phóng cache GPU NGAY sau mỗi module, TRƯỚC khi
        # chuyển module về CPU. Không làm việc này -> command buffer của
        # module sau chồng lên module trước trong hàng đợi MPS, driver
        # Intel Iris (yếu, không phải Apple Silicon) quá tải sau ~200 lần
        # dồn liên tục -> "command buffer exited with error status" / GPU
        # errors bị driver âm thầm ignore ở các layer cuối -> weight rác.
        torch.mps.synchronize()
        del eye, out
        torch.mps.empty_cache()
        module.to("cpu")  # giải phóng bộ nhớ MPS, không giữ hết 224 module cùng lúc

    return weight


def find_gptq_modules(model):
    """Trả về list (dotted_name, module) cho mọi QuantLinear GPTQ trong model."""
    result = []
    for name, module in model.named_modules():
        if hasattr(module, "qweight") and hasattr(module, "qzeros") and hasattr(module, "scales"):
            result.append((name, module))
    return result


def normalize_key_prefix(name: str) -> str:
    if name.startswith("model.model."):
        return name[len("model."):]
    # FIX: lm_head nằm NGOÀI submodule .model trong LlamaForCausalLM gốc,
    # nên qua AutoGPTQForCausalLM wrapper (self.model = LlamaForCausalLM)
    # nó chỉ bị thêm ĐÚNG 1 lớp "model." (thành "model.lm_head.weight"),
    # không phải 2 lớp như các tensor khác -> không match điều kiện trên,
    # không được strip -> nim_inference.nim tìm "lm_head.weight" không
    # thấy -> âm thầm fallback dùng tied embedding SAI (model này không
    # tie embedding) -> toàn bộ logits cuối sai -> sinh token rác.
    if name.startswith("model.lm_head."):
        return name[len("model."):]
    return name


# ============================================================
# 3. Quantization Functions - VECTORIZED + MPS
# ============================================================

def quantize_int8_vectorized(arr: np.ndarray) -> Tuple[np.ndarray, float]:
    flat = arr.flatten()
    max_val = np.max(np.abs(flat))
    if max_val < 1e-12:
        return np.zeros(flat.shape, dtype=np.int8), 1.0
    scale = max_val / 127.0
    quantized = np.clip(np.round(flat / scale), -127, 127).astype(np.int8)
    return quantized, float(scale)


def quantize_int4_asymmetric_per_group(arr: np.ndarray, group_size: int = 128) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Asymmetric int4 quantization, per-GROUP thay vì per-ROW: mỗi group_size cột
    trong 1 hàng dùng 1 scale/zero_point riêng, thay vì 1 scale cho CẢ hàng
    (per-row cũ). Với group_size=128 (giống độ mịn phổ biến của GPTQ gốc),
    sai số double-quantization (GPTQ 4-bit/group -> float32 -> int4 lai) giảm
    mạnh so với per-row (1 scale/4096 cột) - đó là nguyên nhân chính gây lệch
    hidden state ở after_block_0 khi dùng --q4km trước đây.
    """
    assert arr.ndim == 2
    nRows, nCols = arr.shape
    nGroupsPerRow = (nCols + group_size - 1) // group_size
    scales = np.zeros(nRows * nGroupsPerRow, dtype=np.float32)
    zeros = np.zeros(nRows * nGroupsPerRow, dtype=np.float32)
    nBytesPerRow = (nCols + 1) // 2
    packed = np.zeros((nRows, nBytesPerRow), dtype=np.uint8)

    for r in range(nRows):
        row = arr[r]
        q_row = np.zeros(nCols, dtype=np.uint8)
        for g in range(nGroupsPerRow):
            c0 = g * group_size
            c1 = min(c0 + group_size, nCols)
            chunk = row[c0:c1]
            minVal = float(np.min(chunk))
            maxVal = float(np.max(chunk))
            rng = maxVal - minVal
            scale = rng / 15.0 if rng > 1e-12 else 1.0
            zp = (-minVal / scale) if rng > 1e-12 else 0.0
            q_row[c0:c1] = np.clip(np.round(chunk / scale + zp), 0, 15).astype(np.uint8)
            scales[r * nGroupsPerRow + g] = scale
            zeros[r * nGroupsPerRow + g] = zp
        even = q_row[0::2]
        odd = q_row[1::2]
        packed[r, :len(even)] = even
        packed[r, :len(odd)] |= (odd << 4)

    return packed.flatten(), scales, zeros


def quantize_int4_asymmetric_per_row(arr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-row (per output channel) asymmetric int4 - mỗi hàng (arr.shape[0])
    có scale/zero_point RIÊNG thay vì 1 giá trị chung cho cả tensor.
    Weight sau GPTQ-dequant có range rất khác nhau giữa các hàng (mỗi hàng
    GPTQ vốn đã có scale/group riêng) -> quantize per-tensor (1 scale chung)
    kéo giãn range, sai số cộng dồn nặng khi double-quant. Per-row giữ độ
    chính xác gần với per-group gốc của GPTQ, vẫn ra int4 (~4GB).
    Trả về: packed data [nRows, ceil(nCols/2)] uint8, scales[nRows], zeros[nRows]
    """
    if arr.ndim != 2:
        # fallback: tensor 1D (hiếm khi rơi vào nhánh int4, nhưng an toàn)
        packed, scale, zp = quantize_int4_asymmetric(arr)
        return packed, np.array([scale], dtype=np.float32), np.array([zp], dtype=np.float32)

    nRows, nCols = arr.shape
    scales = np.zeros(nRows, dtype=np.float32)
    zeros = np.zeros(nRows, dtype=np.float32)
    nBytesPerRow = (nCols + 1) // 2
    packed = np.zeros((nRows, nBytesPerRow), dtype=np.uint8)

    for r in range(nRows):
        row = arr[r]
        minVal = float(np.min(row))
        maxVal = float(np.max(row))
        rng = maxVal - minVal
        scale = rng / 15.0 if rng > 1e-12 else 1.0
        zp = (-minVal / scale) if rng > 1e-12 else 0.0
        q = np.clip(np.round(row / scale + zp), 0, 15).astype(np.uint8)
        even = q[0::2]
        odd = q[1::2]
        packed[r, :len(even)] = even
        packed[r, :len(odd)] |= (odd << 4)
        scales[r] = scale
        zeros[r] = zp

    return packed.flatten(), scales, zeros


def quantize_int4_asymmetric(arr: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Asymmetric int4 quantization (Q4_K_M style) - VECTORIZED
    """
    flat = arr.flatten()
    min_val = np.min(flat)
    max_val = np.max(flat)
    if max_val - min_val < 1e-12:
        return np.zeros((len(flat) + 1) // 2, dtype=np.uint8), 1.0, 0.0

    scale = (max_val - min_val) / 15.0
    zero_point = -min_val / scale
    quantized = np.clip(np.round(flat / scale + zero_point), 0, 15).astype(np.uint8)

    # Pack 2 giá trị 4-bit vào 1 byte - VECTORIZED
    n = len(quantized)
    packed = np.zeros((n + 1) // 2, dtype=np.uint8)
    even = quantized[0::2]
    odd = quantized[1::2]
    packed[:len(even)] = even
    packed[:len(odd)] |= (odd << 4)

    return packed, float(scale), float(zero_point)


def quantize_int4_symmetric(arr: np.ndarray) -> Tuple[np.ndarray, float]:
    flat = arr.flatten()
    max_val = np.max(np.abs(flat))
    if max_val < 1e-12:
        return np.zeros((len(flat) + 1) // 2, dtype=np.uint8), 1.0
    scale = max_val / 7.0
    quantized = np.clip(np.round(flat / scale), -7, 7).astype(np.int8)

    # Pack 2 giá trị 4-bit vào 1 byte - VECTORIZED
    n = len(quantized)
    packed = np.zeros((n + 1) // 2, dtype=np.uint8)
    quantized_u8 = (quantized & 0x0F).astype(np.uint8)
    even = quantized_u8[0::2]
    odd = quantized_u8[1::2]
    packed[:len(even)] = even
    packed[:len(odd)] |= (odd << 4)

    return packed, float(scale)


def quantize_with_mps(arr: np.ndarray, quant_kind: int) -> Tuple[np.ndarray, float, float]:
    """
    Dùng MPS (Metal) để quantize nếu có GPU - nhanh hơn 10-50x
    """
    if not USE_MPS:
        # Fallback về CPU vectorized
        if quant_kind == QK_INT8:
            quantized, scale = quantize_int8_vectorized(arr)
            return quantized, scale, 0.0
        elif quant_kind == QK_INT4:
            quantized, scale = quantize_int4_symmetric(arr)
            return quantized, scale, 0.0
        elif quant_kind == QK_INT4_ASYMMETRIC:
            quantized, scale, zero_point = quantize_int4_asymmetric(arr)
            return quantized, scale, zero_point
        else:
            return arr.flatten().astype(np.float32), 1.0, 0.0

    try:
        # === FIX: chuyển sang FP16 trước khi lên MPS ===
        arr_tensor = torch.from_numpy(arr.flatten()).float().to("mps")
        max_val = torch.max(torch.abs(arr_tensor)).item()

        if max_val < 1e-12:
            if quant_kind == QK_INT8:
                return np.zeros(arr_tensor.shape, dtype=np.int8), 1.0, 0.0
            elif quant_kind in (QK_INT4, QK_INT4_ASYMMETRIC):
                return np.zeros((len(arr_tensor) + 1) // 2, dtype=np.uint8), 1.0, 0.0
            else:
                return arr.flatten().astype(np.float32), 1.0, 0.0

        if quant_kind == QK_INT8:
            scale = max_val / 127.0
            quantized = torch.clamp(torch.round(arr_tensor / scale), -127, 127).to(torch.int8)
            result = quantized.cpu().numpy()
            return result, float(scale), 0.0

        elif quant_kind == QK_INT4_ASYMMETRIC:
            min_val = torch.min(arr_tensor).item()
            max_val = torch.max(arr_tensor).item()
            scale = (max_val - min_val) / 15.0
            zero_point = -min_val / scale
            quantized = torch.clamp(torch.round(arr_tensor / scale + zero_point), 0, 15).to(torch.uint8)
            quantized_cpu = quantized.cpu().numpy()
            return quantize_int4_asymmetric(quantized_cpu)

        elif quant_kind == QK_INT4:
            scale = max_val / 7.0
            quantized = torch.clamp(torch.round(arr_tensor / scale), -7, 7).to(torch.int8)
            quantized_cpu = quantized.cpu().numpy()
            return quantize_int4_symmetric(quantized_cpu)

        else:
            return arr.flatten().astype(np.float32), 1.0, 0.0

    except Exception as e:
        print(f"  ⚠️ MPS quantize failed, fallback to CPU: {e}")
        if quant_kind == QK_INT8:
            quantized, scale = quantize_int8_vectorized(arr)
            return quantized, scale, 0.0
        elif quant_kind == QK_INT4_ASYMMETRIC:
            quantized, scale, zero_point = quantize_int4_asymmetric(arr)
            return quantized, scale, zero_point
        elif quant_kind == QK_INT4:
            quantized, scale = quantize_int4_symmetric(arr)
            return quantized, scale, 0.0
        else:
            return arr.flatten().astype(np.float32), 1.0, 0.0


INT4_ASYMMETRIC_GROUP_SIZE = 128  ## THÊM: mac dinh group=128 (gan voi granularity GPTQ goc).
                                   ## Doi so nay neu can nen nhe hon nua (group lon hon) hoac
                                   ## chinh xac hon (group nho hon), danh doi voi dung luong.

def pack_tensor_for_nim(arr: np.ndarray, quant_kind: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """
    Pack tensor - dùng MPS nếu có. scales/zeros luôn trả về dạng mảng
    (len=1 cho các kind cũ per-tensor, len=nRows cho int4-asymmetric 2D
    per-row, len=nRows*nGroupsPerRow cho per-group). Tra ve them group_size
    (0 = per-tensor/per-row, >0 = per-group voi group_size do).
    """
    if quant_kind == QK_FP32_RAW:
        return arr.flatten().astype(np.float32), np.array([1.0], dtype=np.float32), np.array([0.0], dtype=np.float32), 0, 0, 0

    elif quant_kind == QK_INT8:
        quantized, scale, _ = quantize_with_mps(arr, QK_INT8)
        return quantized, np.array([scale], dtype=np.float32), np.array([0.0], dtype=np.float32), 0, 0, 0

    elif quant_kind == QK_INT4:
        quantized, scale, _ = quantize_with_mps(arr, QK_INT4)
        return quantized, np.array([scale], dtype=np.float32), np.array([0.0], dtype=np.float32), 0, 0, 0

    elif quant_kind == QK_INT4_ASYMMETRIC:
        if arr.ndim == 2:
            # SUA: per-group (mac dinh 128) thay vi per-row - per-row (1 scale/
            # 4096 cot) qua tho so voi GPTQ goc (per-group ~128 cot/scale) ->
            # double-quantization mat qua nhieu do chinh xac -> hidden state
            # lech ro sau block 0 du dequant GPTQ va cong thuc forward deu dung.
            group_size = min(INT4_ASYMMETRIC_GROUP_SIZE, arr.shape[1])
            packed, scales, zeros = quantize_int4_asymmetric_per_group(arr, group_size)
            return packed, scales, zeros, 0, 0, group_size
        quantized, scale, zero_point = quantize_with_mps(arr, QK_INT4_ASYMMETRIC)
        return quantized, np.array([scale], dtype=np.float32), np.array([zero_point], dtype=np.float32), 0, 0, 0

    else:
        raise ValueError(f"Unsupported quant_kind: {quant_kind}")


# ============================================================
# 4. Write Functions
# ============================================================

def write_string(f, s: str):
    encoded = s.encode('utf-8')
    f.write(struct.pack('<i', len(encoded)))
    f.write(encoded)


def write_quant_tensor(f, name: str, arr: np.ndarray, kind: int = QK_FP32_RAW):
    """Ghi 1 QuantTensor - hỗ trợ per-row/per-group scale/zero_point (mảng, không phải 1 float)"""
    data, scales, zeros, eb, mb, group_size = pack_tensor_for_nim(arr, kind)

    actual_kind = {QK_FP32_RAW: 0, QK_INT8: 1, QK_INT4: 2, QK_INT4_ASYMMETRIC: 3}.get(kind, 0)
    scales = np.asarray(scales, dtype=np.float32).flatten()
    zeros = np.asarray(zeros, dtype=np.float32).flatten()

    f.write(struct.pack('<B', actual_kind))
    f.write(struct.pack('<i', len(arr.shape)))
    for d in arr.shape:
        f.write(struct.pack('<i', d))
    f.write(struct.pack('<i', len(scales)))   # nScales (1=per-tensor, nRows=per-row, nRows*nGroups=per-group)
    f.write(scales.tobytes())
    f.write(zeros.tobytes())
    f.write(struct.pack('<i', group_size))    # THÊM: 0=per-tensor/per-row (cu), >0=per-group
    f.write(struct.pack('<i', eb))
    f.write(struct.pack('<i', mb))
    f.write(struct.pack('<i', data.nbytes))
    f.write(data.tobytes())


def find_gptq_bases(state_dict: Dict) -> Set[str]:
    bases = set()
    for name in state_dict.keys():
        for suf in (".qweight", ".qzeros", ".scales", ".g_idx"):
            if name.endswith(suf):
                bases.add(name[: -len(suf)])
                break
    return bases


# ============================================================
# 5. Main Export
# ============================================================

def export_to_nimq(
    model_name: str,
    output_path: str,
    is_gptq: bool = False,
    quant_kind: int = QK_INT8,
):
    start_time = time.time()

    print("Loading model...")
    model, tokenizer, config = load_model(model_name, is_gptq)

    arch = [
        config["vocab_size"],
        config["hidden_size"],
        config["num_attention_heads"],
        config["num_hidden_layers"],
        config["intermediate_size"] // config["hidden_size"],
    ]

    print(f"Arch: {arch}")
    quant_names = {0: "FP32", 1: "INT8", 2: "INT4", 3: "INT4_ASYMMETRIC"}
    print(f"Quantization: {quant_names.get(quant_kind, 'UNKNOWN')}")
    if USE_MPS:
        print("🚀 Using MPS (Metal GPU) for quantization - 10-50x faster")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    state_dict = model.state_dict()
    gptq_bases = find_gptq_bases(state_dict)
    is_gptq_model = len(gptq_bases) > 0

    device = "mps" if USE_MPS else "cpu"
    if is_gptq_model:
        print(f"🔍 Found {len(gptq_bases)} GPTQ layers (torch + {device}, KHÔNG dùng numpy để tính)")

    normal_tensors = []
    for name, tensor in state_dict.items():
        if "num_batches_tracked" in name:
            continue
        if "model.norm" in name and "weight" not in name:
            continue
        if any(name.endswith(suf) for suf in (".qweight", ".qzeros", ".scales", ".g_idx")):
            continue
        # FIX: bias của các base GPTQ (vd model.layers.0.self_attn.q_proj.bias)
        # LÀ 1 tensor bình thường trong state_dict (không có suffix qweight/
        # qzeros/scales/g_idx) nên trước đây lọt qua các điều kiện loại trừ ở
        # trên, bị ghi vào file Ở ĐÂY với kind=quant_kind (bias bị quantize
        # int4 THEO Ý MUỐN --q4km, dù bias đáng lẽ luôn phải fp32) - RỒI ghi
        # LẠI LẦN NỮA đúng cách (QK_FP32_RAW) trong loop GPTQ bên dưới. Tensor
        # bị ghi 2 lần trùng tên trong file - lãng phí thời gian quantize +
        # dung lượng, và việc load ra ĐÚNG hay không phụ thuộc hoàn toàn vào
        # thứ tự 2 loop này (byName[name]=qt ở phía Nim ghi đè theo thứ tự
        # đọc file - đúng MAY MẮN vì loop GPTQ chạy sau, không phải vì được
        # thiết kế đảm bảo vậy). Bỏ qua hẳn ở đây, để riêng loop GPTQ ghi 1
        # lần duy nhất, đúng ngay từ đầu.
        if is_gptq_model and any(name == base + ".bias" for base in gptq_bases):
            continue
        normal_tensors.append(name)

    total_tensors = len(normal_tensors)
    if is_gptq_model:
        total_tensors += len(gptq_bases)
        for base in gptq_bases:
            if base + ".bias" in state_dict:
                total_tensors += 1

    print(f"Total tensors: {total_tensors}")

    with open(output_path, 'wb') as f:
        write_string(f, "NIMQ2")  # THÊM: bump tu NIMQ1 -> NIMQ2 (them field groupSize, xem quant.nim)
        for v in arch:
            f.write(struct.pack('<i', v))
        f.write(struct.pack('<i', total_tensors))

        # Xử lý tensor thường
        tensor_count = 0
        for name in normal_tensors:
            tensor = state_dict[name]
            tensor_count += 1
            if tensor_count % 20 == 0:
                print(f"  Writing {tensor_count}/{len(normal_tensors)}: {name}")

            norm_name = normalize_key_prefix(name)
            if USE_MPS:
                tensor = tensor.cpu()
            arr = tensor.detach().to(torch.float16).numpy()

            kind = quant_kind
            if "embed" in name.lower() or "lm_head" in name.lower() or "norm" in name.lower():
                kind = QK_FP32_RAW

            write_string(f, norm_name)
            write_quant_tensor(f, norm_name, arr, kind)
            del arr

        # Xử lý GPTQ: torch thuần (KHÔNG numpy) - unpack bit (bitwise/int64
        # index, MPS lỗi) trên CPU, phần compute nặng (trừ+nhân toàn ma
        # trận) chạy THẬT trên MPS bằng torch. Công thức khớp đúng auto_gptq
        # thật (zero+1 rồi mask lại mod 16).
        if is_gptq_model:
            print(f"  Processing {len(gptq_bases)} GPTQ layers (torch + {device})...")
            for base in gptq_bases:
                qw_key = base + ".qweight"
                qz_key = base + ".qzeros"
                sc_key = base + ".scales"
                gi_key = base + ".g_idx"

                if not all(k in state_dict for k in (qw_key, qz_key, sc_key, gi_key)):
                    continue

                qweight_t = state_dict[qw_key].detach()
                qzeros_t = state_dict[qz_key].detach()
                scales_t = state_dict[sc_key].detach()
                g_idx_t = state_dict[gi_key].detach()

                weight_t = dequantize_gptq_torch_mps(qweight_t, qzeros_t, scales_t, g_idx_t, device, bits=4)
                weight_f32 = weight_t.numpy()  # chỉ convert numpy ở bước ghi file (write_quant_tensor cần numpy để pack binary)

                norm_name = normalize_key_prefix(base + ".weight")
                write_string(f, norm_name)
                write_quant_tensor(f, norm_name, weight_f32, quant_kind)

                bias_key = base + ".bias"
                if bias_key in state_dict:
                    bias = state_dict[bias_key].detach().to(torch.float16).cpu().numpy().astype(np.float32)
                    norm_bias = normalize_key_prefix(bias_key)
                    write_string(f, norm_bias)
                    write_quant_tensor(f, norm_bias, bias, QK_FP32_RAW)

                del weight_f32, weight_t, qweight_t, qzeros_t, scales_t, g_idx_t


    elapsed = time.time() - start_time
    file_size = os.path.getsize(output_path) / (1024**3)
    print(f"✅ Exported to {output_path}")
    print(f"   Arch: {arch}")
    print(f"   Tensors: {total_tensors}")
    print(f"   File size: {file_size:.2f} GB")
    print(f"   Time: {elapsed:.1f}s")

    tokenizer_path = os.path.join(os.path.dirname(output_path) or ".", "tokenizer.json")
    vocab_bytes = list(range(min(256, config["vocab_size"])))
    with open(tokenizer_path, 'w') as g:
        json.dump({"vocab_size": min(256, config["vocab_size"]), "bytes": vocab_bytes}, g)
    print(f"   Tokenizer saved to {tokenizer_path}")

    config_path = os.path.join(os.path.dirname(output_path) or ".", "config.nim.json")
    with open(config_path, 'w') as g:
        json.dump(config, g, indent=2)
    print(f"   Config saved to {config_path}")


# ============================================================
# 6. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Export transformers model to .nimq with quantization")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--output", default="model.nimq", help="Output .nimq file")
    parser.add_argument("--gptq", action="store_true", help="Model is GPTQ")
    parser.add_argument("--fp16", action="store_true", help="Store as FP16 (no quantization)")
    parser.add_argument("--int8", action="store_true", help="Quantize to int8 (recommended)")
    parser.add_argument("--int4", action="store_true", help="Quantize to int4 (symmetric)")
    parser.add_argument("--q4km", action="store_true", help="Quantize to int4 asymmetric (Q4_K_M style, better quality)")
    parser.add_argument("--group-size", type=int, default=128,
                         help="Group size cho --q4km (mac dinh 128, giong granularity GPTQ goc). "
                              "Nho hon -> chinh xac hon nhung file to hon; lon hon -> nguoc lai.")
    parser.add_argument("--no-mps", action="store_true", help="Disable MPS (use CPU only)")

    args = parser.parse_args()

    # Override MPS if disabled
    global USE_MPS
    if args.no_mps:
        USE_MPS = False
        print("🔹 MPS disabled by user")

    if args.q4km:
        quant_kind = QK_INT4_ASYMMETRIC
        print("🔹 Using INT4 ASYMMETRIC quantization (Q4_K_M style - better quality)")
    elif args.int4:
        quant_kind = QK_INT4
        print("🔹 Using INT4 SYMMETRIC quantization (faster but lower quality)")
    elif args.int8:
        quant_kind = QK_INT8
        print("🔹 Using INT8 quantization (balanced)")
    elif args.fp16:
        quant_kind = QK_FP32_RAW
        print("🔹 Using FP16 (no quantization)")
    else:
        quant_kind = QK_INT8
        print("🔹 Default: INT8 quantization (use --q4km for better quality)")

    is_gptq = args.gptq
    if not is_gptq:
        try:
            config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
            if hasattr(config, "quantization_config") and config.quantization_config is not None:
                if config.quantization_config.get("quant_method") == "gptq":
                    is_gptq = True
                    print("🔍 Auto-detected GPTQ model")
        except:
            pass

    global INT4_ASYMMETRIC_GROUP_SIZE
    INT4_ASYMMETRIC_GROUP_SIZE = args.group_size
    export_to_nimq(args.model, args.output, is_gptq=is_gptq, quant_kind=quant_kind)

if __name__ == "__main__":
    main()