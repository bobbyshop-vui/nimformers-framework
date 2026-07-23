#!/usr/bin/env python3
"""
export_to_nimq.py - Hỗ trợ asymmetric int4 (Q4_K_M) với MPS / CUDA / ROCm acceleration

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
# Kiểm tra device: CUDA > ROCm > MPS > CPU
# ============================================================
USE_MPS = False
USE_CUDA = False   # bao gồm cả ROCm (torch.cuda API dùng chung)
GPU_DEVICE = "cpu"
GPU_DEVICE_NAME = "CPU"

if torch.cuda.is_available():
    # CUDA (NVIDIA) hoặc ROCm (AMD) - cả hai đều dùng torch.cuda API
    USE_CUDA = True
    GPU_DEVICE = "cuda"
    device_name = torch.cuda.get_device_name(0)
    # ROCm thường có "AMD" hoặc "Radeon" trong tên; CUDA có "NVIDIA" / "Tesla" / "A100"...
    if any(kw in device_name for kw in ("AMD", "Radeon", "gfx")):
        GPU_DEVICE_NAME = f"ROCm/HIP ({device_name})"
        print(f"🔹 ROCm (AMD GPU) available - sẽ dùng GPU cho quantize (FP16): {device_name}")
    else:
        GPU_DEVICE_NAME = f"CUDA ({device_name})"
        print(f"🔹 CUDA (NVIDIA GPU) available - sẽ dùng GPU cho quantize (FP16): {device_name}")
    torch.set_default_dtype(torch.float16)
elif torch.backends.mps.is_available():
    # Apple Metal (MPS) - chỉ khi không có CUDA/ROCm
    USE_MPS = True
    GPU_DEVICE = "mps"
    GPU_DEVICE_NAME = "MPS (Apple Metal)"
    torch.set_default_dtype(torch.float16)
    print("🔹 MPS (Apple Metal) available - sẽ dùng GPU cho quantize (FP16)")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
else:
    GPU_DEVICE = "cpu"
    GPU_DEVICE_NAME = "CPU"
    print("🔹 Không tìm thấy GPU (CUDA/ROCm/MPS) - dùng CPU")

# Alias cho code cũ vẫn tham chiếu USE_MPS
# USE_CUDA thay thế vai trò USE_MPS trong các nhánh CUDA/ROCm
def _has_gpu() -> bool:
    return USE_CUDA or USE_MPS

# ============================================================
# QuantKind
# ============================================================
QK_FP32_RAW = 0
QK_INT8 = 1
QK_INT4 = 2
QK_INT4_ASYMMETRIC = 3


# ============================================================
# 1. Load model
# ============================================================

def load_model(model_name: str, is_gptq: bool = False) -> Tuple[Any, Any, Dict]:
    print(f"Loading model: {model_name}")

    # Luôn load CPU trước. CUDA/ROCm: accelerate có thể load thẳng cuda,
    # nhưng để tránh OOM và nhất quán với MPS (bfloat16 issue), load CPU
    # trước rồi chuyển từng module khi cần - an toàn hơn trên mọi backend.
    device = "cpu"
    print(f"   Using device: {device} (module GPTQ sẽ tự chuyển {GPU_DEVICE_NAME} lúc dequantize)")

    torch_dtype = torch.float16

    if is_gptq:
        try:
            from auto_gptq import AutoGPTQForCausalLM
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
                torch_dtype=torch_dtype,
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
# 2. GPU sync / cache helpers (trừu tượng hóa MPS vs CUDA/ROCm)
# ============================================================

def gpu_synchronize():
    """Đồng bộ GPU - hoạt động với CUDA, ROCm và MPS."""
    if USE_CUDA:
        torch.cuda.synchronize()
    elif USE_MPS:
        torch.mps.synchronize()

def gpu_empty_cache():
    """Xóa cache GPU - hoạt động với CUDA, ROCm và MPS."""
    if USE_CUDA:
        torch.cuda.empty_cache()
    elif USE_MPS:
        torch.mps.empty_cache()


# ============================================================
# 3. GPTQ Dequantization
# ============================================================

def dequantize_gptq_torch_gpu(qweight_t: torch.Tensor, qzeros_t: torch.Tensor,
                               scales_t: torch.Tensor, g_idx_t: torch.Tensor,
                               device: str, bits: int = 4) -> torch.Tensor:
    """
    Dequantize GPTQ 4-bit bằng torch thuần - hỗ trợ CUDA, ROCm và MPS.

    Chiến lược:
    - Bitwise (>> & | int64 indexing): làm trên CPU vì MPS không ổn định với
      int64 bitwise, còn CUDA/ROCm thì ổn nhưng để nhất quán ta để CPU.
    - Phần tính toán nặng (w - z) * scale trên ma trận lớn: đẩy sang GPU
      (CUDA/ROCm/MPS) bằng torch - đây là chỗ GPU thật sự giúp ích.

    Note: với CUDA/ROCm thực ra có thể làm toàn bộ trên GPU vì int64 bitwise
    được hỗ trợ đầy đủ, nhưng giữ CPU cho bước unpack để code đơn giản và
    đảm bảo correctness trên mọi backend kể cả MPS.
    """
    assert bits == 4, "Chỉ hỗ trợ GPTQ 4-bit."
    qweight_t = qweight_t.cpu().to(torch.int64)
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
    zeros_int4 = (zeros_int4 + 1) & 0xF

    zeros_per_row = zeros_int4[g_idx_t, :]
    scales_per_row = scales_t[g_idx_t, :]

    # Phần compute nặng: chuyển sang GPU
    w_f = w_int4.to(torch.float32)
    z_f = zeros_per_row.to(torch.float32)
    if device != "cpu":
        try:
            w_f = w_f.to(device)
            z_f = z_f.to(device)
            scales_per_row = scales_per_row.to(device)
        except Exception as e:
            print(f"  ⚠️ Không chuyển được sang {device} ({e}), tính trên CPU")

    weight = (w_f - z_f) * scales_per_row
    weight = weight.t().contiguous().to(torch.float32).cpu()  # [out_features, in_features]
    return weight


# Alias ngược để không vỡ code cũ gọi tên hàm MPS
dequantize_gptq_torch_mps = dequantize_gptq_torch_gpu


def dequantize_gptq_linear(qweight: np.ndarray, qzeros: np.ndarray,
                            scales: np.ndarray, g_idx: np.ndarray,
                            bits: int = 4) -> np.ndarray:
    """
    FIX (v2): khớp CHÍNH XÁC theo source thật của auto_gptq
    QuantLinear.forward (qlinear_cuda_old.py, nhánh bits=4):

        zeros = ((qzeros >> shift) & 0xF)
        zeros = zeros + 1
        zeros = zeros & 0xF        # <-- QUAN TRỌNG: mask LẠI sau khi +1
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

    zeros_int4 = (zeros_int4 + 1) & 0xF

    scales_f32 = scales.astype(np.float32)
    group_for_row = g_idx.astype(np.int64)
    zeros_per_row = zeros_int4[group_for_row, :]
    scales_per_row = scales_f32[group_for_row, :]

    weight = (w_int4.astype(np.float32) - zeros_per_row.astype(np.float32)) * scales_per_row
    return weight.T.astype(np.float32)


def dequantize_gptq_module(module, device: str) -> np.ndarray:
    """
    Dùng module QuantLinear thật (auto_gptq) để lấy dequantized weight.
    Hỗ trợ CUDA, ROCm và MPS.

    CUDA/ROCm: int64 tensor được hỗ trợ ổn định, không cần ép int32 như MPS.
    MPS: vẫn ép int32 trước khi chuyển device vì MPS có bug với int64.
    """
    in_features = getattr(module, "infeatures", None) or module.in_features
    out_features = getattr(module, "outfeatures", None) or module.out_features

    moved = False
    if device != "cpu":
        try:
            if USE_MPS:
                # MPS bug: int64 tensor bị hỏng -> ép int32 trước
                for attr_name in ("g_idx", "qweight", "qzeros"):
                    if hasattr(module, attr_name):
                        buf = getattr(module, attr_name)
                        if buf is not None and buf.dtype == torch.int64:
                            buf.data = buf.data.to(torch.int32)
            elif USE_CUDA:
                # CUDA/ROCm: int64 ổn, không cần ép kiểu
                pass
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
        out = module(eye)  # [in_features, out_features]

    if had_bias:
        module.bias.data = saved_bias

    weight = out.t().contiguous().to(torch.float32).cpu().numpy()  # [out_features, in_features]
    assert weight.shape == (out_features, in_features), f"shape lệch: {weight.shape} != {(out_features, in_features)}"

    # Phát hiện kết quả lỗi (NaN / toàn 0) - đặc biệt hay xảy ra với MPS
    bad = (not np.isfinite(weight).all()) or (np.abs(weight).max() < 1e-12)
    if bad and moved:
        print(f"  ⚠️ Nghi ngờ GPU command buffer lỗi (NaN/toàn-0) - tính lại CPU cho layer này")
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
        gpu_synchronize()
        del eye, out
        gpu_empty_cache()
        module.to("cpu")

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
    if name.startswith("model.lm_head."):
        return name[len("model."):]
    return name


# ============================================================
# 4. Quantization Functions - VECTORIZED + GPU
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
    Asymmetric int4 quantization per-group (mặc định group_size=128, khớp GPTQ gốc).
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
    if arr.ndim != 2:
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
    """Asymmetric int4 quantization (Q4_K_M style) - VECTORIZED"""
    flat = arr.flatten()
    min_val = np.min(flat)
    max_val = np.max(flat)
    if max_val - min_val < 1e-12:
        return np.zeros((len(flat) + 1) // 2, dtype=np.uint8), 1.0, 0.0

    scale = (max_val - min_val) / 15.0
    zero_point = -min_val / scale
    quantized = np.clip(np.round(flat / scale + zero_point), 0, 15).astype(np.uint8)

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

    n = len(quantized)
    packed = np.zeros((n + 1) // 2, dtype=np.uint8)
    quantized_u8 = (quantized & 0x0F).astype(np.uint8)
    even = quantized_u8[0::2]
    odd = quantized_u8[1::2]
    packed[:len(even)] = even
    packed[:len(odd)] |= (odd << 4)

    return packed, float(scale)


def quantize_with_gpu(arr: np.ndarray, quant_kind: int) -> Tuple[np.ndarray, float, float]:
    """
    Quantize dùng GPU (CUDA / ROCm / MPS) nếu có. Fallback về CPU nếu không.

    CUDA/ROCm: torch.cuda - ổn định, hỗ trợ float16/float32 đầy đủ.
    MPS: torch.mps - ổn với float16/float32, tránh int64 bitwise.
    CPU: numpy vectorized - fallback an toàn.
    """
    if not _has_gpu():
        # Fallback CPU vectorized
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
        arr_tensor = torch.from_numpy(arr.flatten()).float().to(GPU_DEVICE)
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
            max_val_v = torch.max(arr_tensor).item()
            scale = (max_val_v - min_val) / 15.0
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
        print(f"  ⚠️ GPU quantize failed ({GPU_DEVICE_NAME}), fallback to CPU: {e}")
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


# Alias ngược để không vỡ code cũ tham chiếu quantize_with_mps
quantize_with_mps = quantize_with_gpu


INT4_ASYMMETRIC_GROUP_SIZE = 128


def pack_tensor_for_nim(arr: np.ndarray, quant_kind: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """
    Pack tensor - dùng GPU nếu có (CUDA/ROCm/MPS).
    Trả về (data, scales, zeros, eb, mb, group_size).
    """
    if quant_kind == QK_FP32_RAW:
        return arr.flatten().astype(np.float32), np.array([1.0], dtype=np.float32), np.array([0.0], dtype=np.float32), 0, 0, 0

    elif quant_kind == QK_INT8:
        quantized, scale, _ = quantize_with_gpu(arr, QK_INT8)
        return quantized, np.array([scale], dtype=np.float32), np.array([0.0], dtype=np.float32), 0, 0, 0

    elif quant_kind == QK_INT4:
        quantized, scale, _ = quantize_with_gpu(arr, QK_INT4)
        return quantized, np.array([scale], dtype=np.float32), np.array([0.0], dtype=np.float32), 0, 0, 0

    elif quant_kind == QK_INT4_ASYMMETRIC:
        if arr.ndim == 2:
            group_size = min(INT4_ASYMMETRIC_GROUP_SIZE, arr.shape[1])
            packed, scales, zeros = quantize_int4_asymmetric_per_group(arr, group_size)
            return packed, scales, zeros, 0, 0, group_size
        quantized, scale, zero_point = quantize_with_gpu(arr, QK_INT4_ASYMMETRIC)
        return quantized, np.array([scale], dtype=np.float32), np.array([zero_point], dtype=np.float32), 0, 0, 0

    else:
        raise ValueError(f"Unsupported quant_kind: {quant_kind}")


# ============================================================
# 5. Write Functions
# ============================================================

def write_string(f, s: str):
    encoded = s.encode('utf-8')
    f.write(struct.pack('<i', len(encoded)))
    f.write(encoded)


def write_quant_tensor(f, name: str, arr: np.ndarray, kind: int = QK_FP32_RAW):
    """Ghi 1 QuantTensor - hỗ trợ per-row/per-group scale/zero_point."""
    data, scales, zeros, eb, mb, group_size = pack_tensor_for_nim(arr, kind)

    actual_kind = {QK_FP32_RAW: 0, QK_INT8: 1, QK_INT4: 2, QK_INT4_ASYMMETRIC: 3}.get(kind, 0)
    scales = np.asarray(scales, dtype=np.float32).flatten()
    zeros = np.asarray(zeros, dtype=np.float32).flatten()

    f.write(struct.pack('<B', actual_kind))
    f.write(struct.pack('<i', len(arr.shape)))
    for d in arr.shape:
        f.write(struct.pack('<i', d))
    f.write(struct.pack('<i', len(scales)))
    f.write(scales.tobytes())
    f.write(zeros.tobytes())
    f.write(struct.pack('<i', group_size))
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
# 6. Main Export
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
    if _has_gpu():
        print(f"🚀 Using {GPU_DEVICE_NAME} for quantization - 10-50x faster")

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    state_dict = model.state_dict()
    gptq_bases = find_gptq_bases(state_dict)
    is_gptq_model = len(gptq_bases) > 0

    device = GPU_DEVICE  # "cuda" / "mps" / "cpu"
    if is_gptq_model:
        print(f"🔍 Found {len(gptq_bases)} GPTQ layers (torch + {GPU_DEVICE_NAME})")

    normal_tensors = []
    for name, tensor in state_dict.items():
        if "num_batches_tracked" in name:
            continue
        if "model.norm" in name and "weight" not in name:
            continue
        if any(name.endswith(suf) for suf in (".qweight", ".qzeros", ".scales", ".g_idx")):
            continue
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
        write_string(f, "NIMQ2")
        for v in arch:
            f.write(struct.pack('<i', v))
        f.write(struct.pack('<i', total_tensors))

        # Tensor thường
        tensor_count = 0
        for name in normal_tensors:
            tensor = state_dict[name]
            tensor_count += 1
            if tensor_count % 20 == 0:
                print(f"  Writing {tensor_count}/{len(normal_tensors)}: {name}")

            norm_name = normalize_key_prefix(name)
            arr = tensor.detach().cpu().to(torch.float16).numpy()

            kind = quant_kind
            if "embed" in name.lower() or "lm_head" in name.lower() or "norm" in name.lower():
                kind = QK_FP32_RAW

            write_string(f, norm_name)
            write_quant_tensor(f, norm_name, arr, kind)
            del arr

        # GPTQ layers
        if is_gptq_model:
            print(f"  Processing {len(gptq_bases)} GPTQ layers ({GPU_DEVICE_NAME})...")
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

                weight_t = dequantize_gptq_torch_gpu(qweight_t, qzeros_t, scales_t, g_idx_t, device, bits=4)
                weight_f32 = weight_t.numpy()

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
# 7. CLI
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
                         help="Group size cho --q4km (mặc định 128, giống granularity GPTQ gốc).")
    parser.add_argument("--no-gpu", action="store_true",
                         help="Disable tất cả GPU acceleration (CUDA/ROCm/MPS), dùng CPU")
    # Alias cũ cho người quen --no-mps
    parser.add_argument("--no-mps", action="store_true",
                         help="(deprecated) Giống --no-gpu, giữ để tương thích ngược")

    args = parser.parse_args()

    global USE_MPS, USE_CUDA, GPU_DEVICE, GPU_DEVICE_NAME
    if args.no_gpu or args.no_mps:
        USE_CUDA = False
        USE_MPS = False
        GPU_DEVICE = "cpu"
        GPU_DEVICE_NAME = "CPU"
        print("🔹 GPU acceleration disabled by user - dùng CPU")

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
