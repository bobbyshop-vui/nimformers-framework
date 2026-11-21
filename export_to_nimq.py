#!/usr/bin/env python3
"""
Export Hugging Face model weights to Nimformers NIMQ2 tensors.

Supports FP16, BF16 and FP32 source checkpoints, with raw FP32, INT8, symmetric
INT4, grouped asymmetric Q4_K_M-style INT4, and the Nimformer FP4-KM layout.

FIX (2026-07-17): removed the extra transpose (weight_f32.T) after calling
dequantize_gptq_linear() in export_to_nimq(). dequantize_gptq_linear
already transposes once to return the right layout [out_features, in_features]
(the nn.Linear/HF convention). Transposing again at the call site flipped the layout
into [in_features, out_features], so every Linear layer loaded
from GPTQ (q/k/v/o_proj, gate/up/down_proj) was loaded completely wrong on the Nim side,
because loadLinear() in generate.nim assigns .data directly without transposing (as
it assumes the [outF, inF] layout).
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
# QuantKind
# ============================================================
QK_FP32_RAW = 0
QK_INT8 = 1
QK_INT4 = 2
QK_INT4_ASYMMETRIC = 3
QK_FP4KM = 8  # QuantKind.qkFp4KM in quant.nim
FP4KM_GROUP_SIZE = 32
DEFAULT_Q4KM_GROUP_SIZE = 128
QUANT_DEVICE = "cpu"


def _backend_available(name: str) -> bool:
    try:
        if name == "mps":
            return bool(torch.backends.mps.is_available())
        backend = getattr(torch, name, None)
        is_available = getattr(backend, "is_available", None)
        return bool(is_available()) if callable(is_available) else False
    except Exception:
        return False


def resolve_quant_device(requested: str = "auto") -> str:
    """Resolve a PyTorch device, including ROCm's CUDA-compatible device API."""
    requested = str(requested).strip().lower()
    if requested == "auto":
        candidates = ["cuda", "mps", "xpu", "hpu", "npu", "mlu", "musa"]
        for candidate in candidates:
            if _backend_available(candidate):
                try:
                    torch.empty(0, device=candidate)
                    return candidate
                except Exception:
                    continue
        try:
            private_backend = torch._C._get_privateuse1_backend_name()
            if private_backend and private_backend != "privateuseone":
                torch.empty(0, device=private_backend)
                return private_backend
        except Exception:
            pass
        return "cpu"

    if requested == "rocm":
        if not getattr(torch.version, "hip", None):
            raise ValueError("--quant-device rocm requires a PyTorch build with ROCm support")
        requested = "cuda"
    try:
        torch.empty(0, device=requested)
    except Exception as exc:
        raise ValueError(f"PyTorch quantization device {requested!r} is unavailable: {exc}") from exc
    return requested


# ============================================================
# 1. Load model
# ============================================================

def load_model(model_name: str, is_gptq: bool = False,
               source_dtype: str = "auto") -> Tuple[Any, Any, Dict]:
    print(f"Loading model: {model_name}")

    # Load on CPU. Quantization is moved to the selected PyTorch device later.
    device = "cpu"
    print(f"   Using device: {device} (quantization device: {QUANT_DEVICE})")

    dtype_map = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    torch_dtype = dtype_map.get(source_dtype, "auto")

    if is_gptq:
        try:
            from auto_gptq import AutoGPTQForCausalLM
            # AutoGPTQ's packed weights are dequantized through its FP16 path
            # when the caller leaves source dtype on auto.
            gptq_dtype = torch.float16 if source_dtype == "auto" else torch_dtype
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
                torch_dtype=gptq_dtype,
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

def dequantize_gptq_torch(qweight_t: torch.Tensor, qzeros_t: torch.Tensor,
                           scales_t: torch.Tensor, g_idx_t: torch.Tensor,
                           device: str, bits: int = 4) -> torch.Tensor:
    """
    All unpacking and dequantization is done with torch. The heavy arithmetic
    runs on the selected torch device, with bit unpacking kept on CPU for
    backend compatibility.

    Why module.forward(eye) is NOT called directly on the accelerator:
    auto_gptq's QuantLinear.forward() calls `self.g_idx.long()` and
    bitwise_right_shift on the backend, which may not support these ops,
    corrupting int64 data even after an earlier cast to int32 (because
    forward() casts back to .long() = int64 inside, uncontrollable from
    outside when calling module.forward()).

    Fix: module.forward() is NOT called. The 2 steps are done with plain torch:
    1) Bit unpacking (bitwise_right_shift, g_idx indexing) - on the CPU because
       these bitwise/indexing ops are cheap compared to the matrix arithmetic.
    2) The heavy computation - (weight - zero) * scale - runs on the selected
       torch device.
    """
    assert bits == 4, "Only 4-bit GPTQ is supported."
    qweight_t = qweight_t.cpu().to(torch.int64)   # bitwise + shift need the CPU
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
    # Matches the real auto_gptq formula exactly: +1 then masked again mod 16 (wraparound)
    zeros_int4 = (zeros_int4 + 1) & 0xF

    zeros_per_row = zeros_int4[g_idx_t, :]   # indexing on the CPU - safe
    scales_per_row = scales_t[g_idx_t, :]

    # Run the dense arithmetic on the selected torch device.
    w_f = w_int4.to(torch.float32)
    z_f = zeros_per_row.to(torch.float32)
    if device != "cpu":
        try:
            w_f_device = w_f.to(device)
            z_f_device = z_f.to(device)
            scales_device = scales_per_row.to(device)
            w_f, z_f, scales_per_row = w_f_device, z_f_device, scales_device
        except Exception as e:
            print(f"  ⚠️ Cannot move to {device} ({e}), computing on the CPU")
            w_f = w_f.cpu()
            z_f = z_f.cpu()
            scales_per_row = scales_per_row.cpu()

    weight = (w_f - z_f) * scales_per_row
    weight = weight.t().contiguous().to(torch.float32).cpu()  # [out_features, in_features]
    return weight


def dequantize_gptq_linear(qweight: np.ndarray, qzeros: np.ndarray,
                            scales: np.ndarray, g_idx: np.ndarray,
                            bits: int = 4) -> np.ndarray:
    """
    FIX (v2): matches EXACTLY the real auto_gptq source
    QuantLinear.forward (qlinear_cuda_old.py, the bits=4 branch):

        zeros = ((qzeros >> shift) & 0xF)
        zeros = zeros + 1
        zeros = zeros & 0xF        # <-- IMPORTANT: masked AGAIN after +1

    The bug in the previous version: +1 WITHOUT masking again. When the original zero_point = 15
    (0b1111), +1 must wrap to 0 (mod 16) but the old version kept 16 -> every
    group in this case got a shifted offset -> wrong weights -> the model generated
    chaotic tokens although everything else (loading, RoPE, attention...) was right.
    """
    assert bits == 4, "Only 4-bit GPTQ is supported."

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

    # FIX: +1 then masked again mod 16 (as the real auto_gptq does), NO overflow to 16
    zeros_int4 = (zeros_int4 + 1) & 0xF

    scales_f32 = scales.astype(np.float32)
    group_for_row = g_idx.astype(np.int64)
    zeros_per_row = zeros_int4[group_for_row, :]
    scales_per_row = scales_f32[group_for_row, :]

    weight = (w_int4.astype(np.float32) - zeros_per_row.astype(np.float32)) * scales_per_row
    # weight currently has shape [in_features, out_features]; transposed exactly once
    # here to return the nn.Linear convention: [out_features, in_features].
    return weight.T.astype(np.float32)


def dequantize_gptq_module(module, device: str) -> np.ndarray:
    """
    Uses the REAL QuantLinear module (auto_gptq) to get the dequantized weight,
    instead of decoding the bit unpacking itself - guaranteed to match 100% the computation
    auto_gptq uses at real inference, avoiding any order/convention risk.
    The module is loaded on CPU. This module alone is moved to the selected
    device right before computing, then moved back to CPU afterwards so the
    accelerator does not retain all quantized modules at once.
    """
    in_features = getattr(module, "infeatures", None) or module.in_features
    out_features = getattr(module, "outfeatures", None) or module.out_features

    moved = False
    if device != "cpu":
        try:
            # These packed values fit in int32. Convert before transfer for
            # backends that have limited int64 support.
            for attr_name in ("g_idx", "qweight", "qzeros"):
                if hasattr(module, attr_name):
                    buf = getattr(module, attr_name)
                    if buf is not None and buf.dtype == torch.int64:
                        buf.data = buf.data.to(torch.int32)
            module.to(device)
            moved = True
        except Exception as e:
            print(f"  ⚠️ Cannot move the module to {device} ({e}), computing on the CPU")
            module.to("cpu")
            device = "cpu"

    eye = torch.eye(in_features, dtype=torch.float16, device=device)

    had_bias = getattr(module, "bias", None) is not None
    saved_bias = None
    if had_bias:
        saved_bias = module.bias.data.clone()
        module.bias.data.zero_()

    with torch.no_grad():
        out = module(eye)  # [in_features, out_features] = W^T (because y = x @ W^T, x=I)

    if had_bias:
        module.bias.data = saved_bias

    weight = out.t().contiguous().to(torch.float32).cpu().numpy()  # -> [out_features, in_features]
    assert weight.shape == (out_features, in_features), f"shape mismatch: {weight.shape} != {(out_features, in_features)}"

    # Detect failed accelerator execution before writing the checkpoint.
    bad = (not np.isfinite(weight).all()) or (np.abs(weight).max() < 1e-12)
    if bad and moved:
        print(f"  ⚠️ Suspected {device} execution error (NaN/all zeros) - recomputing this layer on CPU")
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
        device_type = torch.device(device).type
        backend = getattr(torch, device_type, None)
        synchronize = getattr(backend, "synchronize", None)
        try:
            if device_type == "mps" and hasattr(torch, "mps"):
                torch.mps.synchronize()
            elif callable(synchronize):
                synchronize(device)
        except Exception:
            pass
        del eye, out
        _clear_device_cache(device)
        module.to("cpu")

    return weight


def find_gptq_modules(model):
    """Returns a list of (dotted_name, module) for every GPTQ QuantLinear in the model."""
    result = []
    for name, module in model.named_modules():
        if hasattr(module, "qweight") and hasattr(module, "qzeros") and hasattr(module, "scales"):
            result.append((name, module))
    return result


def normalize_key_prefix(name: str) -> str:
    if name.startswith("model.model."):
        return name[len("model."):]
    # FIX: lm_head sits OUTSIDE the .model submodule in the original LlamaForCausalLM,
    # so through the AutoGPTQForCausalLM wrapper (self.model = LlamaForCausalLM)
    # it gets only ONE extra "model." prefix (becoming "model.lm_head.weight"),
    # not 2 like the other tensors -> it does not match the condition above,
    # is not stripped -> nim_inference.nim does not find "lm_head.weight"
    # -> silently falls back to the WRONG tied embedding (this model does not
    # tie embeddings) -> all final logits wrong -> garbage tokens.
    if name.startswith("model.lm_head."):
        return name[len("model."):]
    return name


# ============================================================
# 3. PyTorch Quantization Functions (CPU/CUDA/ROCm/MPS/XPU/...)
# ============================================================

def _to_quant_tensor(arr: np.ndarray, device: str) -> torch.Tensor:
    source = np.ascontiguousarray(arr, dtype=np.float32)
    return torch.from_numpy(source).to(device=device, dtype=torch.float32)


def _round_away_from_zero(values: torch.Tensor) -> torch.Tensor:
    """Match Nim's std/math.round behavior rather than torch's ties-to-even."""
    return torch.where(values >= 0, torch.floor(values + 0.5), torch.ceil(values - 0.5))


def _numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().to(device="cpu").contiguous().numpy()


def _pack_flat_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """Pack flat 4-bit values, with the first value in the low nibble."""
    flat = codes.reshape(-1)
    even = flat[0::2].to(torch.uint8) & 0x0F
    odd = flat[1::2].to(torch.uint8) & 0x0F
    packed = even
    if odd.numel():
        packed[:odd.numel()] |= odd << 4
    return packed


def _pack_row_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """Pack 2D row-oriented 4-bit values, padding each odd-width row separately."""
    even = codes[:, 0::2].to(torch.uint8) & 0x0F
    odd = codes[:, 1::2].to(torch.uint8) & 0x0F
    packed = even
    if odd.shape[1]:
        packed[:, :odd.shape[1]] |= odd << 4
    return packed


def _quantize_int8(arr: np.ndarray, device: str):
    values = _to_quant_tensor(arr, device).reshape(-1)
    maximum = values.abs().amax()
    scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 127.0)
    codes = _round_away_from_zero(values / scale).clamp(-127, 127).to(torch.int8)
    return _numpy(codes), float(scale.detach().cpu().item()), np.asarray([0.0], dtype=np.float32), 0, 0, 0


def _quantize_int4_symmetric(arr: np.ndarray, device: str):
    values = _to_quant_tensor(arr, device).reshape(-1)
    maximum = values.abs().amax()
    scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 7.0)
    codes = _round_away_from_zero(values / scale).clamp(-7, 7).to(torch.int8)
    return (_numpy(_pack_flat_nibbles(codes)), float(scale.detach().cpu().item()),
            np.asarray([0.0], dtype=np.float32), 0, 0, 0)


def _quantize_int4_asymmetric(arr: np.ndarray, device: str):
    values = _to_quant_tensor(arr, device).reshape(-1)
    minimum = values.amin()
    maximum = values.amax()
    value_range = maximum - minimum
    scale = torch.where(value_range < 1e-12, torch.ones_like(value_range), value_range / 15.0)
    zero_point = -minimum / scale
    codes = _round_away_from_zero(values / scale + zero_point).clamp(0, 15).to(torch.uint8)
    return (_numpy(_pack_flat_nibbles(codes)), float(scale.detach().cpu().item()),
            float(zero_point.detach().cpu().item()))


def _quantize_q4km_grouped(arr: np.ndarray, group_size: int, device: str):
    if arr.ndim != 2:
        raise ValueError("grouped Q4_K_M quantization requires a 2D tensor")
    if group_size <= 0 or group_size % 2:
        raise ValueError("Q4_K_M group size must be a positive even number")
    rows, cols = arr.shape
    if cols < 2:
        packed, scale, zero_point = _quantize_int4_asymmetric(arr, device)[:3]
        return packed, np.asarray([scale], dtype=np.float32), np.asarray([zero_point], dtype=np.float32), 0, 0, 0

    values = _to_quant_tensor(arr, device)
    groups_per_row = (cols + group_size - 1) // group_size
    scales = torch.empty((rows, groups_per_row), dtype=torch.float32, device=device)
    zeros = torch.empty_like(scales)
    codes = torch.empty((rows, cols), dtype=torch.uint8, device=device)
    for group in range(groups_per_row):
        first = group * group_size
        last = min(first + group_size, cols)
        block = values[:, first:last]
        minimum = block.amin(dim=1)
        maximum = block.amax(dim=1)
        value_range = maximum - minimum
        scale = torch.where(value_range < 1e-12, torch.ones_like(value_range), value_range / 15.0)
        zero_point = -minimum / scale
        codes[:, first:last] = _round_away_from_zero(
            block / scale[:, None] + zero_point[:, None]
        ).clamp(0, 15).to(torch.uint8)
        scales[:, group] = scale
        zeros[:, group] = zero_point

    return (_numpy(_pack_row_nibbles(codes)).reshape(-1), _numpy(scales).reshape(-1),
            _numpy(zeros).reshape(-1), 0, 0, group_size)


def _fp4_magnitude_indices(values: torch.Tensor) -> torch.Tensor:
    indices = torch.zeros_like(values, dtype=torch.int32)
    for threshold in (0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0):
        indices += (values >= threshold).to(torch.int32)
    return indices


def _e4m3_encode_positive(values: torch.Tensor) -> torch.Tensor:
    """Encode E4M3 scales with customfloat.nim's truncating mantissa rule."""
    safe = torch.where(values > 0, values, torch.ones_like(values))
    real_exp = torch.floor(torch.log2(safe))
    new_exp = real_exp.to(torch.int32) + 7
    mantissa = torch.floor((safe / torch.exp2(real_exp) - 1.0) * 8.0).clamp(0, 7).to(torch.int32)
    encoded = ((new_exp.clamp(0, 15) << 3) | mantissa).to(torch.int32)
    overflow = (new_exp >= 15) | ~torch.isfinite(values)
    underflow = (new_exp <= 0) | (values <= 0)
    encoded = torch.where(overflow, torch.full_like(encoded, 0x78), encoded)
    encoded = torch.where(underflow, torch.zeros_like(encoded), encoded)
    return encoded.to(torch.uint8)


def _e4m3_decode(values: torch.Tensor) -> torch.Tensor:
    codes = values.to(torch.int32)
    exponent = (codes >> 3) & 0x0F
    mantissa = codes & 0x07
    decoded = torch.exp2((exponent - 7).to(torch.float32)) * (1.0 + mantissa.to(torch.float32) / 8.0)
    decoded = torch.where(exponent == 0x0F, torch.full_like(decoded, float("inf")), decoded)
    return torch.where((exponent == 0) & (mantissa == 0), torch.zeros_like(decoded), decoded)


def _quantize_fp4km(arr: np.ndarray, device: str):
    if arr.ndim != 2 or arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise ValueError("FP4-KM requires a non-empty 2D tensor")
    if arr.shape[1] % FP4KM_GROUP_SIZE:
        raise ValueError(f"FP4-KM requires columns divisible by {FP4KM_GROUP_SIZE}")

    values = _to_quant_tensor(arr, device).reshape(-1)
    group_count = values.numel() // FP4KM_GROUP_SIZE
    groups = values.reshape(group_count, FP4KM_GROUP_SIZE)
    chunk_groups = 16384

    max_scale = torch.zeros((), dtype=torch.float32, device=device)
    for first in range(0, group_count, chunk_groups):
        block = groups[first:min(first + chunk_groups, group_count)]
        candidate = block.abs().amax(dim=1).amax() / 6.0
        max_scale = torch.maximum(max_scale, candidate)
    global_scale = torch.where(max_scale > 0, max_scale / 224.0, torch.ones_like(max_scale))

    nibble_bytes = values.numel() // 2
    payload = np.empty(nibble_bytes + group_count, dtype=np.uint8)
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32, device=device,
    )
    for first in range(0, group_count, chunk_groups):
        last = min(first + chunk_groups, group_count)
        block = groups[first:last]
        maxima = block.abs().amax(dim=1)
        bases = maxima / 6.0
        best_error = torch.full((last - first,), float("inf"), dtype=torch.float32, device=device)
        best_byte = torch.zeros((last - first,), dtype=torch.uint8, device=device)
        best_scale = torch.zeros((last - first,), dtype=torch.float32, device=device)

        for step in range(8):
            factor = float(np.float32(
                np.float32(0.62) +
                np.float32(np.float32(0.62) * np.float32(step)) / np.float32(7.0)
            ))
            encoded = _e4m3_encode_positive((bases * factor) / global_scale)
            scales = _e4m3_decode(encoded) * global_scale
            valid = (scales > 0) & torch.isfinite(scales)
            inverse = torch.where(valid, 1.0 / scales, torch.zeros_like(scales))
            scaled = block * inverse[:, None]
            indices = _fp4_magnitude_indices(scaled.abs())
            quantized = magnitudes[indices] * scales[:, None]
            quantized = torch.where(scaled < 0, -quantized, quantized)
            errors = (quantized - block).square().sum(dim=1)
            improve = valid & (errors < best_error)
            best_error = torch.where(improve, errors, best_error)
            best_byte = torch.where(improve, encoded, best_byte)
            best_scale = torch.where(improve, scales, best_scale)

        inverse = torch.where(best_scale > 0, 1.0 / best_scale, torch.zeros_like(best_scale))
        scaled = block * inverse[:, None]
        indices = _fp4_magnitude_indices(scaled.abs())
        codes = indices | torch.where(scaled < 0, 8, 0).to(torch.int32)
        packed = codes[:, :16] | (codes[:, 16:] << 4)
        payload[first * 16:last * 16] = _numpy(packed.to(torch.uint8)).reshape(-1)
        payload[nibble_bytes + first:nibble_bytes + last] = _numpy(best_byte)

    return (payload, np.asarray([global_scale.detach().cpu().item()], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32), 4, 3, FP4KM_GROUP_SIZE)


def _pack_tensor_on_device(arr: np.ndarray, quant_kind: int, group_size: int, device: str):
    if quant_kind == QK_INT8:
        return _quantize_int8(arr, device)
    if quant_kind == QK_INT4:
        return _quantize_int4_symmetric(arr, device)
    if quant_kind == QK_INT4_ASYMMETRIC:
        if arr.ndim == 2 and arr.shape[1] >= 2:
            actual_group_size = min(group_size, arr.shape[1])
            actual_group_size -= actual_group_size % 2
            if actual_group_size >= 2:
                return _quantize_q4km_grouped(arr, actual_group_size, device)
        packed, scale, zero_point = _quantize_int4_asymmetric(arr, device)
        return packed, np.asarray([scale], dtype=np.float32), np.asarray([zero_point], dtype=np.float32), 0, 0, 0
    if quant_kind == QK_FP4KM:
        return _quantize_fp4km(arr, device)
    raise ValueError(f"Unsupported quant_kind: {quant_kind}")


def _clear_device_cache(device: str):
    device_type = torch.device(device).type
    try:
        if device_type == "cuda":
            torch.cuda.empty_cache()
        elif device_type == "mps" and hasattr(torch, "mps"):
            torch.mps.empty_cache()
        elif device_type == "xpu" and hasattr(torch, "xpu"):
            torch.xpu.empty_cache()
    except Exception:
        pass


def pack_tensor_for_nim(arr: np.ndarray, quant_kind: int,
                        group_size: int = DEFAULT_Q4KM_GROUP_SIZE):
    """Quantize with PyTorch on the selected device; return CPU payload for NIMQ I/O."""
    global QUANT_DEVICE
    arr = np.asarray(arr, dtype=np.float32)
    if quant_kind == QK_FP32_RAW:
        return arr.reshape(-1).astype(np.float32, copy=False), np.asarray([1.0], dtype=np.float32), np.asarray([0.0], dtype=np.float32), 0, 0, 0
    try:
        return _pack_tensor_on_device(arr, quant_kind, group_size, QUANT_DEVICE)
    except (RuntimeError, NotImplementedError) as exc:
        if QUANT_DEVICE == "cpu":
            raise
        print(f"  ⚠️ {QUANT_DEVICE} quantization failed ({exc}); retrying this and remaining tensors on CPU")
        _clear_device_cache(QUANT_DEVICE)
        QUANT_DEVICE = "cpu"
        return _pack_tensor_on_device(arr, quant_kind, group_size, "cpu")


# ============================================================
# 4. Write Functions
# ============================================================

def write_string(f, s: str):
    encoded = s.encode('utf-8')
    f.write(struct.pack('<i', len(encoded)))
    f.write(encoded)


def write_quant_tensor(f, name: str, arr: np.ndarray, kind: int = QK_FP32_RAW,
                       group_size: int = DEFAULT_Q4KM_GROUP_SIZE):
    """Writes one QuantTensor - supports per-row/per-group scale/zero_point (arrays, not a single float)"""
    data, scales, zeros, eb, mb, packed_group_size = pack_tensor_for_nim(arr, kind, group_size)

    actual_kind = {
        QK_FP32_RAW: 0,
        QK_INT8: 1,
        QK_INT4: 2,
        QK_INT4_ASYMMETRIC: 3,
        QK_FP4KM: 8,
    }.get(kind, 0)
    scales = np.asarray(scales, dtype=np.float32).flatten()
    zeros = np.asarray(zeros, dtype=np.float32).flatten()

    f.write(struct.pack('<B', actual_kind))
    f.write(struct.pack('<i', len(arr.shape)))
    for d in arr.shape:
        f.write(struct.pack('<i', d))
    f.write(struct.pack('<i', len(scales)))   # nScales (1=per-tensor, nRows=per-row, nRows*nGroups=per-group)
    f.write(scales.tobytes())
    f.write(zeros.tobytes())
    f.write(struct.pack('<i', packed_group_size))  # 0=per-tensor; >0=per-group
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
    source_dtype: str = "auto",
    group_size: int = DEFAULT_Q4KM_GROUP_SIZE,
    quant_device: str = "auto",
):
    global QUANT_DEVICE
    start_time = time.time()
    QUANT_DEVICE = resolve_quant_device(quant_device)

    print(f"Loading model (source dtype: {source_dtype})...")
    backend_label = "ROCm" if QUANT_DEVICE.startswith("cuda") and getattr(torch.version, "hip", None) else QUANT_DEVICE
    print(f"Quantization backend: {backend_label} (PyTorch device {QUANT_DEVICE})")
    model, tokenizer, config = load_model(model_name, is_gptq, source_dtype)

    arch = [
        config["vocab_size"],
        config["hidden_size"],
        config["num_attention_heads"],
        config["num_hidden_layers"],
        config["intermediate_size"] // config["hidden_size"],
    ]

    print(f"Arch: {arch}")
    quant_names = {
        QK_FP32_RAW: "FP32 raw",
        QK_INT8: "INT8",
        QK_INT4: "INT4 symmetric",
        QK_INT4_ASYMMETRIC: f"Q4_K_M grouped INT4 (group={group_size})",
        QK_FP4KM: f"FP4-KM (group={FP4KM_GROUP_SIZE})",
    }
    print(f"Quantization: {quant_names.get(quant_kind, 'UNKNOWN')}")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    state_dict = model.state_dict()
    gptq_bases = find_gptq_bases(state_dict)
    is_gptq_model = len(gptq_bases) > 0

    if is_gptq_model:
        print(f"🔍 Found {len(gptq_bases)} GPTQ layers (torch + {QUANT_DEVICE}, NO numpy for the computation)")

    normal_tensors = []
    for name, tensor in state_dict.items():
        if "num_batches_tracked" in name:
            continue
        if "model.norm" in name and "weight" not in name:
            continue
        if any(name.endswith(suf) for suf in (".qweight", ".qzeros", ".scales", ".g_idx")):
            continue
        # FIX: the bias of GPTQ bases (e.g. model.layers.0.self_attn.q_proj.bias)
        # IS a normal tensor in the state_dict (no qweight/
        # qzeros/scales/g_idx suffix), so it used to pass the exclusion conditions
        # above and get written HERE with kind=quant_kind (the bias quantized to
        # int4 PER --q4km, although biases should always be fp32) - THEN written
        # AGAIN correctly (QK_FP32_RAW) in the GPTQ loop below. The tensor
        # was written twice with the same name - wasting quantization time +
        # space, and whether it loads CORRECTLY depended entirely on the
        # order of these 2 loops (byName[name]=qt on the Nim side overwrites in
        # file order - correct BY LUCK because the GPTQ loop runs later, not by
        # design). Skipped entirely here, so only the GPTQ loop writes it once,
        # right from the start.
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
        write_string(f, "NIMQ2")  # bumped from NIMQ1 -> NIMQ2 (adds the groupSize field, see quant.nim)
        for v in arch:
            f.write(struct.pack('<i', v))
        f.write(struct.pack('<i', total_tensors))

        # Normal tensors
        tensor_count = 0
        for name in normal_tensors:
            tensor = state_dict[name]
            tensor_count += 1
            if tensor_count % 20 == 0:
                print(f"  Writing {tensor_count}/{len(normal_tensors)}: {name}")

            norm_name = normalize_key_prefix(name)
            # NIMQ raw storage is float32. Converting through float32 preserves
            # FP16/BF16 checkpoint values exactly and avoids the old forced-FP16
            # truncation of FP32 checkpoints. NumPy does not support BF16 arrays.
            arr = tensor.detach().to(torch.float32).cpu().numpy()

            kind = quant_kind
            if "embed" in name.lower() or "lm_head" in name.lower() or "norm" in name.lower():
                kind = QK_FP32_RAW
            elif kind == QK_FP4KM and (arr.ndim != 2 or arr.shape[1] % FP4KM_GROUP_SIZE):
                print(f"  ⚠️ FP4-KM requires a matrix with columns divisible by {FP4KM_GROUP_SIZE}; "
                      f"keeping {norm_name} as raw FP32")
                kind = QK_FP32_RAW

            write_string(f, norm_name)
            write_quant_tensor(f, norm_name, arr, kind, group_size)
            del arr

        # GPTQ: bit unpacking/indexing stays on CPU for backend compatibility;
        # the dense (weight - zero) * scale step runs on the selected torch device.
        # The formula matches the real auto_gptq
        # (zero+1 then masked again mod 16).
        if is_gptq_model:
            print(f"  Processing {len(gptq_bases)} GPTQ layers (torch + {QUANT_DEVICE})...")
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

                weight_t = dequantize_gptq_torch(qweight_t, qzeros_t, scales_t, g_idx_t, QUANT_DEVICE, bits=4)
                weight_f32 = weight_t.numpy()  # converted to numpy only for writing the file (write_quant_tensor needs numpy to pack the binary)

                norm_name = normalize_key_prefix(base + ".weight")
                write_string(f, norm_name)
                kind = quant_kind
                if kind == QK_FP4KM and (weight_f32.ndim != 2 or
                                         weight_f32.shape[1] % FP4KM_GROUP_SIZE):
                    print(f"  ⚠️ FP4-KM requires a matrix with columns divisible by {FP4KM_GROUP_SIZE}; "
                          f"keeping {norm_name} as raw FP32")
                    kind = QK_FP32_RAW
                write_quant_tensor(f, norm_name, weight_f32, kind, group_size)

                bias_key = base + ".bias"
                if bias_key in state_dict:
                    bias = state_dict[bias_key].detach().to(torch.float32).cpu().numpy()
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

    # === FIX (earlier): tokenizer.json used to be exported as a FAKE byte-identity table
    # {"vocab_size": 256, "bytes": [0..255]} - not the model's real
    # tokenizer at all (the real model uses subword BPE with tens of thousands of tokens and
    # real merges). Now the REAL vocab + merges come from the loaded
    # AutoTokenizer, with the same schema as BPETokenizer in
    # test_nimformer.nim: {"vocab_size", "vocab": {token_str: id}, "merges":
    # ["a b", ...]} - so both sides can read/write each other, no more fake byte map.
    tokenizer_path = os.path.join(os.path.dirname(output_path) or ".", "tokenizer.json")
    exported = False
    # Exporting next to a Hugging Face checkout must not replace its
    # tokenizer.json: nim_inference / formers.n read that file (it has a
    # "model" block), the simplified schema below is only for test_nimformer.
    if os.path.exists(tokenizer_path):
        try:
            with open(tokenizer_path) as g:
                existing = json.load(g)
            if isinstance(existing, dict) and "model" in existing:
                print(f"   Tokenizer: keeping the Hugging Face {tokenizer_path}")
                exported = True
        except Exception:
            pass
    if not exported and getattr(tokenizer, "is_fast", False):
        try:
            backend_json = json.loads(tokenizer.backend_tokenizer.to_str())
            model_block = backend_json.get("model", {})
            vocab = model_block.get("vocab")
            merges_raw = model_block.get("merges")
            if vocab is not None and merges_raw is not None:
                # merges in the tokenizers lib may be list[str] "a b" or
                # list[list[str]] [a, b] depending on the version - normalised to "a b".
                merges = []
                for m in merges_raw:
                    if isinstance(m, (list, tuple)):
                        merges.append(f"{m[0]} {m[1]}")
                    else:
                        merges.append(m)
                with open(tokenizer_path, 'w') as g:
                    json.dump({
                        "vocab_size": len(vocab),
                        "vocab": vocab,
                        "merges": merges,
                    }, g)
                print(f"   Tokenizer (real BPE, {len(vocab)} tokens, {len(merges)} merges) saved to {tokenizer_path}")
                exported = True
        except Exception as e:
            print(f"   ⚠️  Could not get vocab/merges from backend_tokenizer ({e}), falling back")

    if not exported:
        # Fallback: a tokenizer that is not a standard "fast" BPE (e.g. plain SentencePiece
        # unigram) - still exports the real vocab (id -> token string) through
        # get_vocab(), NO more fake byte table. merges stays empty
        # (on the Nim side empty merges mean tokenizing by "whole-word vocab
        # lookup" instead of BPE merges, still the minimal correct semantics).
        try:
            vocab = tokenizer.get_vocab()
            with open(tokenizer_path, 'w') as g:
                json.dump({
                    "vocab_size": len(vocab),
                    "vocab": vocab,
                    "merges": [],
                }, g)
            print(f"   ⚠️  The tokenizer is not a standard fast BPE: exporting the real vocab ({len(vocab)} tokens, "
                  f"no merges) to {tokenizer_path}")
        except Exception as e:
            raise RuntimeError(
                f"Cannot export the real tokenizer for this model ({e}). "
                f"Refusing to fall back to a fake byte table because it would break tokenization at inference."
            )

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
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--source-dtype", "--input-dtype", choices=("auto", "fp16", "bf16", "fp32"),
                        default="auto", help="Checkpoint/load dtype (default: auto, preserve checkpoint dtype)")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--int8", action="store_true", help="Quantize to symmetric INT8")
    target.add_argument("--int4", action="store_true", help="Quantize to symmetric INT4")
    target.add_argument("--q4km", action="store_true",
                        help="Quantize to grouped asymmetric Q4_K_M-style INT4")
    target.add_argument("--fp4km", "--fp4-km", action="store_true",
                        help="Quantize eligible matrices to Nimformer FP4-KM (group size 32)")
    target.add_argument("--raw", "--fp32", "--fp16", dest="raw", action="store_true",
                        help="Write raw FP32 NIMQ tensors (legacy --fp16 alias; NIMQ raw is FP32)")
    parser.add_argument("--group-size", type=int, default=128,
                         help="Even group size for --q4km (default 128). --fp4km uses fixed groups of 32.")
    parser.add_argument("--quant-device", "--device", default="auto",
                        help="PyTorch device for quantization: auto, cpu, cuda, rocm, mps, xpu, or a torch device string")
    parser.add_argument("--no-accelerator", "--no-mps", action="store_true",
                        help="Force CPU quantization (legacy --no-mps alias)")

    args = parser.parse_args()

    if args.no_accelerator and args.quant_device not in ("auto", "cpu"):
        parser.error("--no-accelerator cannot be combined with an explicit accelerator device")
    quant_device = "cpu" if args.no_accelerator else args.quant_device

    if args.group_size <= 0 or args.group_size % 2:
        parser.error("--group-size must be a positive even integer")

    if args.fp4km:
        quant_kind = QK_FP4KM
        print(f"🔹 Using FP4-KM quantization (group size {FP4KM_GROUP_SIZE})")
    elif args.q4km:
        quant_kind = QK_INT4_ASYMMETRIC
        print(f"🔹 Using grouped Q4_K_M-style INT4 (group size {args.group_size})")
    elif args.int4:
        quant_kind = QK_INT4
        print("🔹 Using INT4 SYMMETRIC quantization (faster but lower quality)")
    elif args.int8:
        quant_kind = QK_INT8
        print("🔹 Using INT8 quantization (balanced)")
    elif args.raw:
        quant_kind = QK_FP32_RAW
        print("🔹 Using raw FP32 NIMQ tensors (unquantized)")
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

    export_to_nimq(
        args.model,
        args.output,
        is_gptq=is_gptq,
        quant_kind=quant_kind,
        source_dtype=args.source_dtype,
        group_size=args.group_size,
        quant_device=quant_device,
    )

if __name__ == "__main__":
    main()
