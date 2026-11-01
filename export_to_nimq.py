#!/usr/bin/env python3
"""
export_to_nimq.py - asymmetric int4 (Q4_K_M) support with MPS acceleration

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
# MPS check - FIX: force FP16
# ============================================================
USE_MPS = False
if torch.backends.mps.is_available():
    USE_MPS = True
    # === FIX: force FP16 to avoid the BFloat16 error ===
    torch.set_default_dtype(torch.float16)
    print("🔹 MPS (Metal) available - using the GPU for quantization (FP16)")
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
else:
    print("🔹 MPS not available - using the CPU")

# ============================================================
# QuantKind
# ============================================================
QK_FP32_RAW = 0
QK_INT8 = 1
QK_INT4 = 2
QK_INT4_ASYMMETRIC = 3


# ============================================================
# 1. Load model - FIX: force the dtype for MPS
# ============================================================

def load_model(model_name: str, is_gptq: bool = False) -> Tuple[Any, Any, Dict]:
    print(f"Loading model: {model_name}")

    # FIX: ALWAYS load on the CPU even with USE_MPS=True. accelerate.load_checkpoint_in_model
    # tries set_module_tensor_to_device straight onto MPS with the original dtype of the
    # checkpoint (usually bfloat16) BEFORE it can be cast to float16 ->
    # "TypeError: Trying to convert BFloat16 to the MPS backend" because MPS
    # has no bfloat16. Loading on the CPU first (stable, tested many times),
    # then dequantize_gptq_module() moves each module .to("mps") SEPARATELY
    # (by then the tensors are float16, which MPS accepts).
    device = "cpu"
    print(f"   Using device: {device} (GPTQ modules move to MPS themselves while dequantizing if USE_MPS)")

    # === FIX: force FP16 for MPS ===
    torch_dtype = torch.float16

    if is_gptq:
        try:
            from auto_gptq import AutoGPTQForCausalLM
            # === FIX: torch_dtype added to from_quantized ===
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
    numpy is FORBIDDEN - everything in torch as requested. Real MPS is used for the
    heavy compute.

    Why module.forward(eye) is NOT called directly on MPS (tried, always
    broken): auto_gptq's QuantLinear.forward() calls `self.g_idx.long()` and
    bitwise_right_shift RIGHT ON the MPS device - the PyTorch MPS backend handles these 2 ops
    wrongly (especially on integrated Intel GPUs, not Apple Silicon),
    corrupting int64 data even after an earlier cast to int32 (because
    forward() casts back to .long() = int64 inside, uncontrollable from
    outside when calling module.forward()).

    Fix: module.forward() is NOT called. The 2 steps are done with plain torch:
    1) Bit unpacking (bitwise_right_shift, g_idx indexing) - on the CPU because
       these are the 2 ops MPS gets wrong, and they are CHEAP (not the
       time-consuming part).
    2) The really heavy computation - (weight - zero) * scale over a matrix of
       millions of elements - moved to MPS and run for real with torch (where
       MPS really helps, unlike bitwise/indexing).
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

    # === The heavy compute: moved to MPS and run for real with torch ===
    w_f = w_int4.to(torch.float32)
    z_f = zeros_per_row.to(torch.float32)
    if device != "cpu":
        try:
            w_f = w_f.to(device)
            z_f = z_f.to(device)
            scales_per_row = scales_per_row.to(device)
        except Exception as e:
            print(f"  ⚠️ Cannot move to {device} ({e}), computing on the CPU")

    weight = (w_f - z_f) * scales_per_row   # real torch on MPS (if device=mps)
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
    The module is currently on the CPU (float16, because load_model() always loads on the CPU to avoid
    the accelerate+MPS+BFloat16 error while loading the checkpoint) - this
    module alone is moved to the device (mps) right before computing (it is float16 by then, so MPS
    accepts it), then moved back to the CPU afterwards to release MPS memory gradually instead
    of keeping all 224 modules on the GPU at once.
    """
    in_features = getattr(module, "infeatures", None) or module.in_features
    out_features = getattr(module, "outfeatures", None) or module.out_features

    moved = False
    if device != "cpu":
        try:
            # FIX: PyTorch MPS has a known bug with int64 tensors - .to("mps")
            # may corrupt the data (the cause of g_idx turning into
            # garbage values such as the observed 1087250403). Cast to int32
            # BEFORE moving to the device - int32 is well supported by MPS, and
            # the real values of g_idx/qweight/qzeros all fit within
            # int32 (g_idx: 0..n_groups~86, qweight/qzeros: 4-bit packed
            # in uint32 but the real values are always < 2^31), so no
            # information is lost by the cast.
            for attr_name in ("g_idx", "qweight", "qzeros"):
                if hasattr(module, attr_name):
                    buf = getattr(module, attr_name)
                    if buf is not None and buf.dtype == torch.int64:
                        buf.data = buf.data.to(torch.int32)
            module.to(device)
            moved = True
        except Exception as e:
            print(f"  ⚠️ Cannot move the module to {device} ({e}), computing on the CPU")
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

    # FIX: if the driver silently "ignores" an MPS command buffer (like the
    # "command buffer exited with error status" log at the last layer), the result read
    # back is usually NaN or all 0 (the buffer was never written). Detected
    # -> this layer is recomputed on the CPU (slower but surely right) instead of
    # silently writing garbage weights to the file.
    bad = (not np.isfinite(weight).all()) or (np.abs(weight).max() < 1e-12)
    if bad and moved:
        print(f"  ⚠️ Suspected MPS command buffer error (NaN/all zeros) - recomputing this layer on the CPU")
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
        # FIX: sync + free the GPU cache RIGHT after each module, BEFORE
        # moving the module back to the CPU. Without this -> the command buffers of
        # the next module pile up on the previous one in the MPS queue, the
        # Intel Iris driver (weak, not Apple Silicon) is overloaded after ~200
        # consecutive submissions -> "command buffer exited with error status" / GPU
        # errors silently ignored by the driver at the last layers -> garbage weights.
        torch.mps.synchronize()
        del eye, out
        torch.mps.empty_cache()
        module.to("cpu")  # releases MPS memory, not keeping all 224 modules at once

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
    Asymmetric int4 quantization, per GROUP instead of per ROW: each group_size columns
    of a row use their own scale/zero_point, instead of one scale for the WHOLE row
    (the old per-row). With group_size=128 (the usual granularity of the original GPTQ),
    the double-quantization error (GPTQ 4-bit/group -> float32 -> int4 again) drops
    sharply compared with per-row (1 scale/4096 columns) - the main cause of the drifting
    hidden state at after_block_0 with the earlier --q4km.
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
    Per-row (per output channel) asymmetric int4 - each row (arr.shape[0])
    has its OWN scale/zero_point instead of one value for the whole tensor.
    Weights after GPTQ dequant have very different ranges between rows (each GPTQ row
    already had its own scale/group) -> per-tensor quantization (one shared scale)
    stretches the range, and the error accumulates badly with double quantization. Per-row keeps the
    precision close to GPTQ's original per-group, still int4 (~4GB).
    Returns: packed data [nRows, ceil(nCols/2)] uint8, scales[nRows], zeros[nRows]
    """
    if arr.ndim != 2:
        # fallback: a 1D tensor (rarely takes the int4 branch, but safe)
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

    # Pack 2 4-bit values into 1 byte - VECTORIZED
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

    # Pack 2 4-bit values into 1 byte - VECTORIZED
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
    Uses MPS (Metal) to quantize when a GPU is present - 10-50x faster
    """
    if not USE_MPS:
        # Fall back to the vectorized CPU version
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
        # === FIX: converted to FP16 before going to MPS ===
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


INT4_ASYMMETRIC_GROUP_SIZE = 128  ## ADDED: default group=128 (close to the original GPTQ granularity).
                                   ## Change it for a lighter compression (larger groups) or
                                   ## more precision (smaller groups), trading off size.

def pack_tensor_for_nim(arr: np.ndarray, quant_kind: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int, int]:
    """
    Packs a tensor - with MPS if available. scales/zeros are always returned as arrays
    (len=1 for the old per-tensor kinds, len=nRows for 2D int4-asymmetric
    per-row, len=nRows*nGroupsPerRow for per-group). Also returns group_size
    (0 = per-tensor/per-row, >0 = per-group with that group_size).
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
            # CHANGED: per-group (default 128) instead of per-row - per-row (1 scale/
            # 4096 columns) is too coarse compared with the original GPTQ (per-group ~128 columns/scale) ->
            # double quantization loses too much precision -> the hidden state
            # drifts visibly after block 0 even though the GPTQ dequant and the forward formulas are right.
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
    """Writes one QuantTensor - supports per-row/per-group scale/zero_point (arrays, not a single float)"""
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
        print(f"🔍 Found {len(gptq_bases)} GPTQ layers (torch + {device}, NO numpy for the computation)")

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
            if USE_MPS:
                tensor = tensor.cpu()
            arr = tensor.detach().to(torch.float16).numpy()

            kind = quant_kind
            if "embed" in name.lower() or "lm_head" in name.lower() or "norm" in name.lower():
                kind = QK_FP32_RAW

            write_string(f, norm_name)
            write_quant_tensor(f, norm_name, arr, kind)
            del arr

        # GPTQ: plain torch (NO numpy) - bit unpacking (bitwise/int64
        # indexing, broken on MPS) on the CPU, the heavy compute (subtract+multiply over the whole
        # matrix) REALLY on MPS with torch. The formula matches the real auto_gptq
        # (zero+1 then masked again mod 16).
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
                weight_f32 = weight_t.numpy()  # converted to numpy only for writing the file (write_quant_tensor needs numpy to pack the binary)

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
    parser.add_argument("--fp16", action="store_true", help="Store as FP16 (no quantization)")
    parser.add_argument("--int8", action="store_true", help="Quantize to int8 (recommended)")
    parser.add_argument("--int4", action="store_true", help="Quantize to int4 (symmetric)")
    parser.add_argument("--q4km", action="store_true", help="Quantize to int4 asymmetric (Q4_K_M style, better quality)")
    parser.add_argument("--group-size", type=int, default=128,
                         help="Group size for --q4km (default 128, like the original GPTQ granularity). "
                              "Smaller -> more precise but a larger file; larger -> the opposite.")
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