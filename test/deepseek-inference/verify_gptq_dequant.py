#!/usr/bin/env python3
"""
verify_gptq_dequant.py - Kiểm chứng công thức dequantize_gptq_torch_mps
bằng cách so sánh với auto_gptq THẬT (ground truth) trên 1 tensor nhỏ.

Chạy: python3.10 verify_gptq_dequant.py --model TheBloke/deepseek-coder-6.7B-instruct-GPTQ

In ra sai số max/mean giữa 2 cách tính. Nếu sai số ~0 (< 1e-3) -> công thức
đúng, bug nằm ở chỗ khác (RoPE/attention/embedding/...). Nếu sai số lớn ->
xác nhận đúng là công thức dequant sai, cần sửa tiếp ở đó.
"""
import argparse
import torch
import numpy as np


def dequantize_gptq_torch_mps(qweight_t, qzeros_t, scales_t, g_idx_t, device="cpu", bits=4):
    assert bits == 4
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

    weight = (w_int4.to(torch.float32) - zeros_per_row.to(torch.float32)) * scales_per_row
    return weight.t().contiguous()  # [out_features, in_features]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", default="model.layers.0.self_attn.q_proj",
                     help="tên layer GPTQ để test (không có prefix model. lặp lại)")
    args = ap.parse_args()

    from auto_gptq import AutoGPTQForCausalLM
    print(f"Loading {args.model} (CPU, có thể mất vài phút)...")
    model = AutoGPTQForCausalLM.from_quantized(
        args.model, device="cpu", use_triton=False, use_safetensors=True,
        trust_remote_code=True, disable_exllama=True, use_exllama=False,
        use_exllamav2=False, use_qigen=False, torch_dtype=torch.float16,
    )

    # Tìm đúng module GPTQ theo tên layer yêu cầu
    target = None
    target_name = None
    for name, module in model.named_modules():
        if hasattr(module, "qweight") and args.layer in name:
            target = module
            target_name = name
            break
    if target is None:
        print(f"Không tìm thấy layer chứa '{args.layer}'. Các layer GPTQ có sẵn:")
        for name, module in model.named_modules():
            if hasattr(module, "qweight"):
                print(" -", name)
        return

    print(f"Test layer: {target_name}")
    in_features = target.infeatures
    out_features = target.outfeatures
    print(f"  in_features={in_features} out_features={out_features}")

    # ===== Ground truth: auto_gptq THẬT, qua module.forward(eye) =====
    eye = torch.eye(in_features, dtype=torch.float16, device="cpu")
    had_bias = getattr(target, "bias", None) is not None
    saved_bias = None
    if had_bias:
        saved_bias = target.bias.data.clone()
        target.bias.data.zero_()
    with torch.no_grad():
        out_true = target(eye)  # [in_features, out_features]
    if had_bias:
        target.bias.data = saved_bias
    weight_true = out_true.t().contiguous().to(torch.float32)  # [out_features, in_features]

    # ===== Công thức tự viết (torch, không numpy) =====
    weight_mine = dequantize_gptq_torch_mps(
        target.qweight.detach(), target.qzeros.detach(),
        target.scales.detach(), target.g_idx.detach(), device="cpu",
    )

    diff = (weight_true - weight_mine).abs()
    print(f"\nShape true={tuple(weight_true.shape)} mine={tuple(weight_mine.shape)}")
    print(f"Max abs diff : {diff.max().item():.6f}")
    print(f"Mean abs diff: {diff.mean().item():.6f}")
    print(f"weight_true sample [0,:8]: {weight_true[0,:8].tolist()}")
    print(f"weight_mine sample [0,:8]: {weight_mine[0,:8].tolist()}")

    if diff.max().item() < 1e-2:
        print("\n✅ KHỚP - công thức dequant ĐÚNG. Bug nằm ở chỗ khác (RoPE/attention/embedding/lm_head/...).")
    else:
        print("\n❌ SAI - công thức dequant còn lỗi, cần sửa tiếp ở dequantize_gptq_torch_mps.")


if __name__ == "__main__":
    main()