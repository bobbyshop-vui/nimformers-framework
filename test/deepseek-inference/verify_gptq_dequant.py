#!/usr/bin/env python3
"""
verify_gptq_dequant.py - Checks the dequantize_gptq_torch_mps formula
by comparing it with the REAL auto_gptq (ground truth) on one small tensor.

Run: python3.10 verify_gptq_dequant.py --model TheBloke/deepseek-coder-6.7B-instruct-GPTQ

Prints the max/mean error between the two computations. If the error is ~0 (< 1e-3) the formula
is right and the bug is elsewhere (RoPE/attention/embedding/...). If the error is large,
the dequant formula is confirmed wrong and needs more work there.
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
                     help="GPTQ layer name to test (without the repeated model. prefix)")
    args = ap.parse_args()

    from auto_gptq import AutoGPTQForCausalLM
    print(f"Loading {args.model} (CPU, may take a few minutes)...")
    model = AutoGPTQForCausalLM.from_quantized(
        args.model, device="cpu", use_triton=False, use_safetensors=True,
        trust_remote_code=True, disable_exllama=True, use_exllama=False,
        use_exllamav2=False, use_qigen=False, torch_dtype=torch.float16,
    )

    # Find the GPTQ module for the requested layer name
    target = None
    target_name = None
    for name, module in model.named_modules():
        if hasattr(module, "qweight") and args.layer in name:
            target = module
            target_name = name
            break
    if target is None:
        print(f"No layer containing '{args.layer}'. Available GPTQ layers:")
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

    # ===== Hand-written formula (torch, no numpy) =====
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
        print("\n✅ MATCH - the dequant formula is RIGHT. The bug is elsewhere (RoPE/attention/embedding/lm_head/...).")
    else:
        print("\n❌ WRONG - the dequant formula is still broken, fix dequantize_gptq_torch_mps.")


if __name__ == "__main__":
    main()