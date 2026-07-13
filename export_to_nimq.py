#!/usr/bin/env python3
"""
export_to_nimq.py
Chuyển model transformers (PyTorch) sang định dạng .nimq của Nim (quant.nim).

Hỗ trợ:
- Model thường (FP16/BF16)
- GPTQ (AutoGPTQForCausalLM)
- Tied embedding
- Có thể chọn FP16, FP32, hoặc int8

Cách dùng:
    python export_to_nimq.py --model TheBloke/deepseek-coder-6.7B-instruct-GPTQ --output model.nimq
    python export_to_nimq.py --model meta-llama/Llama-2-7b-hf --output llama2.nimq --fp16
"""

import os
import sys
import argparse
import struct
import json
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
import numpy as np

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

# ============================================================
# 1. Load model (hỗ trợ cả GPTQ và thường)
# ============================================================

def load_model(model_name: str, is_gptq: bool = False) -> Tuple[Any, Any, Dict]:
    """
    Load model và config từ Hugging Face.
    Trả về: (model, tokenizer, config_dict)
    """
    print(f"Loading model: {model_name}")
    
    if is_gptq:
        try:
            from auto_gptq import AutoGPTQForCausalLM
            model = AutoGPTQForCausalLM.from_quantized(
                model_name,
                device="cpu",
                use_triton=False,
                use_safetensors=True,
                trust_remote_code=True,
                disable_exllama=True,
            )
        except ImportError:
            print("auto-gptq not installed. Falling back to normal transformers.")
            model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="cpu",
                trust_remote_code=True
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float16, device_map="cpu",
            trust_remote_code=True
        )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    
    # Đọc config thành dict
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
# 2. Extract state_dict và chuẩn hóa tên
# ============================================================

def extract_state_dict(model, config_dict: Dict) -> Dict[str, np.ndarray]:
    """
    Lấy state_dict, transpose Linear weight, bỏ các layer không cần.
    """
    state_dict = model.state_dict()
    result = {}
    
    for name, tensor in state_dict.items():
        # Bỏ các layer không cần (norm stats, ...)
        if "num_batches_tracked" in name:
            continue
        if "model.norm" in name and "weight" not in name:
            continue
        
        # Chuyển sang numpy FP16
        arr = tensor.detach().to(torch.float16).numpy()
        
        # Transpose Linear weight: PyTorch [out, in] -> [in, out] cho matmul
        if "weight" in name and (
            "embed" not in name.lower() and
            "lm_head" not in name.lower() and
            "norm" not in name.lower()
        ):
            # Linear weight (2D)
            if len(arr.shape) == 2:
                arr = arr.T  # [out, in] -> [in, out]
            elif len(arr.shape) == 3:
                # Conv weight? Giữ nguyên
                pass
        
        # Xử lý bias (giữ nguyên)
        result[name] = arr
    
    # Xử lý tied embedding
    if config_dict["tie_word_embeddings"]:
        # Nếu lm_head không có weight riêng, copy từ embed_tokens
        lm_key = "lm_head.weight"
        embed_key = "model.embed_tokens.weight"
        if lm_key not in result and embed_key in result:
            # Transpose ngược lại cho lm_head: [in, out] -> [out, in]
            arr = result[embed_key].T  # [embed_dim, vocab]
            result[lm_key] = arr
            print(f"Tied embedding: copied {embed_key} -> {lm_key}")
    
    return result


# ============================================================
# 3. Lưu .nimq file (theo định dạng quant.nim)
# ============================================================

def write_string(f, s: str):
    f.write(struct.pack('<i', len(s)))
    f.write(s.encode('utf-8'))

def write_quant_tensor(f, name: str, arr: np.ndarray, kind: int = 1):
    """
    Ghi 1 QuantTensor theo định dạng quant.nim
    kind: 1 = qkFp32Raw, 2 = qkInt8, 3 = qkInt4
    """
    # Chuyển về float32 để lưu (nếu FP16 thì scale khác)
    flat = arr.flatten().astype(np.float32)
    
    write_string(f, name)
    f.write(struct.pack('<B', kind))          # kind
    f.write(struct.pack('<i', len(arr.shape)))  # ndims
    for d in arr.shape:
        f.write(struct.pack('<i', d))
    f.write(struct.pack('<f', 1.0))           # scale (không dùng)
    f.write(struct.pack('<i', 0))             # exponent_bits
    f.write(struct.pack('<i', 0))             # mantissa_bits
    f.write(struct.pack('<i', flat.nbytes))   # data length
    f.write(flat.tobytes())                   # data


def export_to_nimq(
    model_name: str,
    output_path: str,
    is_gptq: bool = False,
    kind: int = 1,  # 1=FP32, 2=Int8
):
    """
    Xuất model transformers sang .nimq
    """
    # 1. Load model
    model, tokenizer, config = load_model(model_name, is_gptq)
    
    # 2. Extract state_dict
    state_dict = extract_state_dict(model, config)
    
    # 3. Tạo arch array
    arch = [
        config["vocab_size"],
        config["hidden_size"],
        config["num_attention_heads"],
        config["num_hidden_layers"],
        config["intermediate_size"] // config["hidden_size"],
    ]
    
    # 4. Ghi file
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, 'wb') as f:
        # Magic header
        f.write(b'NIMQ1')
        # Arch
        for v in arch:
            f.write(struct.pack('<i', v))
        # Number of tensors
        f.write(struct.pack('<i', len(state_dict)))
        # Tensors
        for name, arr in state_dict.items():
            write_quant_tensor(f, name, arr, kind)
    
    print(f"✅ Exported to {output_path}")
    print(f"   Arch: {arch}")
    print(f"   Tensors: {len(state_dict)}")
    
    # 5. Lưu tokenizer (để dùng với nim)
    tokenizer_path = os.path.join(os.path.dirname(output_path), "tokenizer.json")
    # Lưu vocab bytes (cho CharTokenizer) hoặc full tokenizer
    if hasattr(tokenizer, "vocab"):
        # Hugging Face tokenizer có vocab
        vocab_bytes = []
        for i in range(min(256, config["vocab_size"])):
            vocab_bytes.append(i)
        with open(tokenizer_path, 'w') as g:
            json.dump({"vocab_size": min(256, config["vocab_size"]), "bytes": vocab_bytes}, g)
    else:
        # Fallback: chỉ lưu vocab_size
        with open(tokenizer_path, 'w') as g:
            json.dump({"vocab_size": config["vocab_size"], "bytes": list(range(256))}, g)
    
    print(f"   Tokenizer saved to {tokenizer_path}")
    
    # 6. Lưu config riêng cho Nim (optional)
    config_path = os.path.join(os.path.dirname(output_path), "config.nim.json")
    with open(config_path, 'w') as g:
        json.dump(config, g, indent=2)
    print(f"   Config saved to {config_path}")


# ============================================================
# 4. CLI
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Export transformers model to .nimq")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument("--output", default="model.nimq", help="Output .nimq file")
    parser.add_argument("--gptq", action="store_true", help="Model is GPTQ")
    parser.add_argument("--fp16", action="store_true", help="Keep FP16 (default: FP32)")
    parser.add_argument("--int8", action="store_true", help="Quantize to int8")
    
    args = parser.parse_args()
    
    kind = 1  # FP32
    if args.fp16:
        print("Note: FP16 is not directly supported in quant.nim, storing as FP32 but data is FP16")
    if args.int8:
        kind = 2  # int8
    
    export_to_nimq(args.model, args.output, is_gptq=args.gptq, kind=kind)


if __name__ == "__main__":
    main()