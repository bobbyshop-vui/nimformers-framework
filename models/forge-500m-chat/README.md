---
license: apache-2.0
language:
- en
pipeline_tag: text-generation
tags:
- hybrid-ssm
- mha
- moe
- state-space-models
- recurrent-gated-expert
- arg
- hse
- b300
- runpod
- instruct
- chat
base_model: Phase-Technologies/forge-500m-base
---

# FORGE-500M-Chat: First-Principles Optimal Recurrent-Gated Expert Architecture

<div align="center">

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Parameters](https://img.shields.io/badge/Parameters-500M%20Total%20%7C%20320M%20Active-green.svg)]()
[![Hardware](https://img.shields.io/badge/Trained%20On-4%C3%97%20NVIDIA%20B300%20SXM6-76B900.svg?logo=nvidia)]()
[![Architecture](https://img.shields.io/badge/Architecture-Hybrid%20SSM%20%2B%20MHA%20%2B%20HSE--MoE-orange.svg)]()
[![Training Tokens](https://img.shields.io/badge/Tokens-1.66B%20Pretrain%20%2B%20Aligned-purple.svg)]()

</div>

---

## Executive Summary

**FORGE-500M-Chat** is a state-of-the-art hybrid sub-billion parameter conversational language model designed from first principles by **Soham Pal**. It solves the compute accessibility gap in AI research by replacing quadratic self-attention with linear-time recurrent state-space recurrence, paired with hierarchical sparse expert mixture-of-experts (HSE-MoE).

The model was pretrained on **1.66 Billion tokens** from scratch across 4× NVIDIA B300 SXM6 MIG compute slices, aligned via **Supervised Fine-Tuning (SFT)** on multi-turn dialogue with per-token loss masking, and polished using **Direct Preference Optimization (DPO)**.

The full console training trajectory from Step 1 to Step 821 is archived in this repository under [`training_journey.log`](./training_journey.log).

---

## Model Architecture Specifications

FORGE employs an asymmetric, multi-layer hybrid topology designed for maximum parameter utility and linear sequence scalability:

| Specification | Value | Technical Description |
| :--- | :--- | :--- |
| **Total Parameters** | **500,019,616** | 500.0M total weights (FP16/BF16 storage: ~1.0 GB) |
| **Active Parameters** | **~320,000,000** | Active per-token forward pass via Sparse HSE-MoE |
| **Embedding Size** | 211,444,736 | 206,464 vocabulary × 1,024 hidden dimension |
| **Hidden Dimension ($d_{\text{model}}$)** | 1,024 | Intermediate representation width |
| **Total Transformer Layers** | 18 | Interleaved hybrid recurrence + global attention |
| **SSM Recurrence Layers** | 15 (83.3%) | **Adaptive Recurrent Gating (ARG)** linear state space |
| **Attention Layers** | 3 (16.7%) | **Global Multi-Head Attention (MHA)** with RoPE + YaRN |
| **Attention Heads** | 8 | 8 Query heads, 8 Key/Value heads |
| **MoE FFN Layers** | 9 | **Hierarchical Sparse Expert (HSE)** (32 experts, top-2 active) |
| **Dense FFN Layers** | 9 | Dense SwiGLU FFN with $8/3 \times d_{\text{model}}$ expansion |
| **Context Window** | 4,096 | Native context (scalable to 16,384 with YaRN) |
| **Tokenizer** | CRAYON Turbo | 206,464 vocabulary (BPE with subword byte-fallback) |

```
Input Tokens (B, T)
        │
        ▼
   [CRAYON Embedding (206k Vocab, 1024 d_model)]
        │
   ┌────┴───────────────────────────────────────┐
   │ Layer 0..17: Interleaved Blocks           │
   │  ├─ 15× ARG Layers: Linear SSM Recurrence │
   │  ├─  3× Global MHA: Long-Range Attention  │
   │  ├─  9× SwiGLU Dense FFN                  │
   │  └─  9× Hierarchical Sparse Expert (MoE)  │
   └────┬───────────────────────────────────────┘
        ▼
   [Final RMSNorm + Chunked LM Projection Head]
        │
        ▼
   Target Logits / Probabilities
```

---

## The Full Training Journey & Historical Narrative

The pretraining and post-training of FORGE-500M represents an empirical milestone in lean, high-efficiency AI development:

### 1. The Pretraining Stage
- **Cluster**: 4× NVIDIA B300 SXM6 MIG slices (`1g.34gb`, 18 SMs each, CUDA 13.2, Driver 595.91).
- **Global Batch**: 1,048,576 tokens per step (32 micro-batches × 32,768 tokens per batch).
- **Physical Throughput**: 7,337 unique tokens per second (~26.4 Million tokens/hour).
- **Step 1**: Initial Cross-Entropy Loss: **12.512**, Initial Perplexity: **> 150,000**.
- **Step 821**: Final Loss: **4.6473**, Final Perplexity: **104.30**.
- **Cumulative Pretrain Tokens**: **1,660,944,384 tokens** (1.66 Billion tokens).
- **Challenges Overcome**:
  - Survived multiple cloud host network disruptions and PCIe MIG interface resets via resilient checkpoint auto-recovery.
  - Successfully resolved PyTorch 2.5 `nn.Module` dynamic attribute restrictions with custom parameter wrapping (`convert_to_attr_dict`).
  - Implemented chunked linear projection (`chunk_size=1024`) on the 206,464-token vocabulary to eliminate VRAM allocation spikes.

### 2. Supervised Fine-Tuning (SFT)
- **Dataset**: Curated mixture of ShareGPT dialogues, UltraChat multi-turn instructions, CodeFeedback Python programming pairs, MetaMath mathematical reasoning, and OpenOrca synthetic data.
- **Protocol**: Chat template formatting with assistant turn loss masking (no loss on system or user prompts).
- **Optimization**: PyTorch DistributedDataParallel (DDP) across all 4 GPUs with cosine learning rate decay ($2\times 10^{-5} \to 2\times 10^{-6}$) and BF16 autocast.
- **Result**: Loss dropped rapidly to **~4.0**, establishing clean turn-taking and instruction responsiveness.

### 3. Direct Preference Optimization (DPO)
- **Dataset**: Multi-domain preference pairs from UltraFeedback, Anthropic HH-RLHF, and HelpSteer2.
- **Protocol**: Trainable policy model aligned against frozen BF16 SFT reference model with KL divergence regularization ($\beta = 0.1$).
- **Result**: Achieved positive reward margin on chosen vs. rejected completions, eliminating repetitive degradation and improving refusal of hazardous requests.

---

## Complete Training Console Logs Preserved

In accordance with our commitment to total scientific reproducibility, the complete console log file (`training_journey.log`) containing every training step from Step 1 to Step 821, memory footprints, step durations, and loss curves is provided directly in the repository root:

- **Filename**: [`training_journey.log`](./training_journey.log)
- **Size**: ~5.9 MB
- **Coverage**: Steps 1 through 821, SFT, and DPO alignment logs.

---

## How to Run Inference

### 1. Google Colab (1-Click Quickstart)

In a Google Colab notebook (T4 / A100 GPU), paste and run this single cell:

```bash
# 1. Clone repository & enter directory
!git clone https://huggingface.co/Phase-Technologies/FORGE-500M-Chat
%cd FORGE-500M-Chat

# 2. Install dependencies
!pip install -q torch einops triton transformers huggingface_hub

# 3. Generate text immediately
!python run_inference.py --prompt "<|bos|><|usr|>Explain how quantum computers differ from classical computers.<|asst|>" --max_new_tokens 128
```

---

### 2. Pure Python API (Colab / Jupyter / Local)

```python
import sys
import torch
from pathlib import Path
from huggingface_hub import snapshot_download

# 1. Download model repository
repo_dir = Path(snapshot_download(repo_id="Phase-Technologies/FORGE-500M-Chat"))
sys.path.insert(0, str(repo_dir))

from tokenizer.crayon_wrapper import ForgeTokenizer
from config import ForgeModelConfig
from model.forge_model import build_forge_500m

tokenizer = ForgeTokenizer.from_pretrained(str(repo_dir / "tokenizer"))
config = ForgeModelConfig.from_json(str(repo_dir / "model_config.json"))
model = build_forge_500m(config)

# 3. Load BF16 weights
weights = torch.load(repo_dir / "pytorch_model.bin", map_location="cpu")
model.load_state_dict({k: v.to(torch.bfloat16) for k, v in weights.items()})
model = model.to("cuda" if torch.cuda.is_available() else "cpu").to(torch.bfloat16)
model.eval()

# 4. Formatted Chat Prompt
prompt = "<|bos|><|usr|> Explain how quantum computers differ from classical computers in simple terms.<|asst|>"

input_ids = torch.tensor([tokenizer.encode(prompt, add_bos=False, add_eos=False)], device=model.device)

# 5. Autoregressive Generation
with torch.inference_mode():
    output_ids = model.generate(
        input_ids=input_ids,
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.9,
        eos_token_id=tokenizer.special_tokens["<|eos|>"],
    )

print(tokenizer.decode(output_ids[0].tolist(), skip_special_tokens=True))
```

### 3. Special Conversation Tokens

| Token | String | Function |
| :--- | :--- | :--- |
| **BOS** | `<|bos|>` | Beginning of sequence |
| **EOS** | `<|eos|>` | End of sequence / turn termination |
| **SYS** | `<|sys|>` | System instruction block prefix |
| **USR** | `<|usr|>` | User prompt prefix |
| **ASST** | `<|asst|>` | Assistant completion prefix |
| **PAD** | `<|pad|>` | Sequence padding token |

---

## a note from the creator.....

honestly i really wanted to train this beauty for full 15b tokens..... that was the real plan from day one to see how far this custom architecture could push..... but running multiple top tier gpus burns money so fast and i just ran completely out of budget..... literally had to stop pretraining early at 821 steps..... then quickly did sft and dpo right on the pod before shutting it down..... huge shoutout and love to runpod.io for providing the insane compute and infrastructure that made training this beast possible..... every single cent went into this little 500m model and seeing it come alive from scratch with custom cuda kernels and our own tokenizer was pure magic..... maybe one day with better funding we will train the full 3b or 15b token run..... until then i hope you guys enjoy playing with this model as much as i loved building it..... thank you so much to everyone who supported this crazy journey.....

