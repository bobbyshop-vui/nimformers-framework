---
license: apache-2.0
library_name: mamba-ssm
tags:
- mamba
- mamba-3
- state-space-model
- causal-lm
- language-modeling
---

# Mamba-3 SISO 187M

`mamba3-siso-187m` is a pretrained causal language model with 187M
parameters. It is built from stacked blocks, each containing a Mamba-3
SISO mixer followed by a gated MLP. It contains no attention layers and
is released with BF16 weights in the public `mamba_ssm` checkpoint format.

## Model architecture

| Property | Value |
|---|---:|
| Parameters | 187M |
| Layers | 12 |
| Model dimension | 768 |
| SSM state size | 128 |
| SSM head dimension | 64 |
| SSM heads | 24 |
| SSM groups | 1 |
| MIMO | No |
| Chunk size | 64 |
| Context length | 2,048 |

The model was pretrained on 100B tokens from
[FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu).
It uses tied input and output embeddings and the `meta-llama/Llama-3.1-8B` tokenizer.

## Installation

Install CUDA-enabled PyTorch first, followed by the latest Mamba source:

```bash
pip install git+https://github.com/state-spaces/mamba.git --no-build-isolation
```

While this repository is private, authenticate with Hugging Face:

```bash
hf auth login
```

## Usage

```python
import torch
from transformers import AutoTokenizer
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel

model_id = "state-spaces/mamba3-siso-187m"

tokenizer = AutoTokenizer.from_pretrained(
    "meta-llama/Llama-3.1-8B",
)

model = MambaLMHeadModel.from_pretrained(
    model_id,
    device="cuda",
    dtype=torch.bfloat16,
)
model.eval()

input_ids = tokenizer(
    "Mamba-3 is",
    return_tensors="pt",
).input_ids.cuda()

with torch.inference_mode():
    logits = model(input_ids).logits

print(logits.shape)
```

## References

- [Mamba-3: Improved Sequence Modeling using State Space Principles](https://arxiv.org/abs/2603.15569)
- [Official Mamba implementation](https://github.com/state-spaces/mamba)

## Citation

If you use this model, please cite:

```bibtex
@misc{lahoti2026mamba3improvedsequencemodeling,
  title         = {Mamba-3: Improved Sequence Modeling using State Space Principles},
  author        = {Aakash Lahoti and Kevin Y. Li and Berlin Chen and
                   Caitlin Wang and Aviv Bick and J. Zico Kolter and
                   Tri Dao and Albert Gu},
  year          = {2026},
  eprint        = {2603.15569},
  archivePrefix = {arXiv},
  primaryClass  = {cs.LG},
  url           = {https://arxiv.org/abs/2603.15569}
}
```
