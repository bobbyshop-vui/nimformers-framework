# Codestral Mamba v0.1 — class source bundle

This ZIP contains class definitions and the exact published architecture configuration for **mistralai/Mamba-Codestral-7B-v0.1**. No model weights are included.

## What is inside

- `src/codestral_mamba.py` — standalone PyTorch class definitions:
  - `CodestralMambaConfig`
  - `CodestralMambaBlock`
  - `CodestralMambaModel`
  - `CodestralMambaForCausalLM`
- `upstream/config.json` — published Hugging Face architecture configuration.
- `upstream/params.json` — published original Mistral parameter configuration.
- `upstream_sources.txt` — exact upstream source links used as references.

## Exact architecture configuration

The published HF config identifies the model as `Mamba2ForCausalLM` with 64 layers, hidden size 4096, state size 128, 128 heads, head dimension 64, 8 groups, convolution kernel 4, expand 2, vocabulary size 32768, and bf16 weights.

## Dependencies

The class wrapper intentionally delegates the high-performance Mamba-2 SSM kernel to the public `mamba_ssm` implementation rather than bundling CUDA/Triton kernels.

```bash
python3.10 -m pip install torch mamba-ssm causal-conv1d einops
```

The official model card recommends `mamba_ssm` and `causal-conv1d`; Mistral's inference repository also publishes Codestral-Mamba support.

## Important

`codestral_mamba.py` is a clean standalone implementation of the class structure/configuration for portability. It is **not a verbatim copy of Hugging Face's current 947-line `modeling_mamba2.py`**, because that file is part of Transformers and evolves independently. The ZIP therefore also records the upstream URLs so the exact upstream implementation can be retrieved.
