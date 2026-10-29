
# Codestral Mamba v0.1 — bộ source class

[English](README.md) | [Tiếng Việt](README.vi.md)

Bộ ZIP chứa định nghĩa class và cấu hình kiến trúc được công bố của
**mistralai/Mamba-Codestral-7B-v0.1**; không kèm trọng số model.


## Nội dung

- `src/codestral_mamba.py`: các class PyTorch độc lập:
  - `CodestralMambaConfig`
  - `CodestralMambaBlock`
  - `CodestralMambaModel`
  - `CodestralMambaForCausalLM`
- `upstream/config.json`: cấu hình kiến trúc Hugging Face được công bố.
- `upstream/params.json`: cấu hình tham số Mistral gốc được công bố.
- `upstream_sources.txt`: liên kết source upstream được dùng tham khảo.


## Cấu hình kiến trúc

Config HF công bố nhận diện `Mamba2ForCausalLM`: 64 layer, hidden size 4096,
state size 128, 128 head, head dimension 64, 8 group, convolution kernel 4,
expand 2, vocabulary 32768 và trọng số bf16.


## Phụ thuộc

Wrapper class giao kernel Mamba-2 SSM hiệu năng cao cho implementation
`mamba_ssm` công khai, thay vì đóng gói kernel CUDA/Triton.

```bash
python3.10 -m pip install torch mamba-ssm causal-conv1d einops
```

Model card chính thức khuyến nghị `mamba_ssm` và `causal-conv1d`; repo
inference của Mistral cũng có hỗ trợ Codestral-Mamba.


## Lưu ý

`codestral_mamba.py` triển khai độc lập cấu trúc class/config để dễ di
chuyển. Nó **không sao chép nguyên văn file `modeling_mamba2.py` 947 dòng
được mô tả trong bản nguồn Hugging Face**, vì file thuộc Transformers và
phát triển độc lập. ZIP lưu URL upstream để có thể lấy implementation gốc.
