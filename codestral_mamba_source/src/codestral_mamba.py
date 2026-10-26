"""Standalone class definition for the Mamba-Codestral-7B-v0.1 architecture.

This bundle intentionally does NOT include model weights.
It uses the public mamba_ssm Mamba2 implementation for the SSM kernel.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import nn

try:
    from mamba_ssm import Mamba2
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "Install mamba-ssm first: python3.10 -m pip install mamba-ssm causal-conv1d"
    ) from exc


@dataclass
class CodestralMambaConfig:
    model_type: str = "mamba2"
    vocab_size: int = 32768
    hidden_size: int = 4096
    num_hidden_layers: int = 64
    state_size: int = 128
    conv_kernel: int = 4
    expand: int = 2
    num_heads: int = 128
    head_dim: int = 64
    n_groups: int = 8
    chunk_size: int = 256
    hidden_act: str = "silu"
    rms_norm: bool = True
    norm_before_gate: bool = True
    residual_in_fp32: bool = True
    use_bias: bool = False
    use_conv_bias: bool = True
    layer_norm_epsilon: float = 1e-5
    time_step_rank: int = 256
    time_step_min: float = 0.001
    time_step_max: float = 0.1
    time_step_floor: float = 1e-4

    @classmethod
    def from_dict(cls, values: dict) -> "CodestralMambaConfig":
        aliases = {
            "d_model": "hidden_size",
            "d_state": "state_size",
            "d_conv": "conv_kernel",
            "headdim": "head_dim",
            "ngroups": "n_groups",
        }
        data = dict(values)
        for old, new in aliases.items():
            if old in data and new not in data:
                data[new] = data.pop(old)
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)


class CodestralMambaBlock(nn.Module):
    """One pre-norm Mamba-2 residual block."""

    def __init__(self, config: CodestralMambaConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.layer_norm_epsilon)
        self.mixer = Mamba2(
            d_model=config.hidden_size,
            d_state=config.state_size,
            d_conv=config.conv_kernel,
            expand=config.expand,
            headdim=config.head_dim,
            ngroups=config.n_groups,
            rmsnorm=config.rms_norm,
            norm_before_gate=config.norm_before_gate,
            chunk_size=config.chunk_size,
            bias=config.use_bias,
            conv_bias=config.use_conv_bias,
            layer_idx=layer_idx,
        )

    def forward(self, hidden_states: torch.Tensor, **kwargs) -> torch.Tensor:
        residual = hidden_states
        x = self.norm(hidden_states)
        if self.config.residual_in_fp32:
            residual = residual.float()
        x = self.mixer(x, **kwargs)
        return residual + x


class CodestralMambaModel(nn.Module):
    """Codestral-Mamba backbone: embeddings -> 64 Mamba-2 blocks -> final norm."""

    def __init__(self, config: Optional[CodestralMambaConfig] = None):
        super().__init__()
        self.config = config or CodestralMambaConfig()
        self.embeddings = nn.Embedding(self.config.vocab_size, self.config.hidden_size)
        self.layers = nn.ModuleList(
            [CodestralMambaBlock(self.config, i) for i in range(self.config.num_hidden_layers)]
        )
        self.norm_f = nn.RMSNorm(
            self.config.hidden_size, eps=self.config.layer_norm_epsilon
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embeddings

    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        hidden_states = self.embeddings(input_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states)
        return self.norm_f(hidden_states)


class CodestralMambaForCausalLM(nn.Module):
    """Codestral-Mamba backbone plus an untied LM head."""

    def __init__(self, config: Optional[CodestralMambaConfig] = None):
        super().__init__()
        self.config = config or CodestralMambaConfig()
        self.backbone = CodestralMambaModel(self.config)
        self.lm_head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.backbone.get_input_embeddings()

    def forward(
        self,
        input_ids: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        hidden_states = self.backbone(input_ids)
        logits = self.lm_head(hidden_states).float()
        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )
        return logits, loss


CODESRAL_MAMBA_CONFIG = CodestralMambaConfig()

__all__ = [
    "CodestralMambaConfig",
    "CodestralMambaBlock",
    "CodestralMambaModel",
    "CodestralMambaForCausalLM",
    "CODESRAL_MAMBA_CONFIG",
]
