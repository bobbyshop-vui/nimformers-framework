"""Llama-family inference over NIMQ checkpoints."""
from __future__ import annotations

import ctypes as ct
from dataclasses import dataclass

import numpy as np

from ._lib import F32, I64, NimformersError
from .tokenizer import Tokenizer

__all__ = ["LlamaConfig", "LlamaModel"]


@dataclass(frozen=True)
class LlamaConfig:
    """Fields of a Hugging Face Llama-style ``config.json``."""

    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    max_position_embeddings: int
    rms_norm_eps: float
    rope_theta: float

    @classmethod
    def load(cls, lib, config_path) -> "LlamaConfig":
        ints = np.zeros(7, dtype=np.int64)
        floats = np.zeros(2, dtype=np.float32)
        status = lib.nimf_llama_config_load(
            str(config_path).encode(), ints.ctypes.data_as(I64),
            floats.ctypes.data_as(F32))
        if status != 0:
            msg = lib.nimf_last_error()
            raise NimformersError(
                msg.decode("utf-8", "replace") if msg else "config load failed")
        return cls(*(int(v) for v in ints), float(floats[0]), float(floats[1]))

    @classmethod
    def from_handle(cls, lib, handle) -> "LlamaConfig":
        ints = np.zeros(7, dtype=np.int64)
        floats = np.zeros(2, dtype=np.float32)
        if lib.nimf_llama_config(handle, ints.ctypes.data_as(I64),
                                 floats.ctypes.data_as(F32)) != 0:
            msg = lib.nimf_last_error()
            raise NimformersError(
                msg.decode("utf-8", "replace") if msg else "config read failed")
        return cls(*(int(v) for v in ints), float(floats[0]), float(floats[1]))


class LlamaModel:
    """A loaded Llama/RoPE/RMSNorm/SwiGLU/GQA model.

    Loading a multi-GB checkpoint is slow and memory hungry; this is the same
    loader used by the ``nim_inference`` command-line program.  Use
    ``upload_resident=True`` to keep quantized weights on the GPU.
    """

    def __init__(self, ctx, handle):
        if not handle:
            raise NimformersError("failed to load Llama model")
        self.ctx = ctx
        self.lib = ctx.lib
        self.handle = handle

    @classmethod
    def load(cls, ctx, config_path, weights_path,
             upload_resident: bool = True) -> "LlamaModel":
        handle = ctx.lib.nimf_llama_load(
            ctx._ctx, str(config_path).encode(), str(weights_path).encode(),
            1 if upload_resident else 0)
        if not handle:
            ctx._raise()
        return cls(ctx, handle)

    @property
    def config(self) -> LlamaConfig:
        return LlamaConfig.from_handle(self.lib, self.handle)

    def generate(self, tokenizer: Tokenizer, prompt: str,
                 system: str = "", max_new_tokens: int = 128,
                 temperature: float = 0.7, top_p: float = 0.9,
                 use_chat_template: bool = True) -> str:
        """Generate a completion and return the decoded text."""
        ptr = self.lib.nimf_llama_generate(
            self.ctx._ctx, self.handle, tokenizer.handle,
            str(prompt).encode(), str(system).encode(), int(max_new_tokens),
            float(temperature), float(top_p), 1 if use_chat_template else 0)
        if not ptr:
            self.ctx._raise()
        try:
            return ct.string_at(ptr).decode("utf-8", "replace")
        finally:
            self.lib.nimf_free_string(ptr)

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.lib.nimf_llama_destroy(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        self.close()

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    def __repr__(self) -> str:
        try:
            cfg = self.config
            return (f"LlamaModel(vocab={cfg.vocab_size}, hidden={cfg.hidden_size}, "
                    f"layers={cfg.num_layers})")
        except Exception:
            return "LlamaModel(<closed>)"
