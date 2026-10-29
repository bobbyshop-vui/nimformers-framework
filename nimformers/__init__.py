"""Nimformers: Nim transformer/GPU kernels exposed to Python via a C ABI.

Quick start::

    import numpy as np
    from nimformers import Context

    with Context() as ctx:
        a = np.random.randn(4, 8).astype(np.float32)
        b = np.random.randn(8, 16).astype(np.float32)
        print(ctx.matmul(a, b).shape)
        print(ctx.softmax(a).sum(axis=-1))

Build the shared library first with ``make build-for-python``.
"""
from __future__ import annotations

from ._lib import (
    NimformersError,
    abi_version,
    get_library,
    library_path,
    load_library,
)
from .core import Context, Nimformers, ResidentF32Weights, ResidentQ4Weights, cpu_info
from .model import LlamaConfig, LlamaModel
from .quant import KIND_IDS, QuantKind, QuantTensor
from .state import StateDict
from .tensor import Tensor
from .tokenizer import SentencePieceTokenizer, Tokenizer
from .vm import Op, VMProgram

__version__ = "2.0.0"
__all__ = [
    "Context",
    "Nimformers",
    "Tensor",
    "QuantTensor",
    "QuantKind",
    "KIND_IDS",
    "StateDict",
    "Tokenizer",
    "SentencePieceTokenizer",
    "LlamaConfig",
    "LlamaModel",
    "VMProgram",
    "Op",
    "NimformersError",
    "ResidentF32Weights",
    "ResidentQ4Weights",
    "abi_version",
    "cpu_info",
    "library_path",
    "load_library",
    "get_library",
    "__version__",
]
