"""Backwards-compatible import location for the original ctypes wrapper.

New code should use :mod:`nimformers` directly::

    from nimformers import Context
"""
from ._lib import NimformersError
from .core import Context as Nimformers, Context
from .quant import QuantTensor
from .state import StateDict
from .tensor import Tensor as NimTensor, Tensor
from .vm import VMProgram

__all__ = [
    "Nimformers",
    "Context",
    "NimTensor",
    "Tensor",
    "QuantTensor",
    "StateDict",
    "VMProgram",
    "NimformersError",
]
