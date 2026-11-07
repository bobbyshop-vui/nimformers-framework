"""Stdlib-only symbolic model tracing, reference evaluation, and FIR export."""

from . import nn
from . import dtypes
from .codegen import LoweringError
from .emitter import emit, export
from .graph import (Graph, Node, Value, Weight, eliminate_dead_code, fuse_elementwise,
                    constant_fold, common_subexpressions, fold_views, select_q4,
                    deduplicate_kernels, optimize_graph, PASSES)
from .patterns import (fuse_attention, fuse_rotary_attention, push_slice_into_conv, fuse_conv_state,
                       fuse_pointwise, fuse_state_update_reduction, fuse_reduction_prologue, fuse_norms)
from .refusion import (refuse_norms, refuse_linear, refuse_rotary, refuse_activations,
                       refuse_additive_masks)
from .tensor import (TensorSpec, Dim, ProxyTensor, Size, Device, Unsupported, tensor, matmul, softmax,
                     log_softmax, silu, gelu, gelu_new, relu, tanh, sigmoid, softplus, exp, log, sqrt,
                     rsqrt, sin, cos, abs, neg, maximum, minimum, clamp, sum, mean, max, min, amax, amin,
                     argmax, argmin, var, std, logsumexp, reshape, transpose, permute, chunk, split, cat,
                     stack, where, masked_fill, rotary_embedding, scaled_dot_product_attention, cumsum,
                     scan, arange, zeros, ones, full, zeros_like, ones_like, full_like, eye, tril, triu,
                     flip, roll, outer, einsum, repeat_interleave, index_select, gather, static_values)
from .dtypes import (float32, float16, bfloat16, float64, int64, int32, int16, int8, uint8, bool_,
                     finfo, iinfo, promote_types)
from .nn import (Module, Parameter, ModuleList, ModuleDict, ParameterList, ParameterDict, Sequential,
                 Linear, Embedding, LayerNorm, RMSNorm, Conv1d, Dropout, Identity, GELU)
from .tracer import trace, script
from .reference import evaluate, random_parameters, compare_graphs
from .report import report, fir_stats
from .vm import FirModel, FirVmError
from .torch_adapter import from_torch
from . import torch_compat
from . import torch_compat as torch
