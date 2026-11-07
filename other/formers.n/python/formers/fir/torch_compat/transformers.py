"""The pieces of HF ``transformers`` that modeling files import, over ``torch_compat``.

``PretrainedConfig`` (attribute access over config.json, ``attribute_map`` aliases),
``ACT2FN``, ``Conv1D``, ``PreTrainedModel``, ``ModelOutput`` classes,
``ROPE_INIT_FUNCTIONS["default"]`` and a ``StaticCache`` whose per-layer K/V
storage are fir states written at ``cache_position`` (one token per step, the
FIR decode ABI), so HF-style attention traces into ``attention_decode``.
"""

from collections import OrderedDict
import copy
import json
import math
from pathlib import Path

from . import nn
from .nn import functional as F
from ..tensor import Unsupported, arange, tanh, sigmoid, pow as _pow
from .. import dtypes as D


# ---- configuration -----------------------------------------------------------------------

_ALIASES = {
    "hidden_size": ("n_embd", "d_model", "dim"),
    "num_attention_heads": ("n_head", "num_heads", "n_heads"),
    "num_hidden_layers": ("n_layer", "num_layers", "n_layers"),
    "max_position_embeddings": ("n_positions", "n_ctx", "max_seq_len"),
    "intermediate_size": ("n_inner", "ffn_dim"),
    "layer_norm_epsilon": ("layer_norm_eps", "norm_epsilon"),
}

_DEFAULTS = {
    "_attn_implementation": "eager", "output_attentions": False, "output_hidden_states": False,
    "use_return_dict": True, "return_dict": True, "use_cache": True, "torchscript": False,
    "is_encoder_decoder": False, "tie_word_embeddings": True, "pruned_heads": {}, "torch_dtype": None,
    "pad_token_id": None, "bos_token_id": None, "eos_token_id": None, "rope_scaling": None,
    "attention_dropout": 0.0, "pretraining_tp": 1, "attention_bias": False, "mlp_bias": False,
    "_name_or_path": "", "architectures": None, "problem_type": None,
}


class PretrainedConfig:
    """Attribute access over a config dict; ``to_dict()`` returns the given keys only (fir.trace config)."""

    model_type = ""
    attribute_map = {}

    def __init__(self, **kwargs):
        object.__setattr__(self, "_keys", [])
        for k, v in kwargs.items(): setattr(self, k, v)

    def __setattr__(self, name, value):
        name = type(self).attribute_map.get(name, name)
        if not name.startswith("_") and name not in self._keys: self._keys.append(name)
        object.__setattr__(self, name, value)

    def __getattr__(self, name):
        if name.startswith("__"): raise AttributeError(name)
        mapped = type(self).attribute_map.get(name)
        if mapped is not None and mapped in self.__dict__: return self.__dict__[mapped]
        for alias in _ALIASES.get(name, ()):
            if alias in self.__dict__: return self.__dict__[alias]
        for canonical, aliases in _ALIASES.items():
            if name in aliases and canonical in self.__dict__: return self.__dict__[canonical]
        if name == "num_key_value_heads": return self.num_attention_heads
        if name == "head_dim":
            try: return self.hidden_size // self.num_attention_heads
            except AttributeError: pass
        if name in _DEFAULTS: return copy.copy(_DEFAULTS[name])
        raise AttributeError("%s has no attribute %r" % (type(self).__name__, name))

    def __contains__(self, key): return key in self.__dict__
    def get(self, key, default=None): return getattr(self, key, default)
    def to_dict(self): return {k: self.__dict__[k] for k in self._keys}
    def to_json_string(self): return json.dumps(self.to_dict(), indent=2, sort_keys=True)
    def __repr__(self): return "%s %s" % (type(self).__name__, self.to_json_string())

    @classmethod
    def from_dict(cls, config, **kwargs):
        return cls(**dict(config, **kwargs))

    @classmethod
    def from_json_file(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text()))

    @classmethod
    def from_pretrained(cls, path, **kwargs):
        path = Path(path)
        return cls.from_dict(json.loads((path / "config.json" if path.is_dir() else path).read_text()), **kwargs)


AutoConfig = PretrainedConfig


# ---- activations -----------------------------------------------------------------------------

class NewGELUActivation(nn.Module):
    """HF gelu_new: the tanh approximation written as in transformers (re-fused into gelu_new)."""
    def forward(self, input):
        return 0.5 * input * (1.0 + tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * _pow(input, 3.0))))


class FastGELUActivation(nn.Module):
    def forward(self, input):
        return 0.5 * input * (1.0 + tanh(input * 0.7978845608 * (1.0 + 0.044715 * input * input)))


class QuickGELUActivation(nn.Module):
    def forward(self, input): return input * sigmoid(1.702 * input)


class GELUActivation(nn.Module):
    def forward(self, input): return F.gelu(input)


class PytorchGELUTanh(nn.Module):
    def forward(self, input): return F.gelu(input, approximate="tanh")


class MishActivation(nn.Module):
    def forward(self, input): return F.mish(input)


class LinearActivation(nn.Module):
    def forward(self, input): return input


class ClassInstantier(OrderedDict):
    def __getitem__(self, key):
        content = super().__getitem__(key)
        cls, kwargs = content if isinstance(content, tuple) else (content, {})
        return cls(**kwargs)


ACT2CLS = {
    "gelu": GELUActivation, "gelu_new": NewGELUActivation, "gelu_fast": FastGELUActivation,
    "gelu_pytorch_tanh": PytorchGELUTanh, "quick_gelu": QuickGELUActivation, "gelu_python": GELUActivation,
    "linear": LinearActivation, "mish": MishActivation, "relu": nn.ReLU, "relu6": nn.ReLU6,
    "leaky_relu": nn.LeakyReLU, "sigmoid": nn.Sigmoid, "silu": nn.SiLU, "swish": nn.SiLU, "tanh": nn.Tanh,
    "elu": nn.ELU, "selu": nn.SELU, "softplus": nn.Softplus,
}
ACT2FN = ClassInstantier(ACT2CLS)


def get_activation(name): return ACT2FN[name]


# ---- layers --------------------------------------------------------------------------------------

class Conv1D(nn.Module):
    """HF pytorch_utils.Conv1D (GPT-2): weight ``[nx, nf]``, ``addmm(bias, x, weight)``."""
    def __init__(self, nf, nx):
        super().__init__()
        self.nf, self.nx = nf, nx
        self.weight = nn.Parameter(_empty(nx, nf))
        self.bias = nn.Parameter(_empty(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        from . import addmm
        size_out = x.size()[:-1] + (self.nf,)
        x = addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        return x.view(size_out)


def _empty(*size):
    from . import empty
    return empty(*size)


class PreTrainedModel(nn.Module):
    config_class = PretrainedConfig
    base_model_prefix = "model"
    main_input_name = "input_ids"
    _no_split_modules = []
    supports_gradient_checkpointing = False

    def __init__(self, config, *inputs, **kwargs):
        super().__init__()
        self.config = config

    def post_init(self): pass
    def _init_weights(self, module): pass
    def tie_weights(self): pass
    def gradient_checkpointing_enable(self, *a, **k): pass

    @classmethod
    def from_config(cls, config): return cls(config)


class GradientCheckpointingLayer(nn.Module):
    gradient_checkpointing = False


# ---- outputs --------------------------------------------------------------------------------------

class ModelOutput(OrderedDict):
    """Attribute and key access to the non-None fields, in declaration order."""
    _fields = ()

    def __init__(self, *args, **kwargs):
        super().__init__()
        values = dict(zip(self._fields, args), **kwargs)
        for k in self._fields:
            v = values.get(k)
            object.__setattr__(self, k, v)
            if v is not None: OrderedDict.__setitem__(self, k, v)

    def __getitem__(self, k):
        return self.to_tuple()[k] if isinstance(k, (int, slice)) else OrderedDict.__getitem__(self, k)

    def to_tuple(self): return tuple(self[k] for k in self.keys())


def _output(name, fields):
    return type(name, (ModelOutput,), {"_fields": tuple(fields)})


BaseModelOutputWithPast = _output("BaseModelOutputWithPast", ("last_hidden_state", "past_key_values", "hidden_states", "attentions"))
CausalLMOutputWithPast = _output("CausalLMOutputWithPast", ("loss", "logits", "past_key_values", "hidden_states", "attentions"))
BaseModelOutputWithPastAndCrossAttentions = _output("BaseModelOutputWithPastAndCrossAttentions",
                                                    ("last_hidden_state", "past_key_values", "hidden_states", "attentions", "cross_attentions"))
CausalLMOutputWithCrossAttentions = _output("CausalLMOutputWithCrossAttentions",
                                            ("loss", "logits", "past_key_values", "hidden_states", "attentions", "cross_attentions"))


# ---- RoPE --------------------------------------------------------------------------------------------

def _compute_default_rope_parameters(config=None, device=None, seq_len=None, **kwargs):
    base = config.rope_theta if hasattr(config, "rope_theta") else 10000.0
    partial = getattr(config, "partial_rotary_factor", 1.0)
    head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
    dim = int(head_dim * partial)
    from . import int64, float32
    inv_freq = 1.0 / (base ** (arange(0, dim, 2, dtype=int64).to(dtype=float32) / dim))
    return inv_freq, 1.0


ROPE_INIT_FUNCTIONS = {"default": _compute_default_rope_parameters}


def dynamic_rope_update(fn): return fn


# ---- static KV cache -------------------------------------------------------------------------------------

class StaticLayer(nn.Module):
    """K/V storage ``[kv_heads, max_cache_len, head_dim]`` written one row per step."""
    def __init__(self, kv_heads, max_cache_len, head_dim):
        super().__init__()
        self.kv_heads, self.max_cache_len, self.head_dim = kv_heads, max_cache_len, head_dim
        self.register_state("keys", (kv_heads, max_cache_len, head_dim))
        self.register_state("values", (kv_heads, max_cache_len, head_dim))

    def update(self, key_states, value_states, cache_position):
        b, h, length, d = key_states.shape
        if b != 1 or length != 1:
            raise Unsupported("StaticCache writes one token per step (batch 1 decode ABI); got %s" % list(key_states.shape))
        position = cache_position.reshape(())
        keys = self.keys.update(key_states.reshape(h, d), position, dim=1)
        values = self.values.update(value_states.reshape(h, d), position, dim=1)
        return keys.reshape(1, h, self.max_cache_len, d), values.reshape(1, h, self.max_cache_len, d)


class StaticCache(nn.Module):
    """transformers.StaticCache for decode tracing: ``update(k, v, layer_idx, {"cache_position": p})``."""
    is_compileable = True

    def __init__(self, config, max_batch_size=1, max_cache_len=None, device=None, dtype=None, **kwargs):
        super().__init__()
        if max_batch_size != 1: raise Unsupported("StaticCache for batch > 1 (the FIR decode ABI is batch 1)")
        heads = config.num_attention_heads
        kv_heads = getattr(config, "num_key_value_heads", None) or heads
        head_dim = getattr(config, "head_dim", None) or config.hidden_size // heads
        self.max_cache_len = max_cache_len or config.max_position_embeddings
        self.layers = nn.ModuleList(StaticLayer(kv_heads, self.max_cache_len, head_dim)
                                    for _ in range(config.num_hidden_layers))

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if not cache_kwargs or cache_kwargs.get("cache_position") is None:
            raise ValueError("StaticCache.update needs cache_kwargs['cache_position']")
        return self.layers[layer_idx].update(key_states, value_states, cache_kwargs["cache_position"])

    def get_max_cache_shape(self, layer_idx=0): return self.max_cache_len
    def get_seq_length(self, layer_idx=0):
        raise Unsupported("the cache length is a runtime value; pass cache_position explicitly")
    def __len__(self): return len(self.layers)


class StaticCacheDecoder(nn.Module):
    """``forward(token, position)`` over an HF causal LM and a ``StaticCache`` (the FIR decode ABI).

    The wrapped model keeps its own module paths (``model.layers.0...``,
    ``lm_head``), so traced weight names are the checkpoint names; the cache is
    the ``past_key_values`` submodule. The model is called as transformers'
    ``generate`` calls it for one token: ``input_ids``, ``position_ids``,
    ``cache_position`` and ``past_key_values``; ``logits`` is the output.
    """

    def __init__(self, model, config=None, max_cache_len=None):
        super().__init__()
        config = config if config is not None else model.config
        self.__dict__["model"] = model
        self.past_key_values = StaticCache(config, max_cache_len=max_cache_len)

    def named_modules(self, memo=None, prefix="", remove_duplicate=True):
        if isinstance(memo, str): memo, prefix = None, memo
        memo = set() if memo is None else memo
        if remove_duplicate: memo.add(id(self))
        yield prefix, self
        yield from self.model.named_modules(memo, prefix, remove_duplicate)
        yield from self.past_key_values.named_modules(memo, prefix + ("." if prefix else "") + "past_key_values",
                                                      remove_duplicate)

    def forward(self, token, position):
        out = self.model(input_ids=token, position_ids=position, past_key_values=self.past_key_values,
                         cache_position=position.reshape(-1), use_cache=True)
        return out.logits if isinstance(out, ModelOutput) else out
