"""Low-level ctypes loader and prototypes for the Nimformers C ABI.

The shared library is looked up in this order:

1. an explicit path passed to :func:`load_library`
2. the ``NIMFORMERS_LIB`` environment variable
3. the ``nimformers`` package directory
4. the repository root (one level above the package)
5. the current working directory and ``build/``
6. the platform's dynamic loader (``find_library("nimformers")``)

Build the library with ``make build-for-python``.
"""
from __future__ import annotations

import ctypes as ct
import os
import platform
from ctypes.util import find_library
from pathlib import Path
from typing import Optional

__all__ = [
    "load_library",
    "get_library",
    "library_path",
    "abi_version",
    "NimformersError",
    "F32",
    "I32",
    "I64",
    "U8",
    "U32",
]


class NimformersError(RuntimeError):
    """Raised when the Nimformers C ABI reports an error."""


# ctypes shorthands used by the prototype table
V = ct.c_void_p
F32 = ct.POINTER(ct.c_float)
I32 = ct.POINTER(ct.c_int32)
I64 = ct.POINTER(ct.c_int64)
U8 = ct.POINTER(ct.c_uint8)
U32 = ct.POINTER(ct.c_uint32)
CINT = ct.c_int
CI64 = ct.c_int64
CU32 = ct.c_uint32
CF32 = ct.c_float
CSTR = ct.c_char_p
PINT = ct.POINTER(ct.c_int)


def _library_filename() -> str:
    system = platform.system()
    if system == "Darwin":
        return "libnimformers.dylib"
    if system == "Windows":
        return "nimformers.dll"
    return "libnimformers.so"


def _candidate_paths(explicit: Optional[str] = None):
    if explicit:
        yield Path(explicit).expanduser()
        return
    env = os.environ.get("NIMFORMERS_LIB")
    if env:
        yield Path(env).expanduser()
    name = _library_filename()
    package_dir = Path(__file__).resolve().parent
    repo_root = package_dir.parent
    cwd = Path.cwd()
    for root in (package_dir, repo_root, cwd, cwd / "build", repo_root / "build"):
        yield root / name
    # Also accept the other Unix extension when cross-building.
    for root in (package_dir, repo_root, cwd):
        for alt in ("libnimformers.dylib", "libnimformers.so", "nimformers.dll"):
            yield root / alt


def library_path(explicit: Optional[str] = None) -> Path:
    """Return the first existing library path, or raise a helpful error."""
    if explicit:
        candidate = Path(explicit).expanduser()
        if candidate.is_file():
            return candidate
        raise NimformersError(f"shared library not found at {candidate}")
    for candidate in _candidate_paths():
        if candidate.is_file():
            return candidate
    found = find_library("nimformers")
    if found:
        return Path(found)
    name = _library_filename()
    raise NimformersError(
        f"could not find {name}. Build it with `make build-for-python` "
        "or set NIMFORMERS_LIB to the full path of the shared library."
    )


_LIB_CACHE: dict[str, ct.CDLL] = {}


def get_library(explicit: Optional[str] = None) -> ct.CDLL:
    """Load (and cache) the shared library, binding all prototypes."""
    path = library_path(explicit)
    key = str(path.resolve())
    lib = _LIB_CACHE.get(key)
    if lib is None:
        try:
            lib = ct.CDLL(key)
        except OSError as exc:  # pragma: no cover - platform dependent
            raise NimformersError(f"failed to load {key}: {exc}") from exc
        _bind(lib)
        _LIB_CACHE[key] = lib
    return lib


def load_library(explicit: Optional[str] = None) -> ct.CDLL:
    """Alias for :func:`get_library`; kept for symmetry with the C API."""
    return get_library(explicit)


def _fn(lib, name, args, restype):
    try:
        fn = getattr(lib, name)
    except AttributeError as exc:
        raise NimformersError(
            f"shared library is missing `{name}`; rebuild with `make build-for-python`"
        ) from exc
    fn.argtypes = args
    fn.restype = restype
    return fn


def _bind(lib: ct.CDLL) -> None:
    # ---- runtime -------------------------------------------------------
    _fn(lib, "nimf_abi_version", [], ct.c_uint32)
    _fn(lib, "nimf_version", [], CSTR)
    _fn(lib, "nimf_last_error", [], CSTR)
    _fn(lib, "nimf_clear_error", [], None)
    _fn(lib, "nimf_free_string", [V], None)
    _fn(lib, "nimf_cpu_isa", [], CINT)
    _fn(lib, "nimf_cpu_isa_name", [], CSTR)
    _fn(lib, "nimf_cpu_features", [], CU32)
    _fn(lib, "nimf_cpu_cache_sizes", [I64, I64, I64], CINT)
    _fn(lib, "nimf_cpu_block_sizes", [I64, I64, I64], CINT)

    _fn(lib, "nimf_context_create", [CINT], V)
    _fn(lib, "nimf_context_create_name", [CSTR], V)
    _fn(lib, "nimf_context_destroy", [V], CINT)
    _fn(lib, "nimf_context_backend", [V], CINT)
    _fn(lib, "nimf_context_backend_name", [V], CSTR)
    _fn(lib, "nimf_context_sync", [V], CINT)
    _fn(lib, "nimf_context_set_cpu_fallback", [V, CINT], CINT)

    # ---- tensors -------------------------------------------------------
    _fn(lib, "nimf_tensor_create", [F32, CI64], V)
    _fn(lib, "nimf_tensor_create_shaped", [F32, CI64, I64, CI64], V)
    _fn(lib, "nimf_tensor_zeros", [I64, CI64, CF32], V)
    _fn(lib, "nimf_tensor_randn", [I64, CI64, CF32], V)
    _fn(lib, "nimf_tensor_clone", [V], V)
    _fn(lib, "nimf_tensor_destroy", [V], CINT)
    _fn(lib, "nimf_tensor_rank", [V], CI64)
    _fn(lib, "nimf_tensor_numel", [V], CI64)
    _fn(lib, "nimf_tensor_shape", [V, I64, CI64], CI64)
    _fn(lib, "nimf_tensor_data", [V], F32)
    _fn(lib, "nimf_tensor_len", [V], CI64)
    _fn(lib, "nimf_tensor_set_data", [V, F32, CI64], CINT)
    _fn(lib, "nimf_tensor_fill", [V, CF32], CINT)
    _fn(lib, "nimf_tensor_reshape", [V, I64, CI64], CINT)

    # ---- elementwise ---------------------------------------------------
    for name in ("nimf_add", "nimf_sub", "nimf_mul", "nimf_div"):
        _fn(lib, name, [V, F32, F32, F32, CI64], CINT)
    for name in ("nimf_relu", "nimf_sqrt", "nimf_exp", "nimf_log",
                 "nimf_sigmoid", "nimf_tanh", "nimf_neg"):
        _fn(lib, name, [V, F32, F32, CI64], CINT)
    _fn(lib, "nimf_pow", [V, F32, F32, CI64, CF32], CINT)
    _fn(lib, "nimf_add_scalar", [V, F32, F32, CI64, CF32], CINT)
    _fn(lib, "nimf_mul_scalar", [V, F32, F32, CI64, CF32], CINT)
    _fn(lib, "nimf_apflu", [V, F32, F32, CI64, CF32, CF32], CINT)
    _fn(lib, "nimf_apflu_backward", [V, F32, F32, F32, CI64, CF32, CF32], CINT)

    # ---- matmul --------------------------------------------------------
    _fn(lib, "nimf_matmul", [V, F32, F32, F32, CI64, CI64, CI64], CINT)
    _fn(lib, "nimf_matmul2", [V, F32, F32, F32, CI64, CI64, CI64,
                              F32, F32, F32, CI64, CI64, CI64], CINT)
    _fn(lib, "nimf_matmul_q4", [V, F32, CI64, U8, CI64, F32, F32, CI64, CI64,
                                F32, CI64, CI64, CI64, CI64, CI64], CINT)
    _fn(lib, "nimf_matmul_nt", [V, F32, F32, F32, CI64, CI64, CI64], CINT)

    # ---- resident weights ---------------------------------------------
    _fn(lib, "nimf_resident_q4_upload", [V, U8, CI64, F32, F32, CI64, CI64,
                                         CI64, CI64, CI64, CI64], V)
    _fn(lib, "nimf_resident_q4_linear", [V, V, F32, CI64, CI64, F32], CINT)
    _fn(lib, "nimf_resident_q4_destroy", [V, V], CINT)
    _fn(lib, "nimf_resident_f32_upload", [V, F32, CI64], V)
    _fn(lib, "nimf_resident_f32_linear", [V, V, F32, CI64, CI64, CI64, CI64, F32], CINT)
    _fn(lib, "nimf_resident_f32_destroy", [V, V], CINT)

    # ---- neural-network kernels ---------------------------------------
    _fn(lib, "nimf_softmax", [V, F32, F32, CI64, CI64], CINT)
    _fn(lib, "nimf_layernorm", [V, F32, F32, F32, F32, CI64, CI64, CF32], CINT)
    _fn(lib, "nimf_add_layernorm", [V, F32, F32, F32, F32, F32, F32,
                                    CI64, CI64, CF32], CINT)
    _fn(lib, "nimf_layernorm_backward", [V, F32, F32, F32, F32, CI64, CI64,
                                         CF32, F32, F32, F32], CINT)
    _fn(lib, "nimf_attention", [V, F32, F32, F32, F32, CI64, CI64, CI64,
                                CI64, CI64, CF32, F32, F32], CINT)
    _fn(lib, "nimf_attention_backward", [V, F32, F32, F32, F32, F32, CI64,
                                         CI64, CI64, CI64, CF32, F32, F32, F32], CINT)
    _fn(lib, "nimf_embedding_lookup", [V, F32, CI64, I32, CI64, F32, CI64, CI64], CINT)
    _fn(lib, "nimf_gather", [V, F32, CI64, I32, CI64, F32], CINT)
    _fn(lib, "nimf_compare", [V, F32, F32, CI64, CINT, U8], CINT)
    _fn(lib, "nimf_dropout", [V, F32, F32, CI64, CF32, CINT, CU32], CINT)
    _fn(lib, "nimf_dropout_mask", [V, CI64, CF32, CU32, F32], CINT)
    _fn(lib, "nimf_argmax", [V, F32, CI64, CI64, CI64, CI64, F32, I64], CINT)
    _fn(lib, "nimf_masked_fill", [F32, U8, CI64, CF32, F32], CINT)
    _fn(lib, "nimf_where", [U8, F32, F32, CI64, F32], CINT)

    # ---- reductions / scans -------------------------------------------
    for name in ("nimf_sum", "nimf_mean", "nimf_max"):
        _fn(lib, name, [V, F32, CI64, I64, CI64, CI64, F32], CINT)
    for name in ("nimf_cumsum", "nimf_cumprod"):
        _fn(lib, name, [V, F32, CI64, I64, CI64, CI64, F32], CINT)
    _fn(lib, "nimf_affine_scan", [V, F32, F32, CI64, I64, CI64, CI64, F32], CINT)

    # ---- shape movement -----------------------------------------------
    _fn(lib, "nimf_permute", [V, F32, CI64, I32, I32, I32, CI64, F32], CINT)
    _fn(lib, "nimf_split_slice", [V, F32, CI64, CI64, CI64, CI64, CI64, F32], CINT)
    _fn(lib, "nimf_concat", [V, F32, CI64, I32, I32, I32, CI64, CI64, CI64,
                             CI64, F32], CINT)

    # ---- quantization --------------------------------------------------
    _fn(lib, "nimf_quantize", [F32, CI64, I64, CI64, CINT, CINT, CINT], V)
    _fn(lib, "nimf_quant_dequantize", [V, F32, CI64], CI64)
    _fn(lib, "nimf_quant_dequantize_transposed", [V, F32, CI64], CI64)
    _fn(lib, "nimf_quant_kind", [V], CINT)
    _fn(lib, "nimf_quant_rank", [V], CI64)
    _fn(lib, "nimf_quant_shape", [V, I64, CI64], CI64)
    _fn(lib, "nimf_quant_numel", [V], CI64)
    _fn(lib, "nimf_quant_group_size", [V], CI64)
    _fn(lib, "nimf_quant_cf_bits", [V, PINT, PINT], CINT)
    _fn(lib, "nimf_quant_data_len", [V], CI64)
    _fn(lib, "nimf_quant_data", [V], U8)
    _fn(lib, "nimf_quant_scale_len", [V], CI64)
    _fn(lib, "nimf_quant_scale", [V], F32)
    _fn(lib, "nimf_quant_zero_len", [V], CI64)
    _fn(lib, "nimf_quant_zero", [V], F32)
    _fn(lib, "nimf_quant_destroy", [V], CINT)

    # ---- NIMQ state dicts ---------------------------------------------
    _fn(lib, "nimf_state_dict_load", [CSTR, CINT], V)
    _fn(lib, "nimf_state_dict_create", [], V)
    _fn(lib, "nimf_state_dict_append", [V, CSTR, V], CINT)
    _fn(lib, "nimf_state_dict_set_arch", [V, I64], CINT)
    _fn(lib, "nimf_state_dict_save", [CSTR, V, I64], CINT)
    _fn(lib, "nimf_state_dict_count", [V], CI64)
    _fn(lib, "nimf_state_dict_arch", [V, I64, CI64], CINT)
    _fn(lib, "nimf_state_dict_name", [V, CI64], V)
    _fn(lib, "nimf_state_dict_tensor", [V, CI64], V)
    _fn(lib, "nimf_state_dict_destroy", [V], CINT)

    # ---- custom floats / APF ------------------------------------------
    _fn(lib, "nimf_customfloat_encode", [V, F32, CI64, CINT, CINT, U8, CI64, I64], CINT)
    _fn(lib, "nimf_customfloat_decode", [V, U8, CI64, CI64, CINT, CINT, F32], CINT)
    _fn(lib, "nimf_apf_build_dtype", [F32, CI64, F32, CI64, CF32, CINT, PINT, PINT], CINT)
    _fn(lib, "nimf_apf_cast_training", [F32, CI64, F32, CI64, CF32, CINT, U8,
                                        CI64, I64, PINT, PINT], CINT)
    _fn(lib, "nimf_apf_decode_training", [U8, CI64, CINT, CINT, F32, CI64], CI64)

    # ---- Llama inference ----------------------------------------------
    _fn(lib, "nimf_llama_config_load", [CSTR, I64, F32], CINT)
    _fn(lib, "nimf_llama_load", [V, CSTR, CSTR, CINT], V)
    _fn(lib, "nimf_llama_config", [V, I64, F32], CINT)
    _fn(lib, "nimf_llama_generate", [V, V, V, CSTR, CSTR, CI64, CF32, CF32, CINT], V)
    _fn(lib, "nimf_llama_destroy", [V], CINT)

    # ---- tokenizers ----------------------------------------------------
    _fn(lib, "nimf_tokenizer_load", [CSTR], V)
    _fn(lib, "nimf_tokenizer_destroy", [V], CINT)
    _fn(lib, "nimf_tokenizer_vocab_size", [V], CI64)
    _fn(lib, "nimf_tokenizer_total_vocab_size", [V], CI64)
    _fn(lib, "nimf_tokenizer_special_ids", [V, I64, I64, I64], CINT)
    _fn(lib, "nimf_tokenizer_encode", [V, CSTR, CINT, I32, CI64], CI64)
    _fn(lib, "nimf_tokenizer_apply_chat", [V, CSTR, CSTR, I32, CI64], CI64)
    _fn(lib, "nimf_tokenizer_decode", [V, I32, CI64, CINT], V)
    _fn(lib, "nimf_sentencepiece_load", [CSTR], V)
    _fn(lib, "nimf_sentencepiece_destroy", [V], CINT)
    _fn(lib, "nimf_sentencepiece_encode", [V, CSTR, I32, CI64], CI64)
    _fn(lib, "nimf_sentencepiece_decode", [V, I32, CI64], V)
    _fn(lib, "nimf_sentencepiece_piece_id", [V, CSTR], CI64)
    _fn(lib, "nimf_sentencepiece_normalizer", [V], V)

    # ---- micro-VM ------------------------------------------------------
    _fn(lib, "nimf_vm_create", [], V)
    _fn(lib, "nimf_vm_destroy", [V], CINT)
    _fn(lib, "nimf_vm_len", [V], CI64)
    _fn(lib, "nimf_vm_words", [V, U32, CI64], CI64)
    _fn(lib, "nimf_vm_set_words", [V, U32, CI64], CINT)
    _fn(lib, "nimf_vm_emit_word", [V, CU32], CINT)
    _fn(lib, "nimf_vm_emit", [V, CU32, CU32, CU32, CU32, CU32], CINT)
    _fn(lib, "nimf_vm_save", [V, CSTR], CINT)
    _fn(lib, "nimf_vm_load", [CSTR], V)
    _fn(lib, "nimf_vm_merge", [ct.POINTER(V), CI64], V)
    _fn(lib, "nimf_vm_lower_msl", [V, CSTR], V)
    _fn(lib, "nimf_vm_lower_glsl", [V, CSTR], V)
    _fn(lib, "nimf_vm_key", [V, CSTR], V)
    _fn(lib, "nimf_vm_run", [V, V, F32, CI64, CI64, CI64, CI64], CINT)
    _fn(lib, "nimf_micro_vm_dispatch", [V, U32, CI64, F32, CI64, CI64, CI64, CI64], CINT)


def abi_version(lib: Optional[ct.CDLL] = None) -> int:
    """Return the numeric ABI version of the loaded library."""
    lib = lib or get_library()
    return int(lib.nimf_abi_version())


def last_error(lib: Optional[ct.CDLL] = None) -> str:
    lib = lib or get_library()
    msg = lib.nimf_last_error()
    return msg.decode("utf-8", "replace") if msg else ""


def free_string(lib: ct.CDLL, value) -> None:
    """Free a string allocated by the library."""
    if value:
        lib.nimf_free_string(value)
