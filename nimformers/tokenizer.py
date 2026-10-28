"""Hugging Face and SentencePiece tokenizers backed by Nimformers."""
from __future__ import annotations

import ctypes as ct
from typing import Sequence

import numpy as np

from ._lib import I32, I64, NimformersError

__all__ = ["Tokenizer", "SentencePieceTokenizer"]


class _TokenizerBase:
    _destroy_name = "nimf_tokenizer_destroy"
    _encode_name = "nimf_tokenizer_encode"
    _decode_name = "nimf_tokenizer_decode"

    def __init__(self, lib, handle):
        if not handle:
            raise NimformersError("failed to load tokenizer")
        self.lib = lib
        self.handle = handle

    # -- helpers --------------------------------------------------------
    def _ids_out(self, count: int) -> np.ndarray:
        return np.empty(max(count, 1), dtype=np.int32)

    def _decode_string(self, ptr) -> str:
        if not ptr:
            self._raise()
        try:
            return ct.string_at(ptr).decode("utf-8", "replace")
        finally:
            self.lib.nimf_free_string(ptr)

    def _raise(self):
        msg = self.lib.nimf_last_error()
        raise NimformersError(
            msg.decode("utf-8", "replace") if msg else "tokenizer operation failed")

    # -- lifetime -------------------------------------------------------
    def close(self) -> None:
        if getattr(self, "handle", None):
            getattr(self.lib, self._destroy_name)(self.handle)
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


class Tokenizer(_TokenizerBase):
    """A Hugging Face tokenizer directory (BPE or SentencePiece)."""

    @classmethod
    def load(cls, lib, model_dir) -> "Tokenizer":
        handle = lib.nimf_tokenizer_load(str(model_dir).encode())
        if not handle:
            msg = lib.nimf_last_error()
            raise NimformersError(
                msg.decode("utf-8", "replace") if msg else "tokenizer load failed")
        return cls(lib, handle)

    @property
    def vocab_size(self) -> int:
        return int(self.lib.nimf_tokenizer_vocab_size(self.handle))

    @property
    def total_vocab_size(self) -> int:
        return int(self.lib.nimf_tokenizer_total_vocab_size(self.handle))

    @property
    def special_ids(self) -> tuple[int, int, int]:
        bos, eos, pad = (np.zeros(1, np.int64) for _ in range(3))
        status = self.lib.nimf_tokenizer_special_ids(
            self.handle,
            bos.ctypes.data_as(I64), eos.ctypes.data_as(I64),
            pad.ctypes.data_as(I64))
        if status != 0:
            self._raise()
        return int(bos[0]), int(eos[0]), int(pad[0])

    @property
    def bos_id(self) -> int:
        return self.special_ids[0]

    @property
    def eos_id(self) -> int:
        return self.special_ids[1]

    @property
    def pad_id(self) -> int:
        return self.special_ids[2]

    def encode(self, text: str, add_special_tokens: bool = True) -> np.ndarray:
        n = int(self.lib.nimf_tokenizer_encode(
            self.handle, str(text).encode(), 1 if add_special_tokens else 0,
            None, 0))
        if n < 0:
            self._raise()
        out = self._ids_out(n)
        got = int(self.lib.nimf_tokenizer_encode(
            self.handle, str(text).encode(), 1 if add_special_tokens else 0,
            out.ctypes.data_as(I32), n))
        if got < 0:
            self._raise()
        return out[:got].copy()

    def apply_chat_template(self, user_text: str,
                            system_text: str = "") -> np.ndarray:
        n = int(self.lib.nimf_tokenizer_apply_chat(
            self.handle, str(user_text).encode(), str(system_text).encode(),
            None, 0))
        if n < 0:
            self._raise()
        out = self._ids_out(n)
        got = int(self.lib.nimf_tokenizer_apply_chat(
            self.handle, str(user_text).encode(), str(system_text).encode(),
            out.ctypes.data_as(I32), n))
        if got < 0:
            self._raise()
        return out[:got].copy()

    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        arr = np.ascontiguousarray(ids, dtype=np.int32).reshape(-1)
        ptr = self.lib.nimf_tokenizer_decode(
            self.handle, arr.ctypes.data_as(I32), arr.size,
            1 if skip_special_tokens else 0)
        return self._decode_string(ptr)

    def __len__(self) -> int:
        return self.vocab_size

    def __repr__(self) -> str:
        try:
            return f"Tokenizer(vocab_size={self.vocab_size})"
        except Exception:
            return "Tokenizer(<closed>)"


class SentencePieceTokenizer(_TokenizerBase):
    """A raw SentencePiece ``tokenizer.model`` file."""

    _destroy_name = "nimf_sentencepiece_destroy"
    _encode_name = "nimf_sentencepiece_encode"
    _decode_name = "nimf_sentencepiece_decode"

    @classmethod
    def load(cls, lib, model_path) -> "SentencePieceTokenizer":
        handle = lib.nimf_sentencepiece_load(str(model_path).encode())
        if not handle:
            msg = lib.nimf_last_error()
            raise NimformersError(
                msg.decode("utf-8", "replace") if msg else "SentencePiece load failed")
        return cls(lib, handle)

    @property
    def normalizer(self) -> str:
        ptr = self.lib.nimf_sentencepiece_normalizer(self.handle)
        return self._decode_string(ptr)

    def encode(self, text: str) -> np.ndarray:
        n = int(self.lib.nimf_sentencepiece_encode(
            self.handle, str(text).encode(), None, 0))
        if n < 0:
            self._raise()
        out = self._ids_out(n)
        got = int(self.lib.nimf_sentencepiece_encode(
            self.handle, str(text).encode(), out.ctypes.data_as(I32), n))
        if got < 0:
            self._raise()
        return out[:got].copy()

    def decode(self, ids: Sequence[int]) -> str:
        arr = np.ascontiguousarray(ids, dtype=np.int32).reshape(-1)
        ptr = self.lib.nimf_sentencepiece_decode(
            self.handle, arr.ctypes.data_as(I32), arr.size)
        return self._decode_string(ptr)

    def piece_id(self, piece: str) -> int:
        return int(self.lib.nimf_sentencepiece_piece_id(
            self.handle, str(piece).encode()))

    def __repr__(self) -> str:
        return "SentencePieceTokenizer()"
