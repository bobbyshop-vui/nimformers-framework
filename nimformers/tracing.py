"""Record a model run into a FIR trace (``nf.trace``).

A trace is what ``firlift`` turns into ``model.fir``.  This module wraps the recorder of the
shared library::

    import nimformers as nf

    with nf.trace("run.fprobe", steps="decode:2"):
        for pos, token in enumerate(tokens):
            logits = model.step(token)          # Session.step() marks every step

    # run.fprobe (+ run.fprobe.data) now exist; lift them with firlift.

``trace(...)`` starts recording immediately and returns a :class:`Trace`, so it works both as a
context manager and as ``t = nf.trace(...); ...; t.stop()``.
"""
from __future__ import annotations

import os
from typing import Optional

from ._lib import NimformersError, get_library
from .session import _LIVE_SESSIONS

__all__ = ["Trace", "trace", "trace_active", "trace_file", "trace_stop"]


def _raise(lib):
    msg = lib.nimf_last_error()
    raise NimformersError(msg.decode("utf-8", "replace") if msg else "trace operation failed")


class Trace:
    """A running recording.  Created by :func:`trace`; stop it with :meth:`stop` or ``with``."""

    def __init__(self, path, steps: str = "", data: bool = True, data_max: int = 0):
        self.lib = get_library()
        self.path = os.fspath(path)
        self.steps = steps
        self.data = bool(data)
        self.data_max = int(data_max)
        if self.data_max < 0:
            raise ValueError("data_max must be >= 0")
        rc = self.lib.nimf_trace_start(
            self.path.encode(), steps.encode(), 1 if self.data else 0, self.data_max)
        if rc != 0:
            _raise(self.lib)
        self._stopped = False
        # Kernels compiled before the recording started carry no bytecode in it; compile them again.
        for sess in list(_LIVE_SESSIONS):
            sess._reregister()

    @property
    def active(self) -> bool:
        """True while this recording is the active one."""
        return not self._stopped and bool(self.lib.nimf_trace_active())

    @property
    def data_path(self) -> Optional[str]:
        """Path of the exact-bytes sidecar, or ``None`` when ``data=False``."""
        return self.path + ".data" if self.data else None

    def stop(self) -> None:
        """Flush ``path`` and close the sidecar.  Calling it twice is harmless."""
        if self._stopped:
            return
        self._stopped = True
        if self.lib.nimf_trace_stop() != 0:
            _raise(self.lib)

    def __enter__(self) -> "Trace":
        return self

    def __exit__(self, *_exc) -> None:
        self.stop()

    def __repr__(self) -> str:
        return f"Trace(path={self.path!r}, steps={self.steps!r}, active={self.active})"


def trace(path, steps: str = "", data: bool = True, data_max: int = 0) -> Trace:
    """Start recording a FIR trace at ``path`` and return the running :class:`Trace`.

    :param path: trace file; its directory must exist.  ``<path>.data`` receives exact upload bytes.
    :param steps: steps to record, e.g. ``"decode:2"`` or ``"prefill:8,decode:2"``; ``""`` = all.
        The lifter needs at least two decode steps to tell repeating state from changing state.
    :param data: also write the exact bytes of every host upload (needed for weights and tables
        that are not in a checkpoint).
    :param data_max: with ``data``, skip uploads larger than this many bytes (0 = unlimited).
    """
    return Trace(path, steps, data, data_max)


def trace_active() -> bool:
    """True while any trace is being recorded in this process."""
    return bool(get_library().nimf_trace_active())


def trace_file() -> str:
    """Path of the trace of this process, or ``""`` when none was started."""
    raw = get_library().nimf_trace_file()
    return raw.decode() if raw else ""


def trace_stop() -> None:
    """Stop whatever trace is active (no-op when none is)."""
    lib = get_library()
    if lib.nimf_trace_stop() != 0:
        _raise(lib)
