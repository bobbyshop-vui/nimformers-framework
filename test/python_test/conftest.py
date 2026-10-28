"""Shared pytest fixtures for the Nimformers Python bindings."""
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nimformers import Context, NimformersError  # noqa: E402


@pytest.fixture(scope="session")
def ctx():
    try:
        context = Context()
    except NimformersError as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"libnimformers is not built/usable: {exc}")
    yield context
    context.close()


@pytest.fixture()
def rng():
    return np.random.default_rng(1234)


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT
