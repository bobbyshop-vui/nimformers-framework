"""Quantization, custom-float, APF, and NIMQ state-dict tests."""
from pathlib import Path

import numpy as np
import pytest

from nimformers import NimformersError, QuantTensor, StateDict


def test_quantize_roundtrip(ctx, rng):
    x = rng.standard_normal((32, 64)).astype(np.float32)
    tolerances = {
        "fp32": 0.0,
        "int8": 0.02,
        "int4": 0.4,
        "int4_asym": 0.4,
        "fp8_e4m3": 0.5,
        "fp8_e5m2": 0.9,
        "fp4_km": 0.7,
    }
    for kind, tol in tolerances.items():
        q = ctx.quantize(x, kind)
        assert isinstance(q, QuantTensor)
        assert q.kind == kind
        assert q.shape == x.shape
        assert q.numel == x.size
        assert q.nbytes > 0
        y = q.dequantize()
        assert y.shape == x.shape
        assert float(np.abs(y - x).max()) <= tol, (kind, float(np.abs(y - x).max()))
        if kind == "fp4_km":
            assert q.group_size == 32
            assert q.cf_bits == (4, 3)
        q.close()


def test_quantize_custom(ctx, rng):
    x = rng.standard_normal((8, 8)).astype(np.float32)
    q = ctx.quantize(x, "custom", exponent_bits=5, mantissa_bits=10)
    assert q.kind == "custom"
    assert q.cf_bits == (5, 10)
    assert float(np.abs(q.dequantize() - x).max()) < 0.1
    q.close()


def test_fp4_requires_2d_and_group_aligned(ctx, rng):
    with pytest.raises(NimformersError):
        ctx.quantize(rng.standard_normal((64,)).astype(np.float32), "fp4_km")
    with pytest.raises(NimformersError):
        ctx.quantize(rng.standard_normal((8, 31)).astype(np.float32), "fp4_km")


def test_quant_payload_views(ctx, rng):
    x = rng.standard_normal((4, 8)).astype(np.float32)
    q = ctx.quantize(x, "int4_asym")
    raw = q.data()
    scales = q.scales()
    zeros = q.zeros()
    assert raw.dtype == np.uint8 and raw.size == q.nbytes
    assert scales.dtype == np.float32 and scales.size > 0
    assert zeros.size == scales.size
    q.close()


def test_customfloat_roundtrip(ctx, rng):
    x = rng.standard_normal((8, 6)).astype(np.float32)
    encoded = ctx.customfloat_encode(x, 4, 3)
    assert encoded.dtype == np.uint8 and encoded.size > 0
    decoded = ctx.customfloat_decode(encoded, x.size, 4, 3).reshape(x.shape)
    assert float(np.abs(decoded - x).max()) < 0.3


def test_apf_build_and_roundtrip(ctx, rng):
    x = rng.standard_normal((16, 8)).astype(np.float32)
    e, m = ctx.apf_build_dtype(x)
    assert 1 <= e <= 8
    assert m >= 2
    encoded, (e2, m2) = ctx.apf_cast_training(x)
    decoded = ctx.apf_decode_training(encoded, e2, m2, x.size).reshape(x.shape)
    # APF is an adaptive-precision codec: relative error is ~2^-mantissa for
    # normal values, with a small absolute floor near underflow.
    assert float(np.abs(decoded - x).max()) < 0.02


def test_q4_linear_matches_dequant(ctx, rng):
    # [N, K] int4_asym weight + activation [M, K]
    w = rng.standard_normal((16, 32)).astype(np.float32)
    x = rng.standard_normal((5, 32)).astype(np.float32)
    q = ctx.quantize(w, "int4_asym")
    y = ctx.linear_q4(x, q)
    np.testing.assert_allclose(y, x @ q.dequantize().T, rtol=2e-5, atol=2e-4)
    np.testing.assert_array_equal(q.dequantize_transposed(), q.dequantize().T)
    q.close()


def test_state_dict_roundtrip(ctx, rng, tmp_path):
    a = rng.standard_normal((8, 12)).astype(np.float32)
    b = rng.standard_normal((3,)).astype(np.float32)
    qa = ctx.quantize(a, "int8")
    qb = ctx.quantize(b, "fp32")
    sd = ctx.create_state_dict((8, 12, 3, 2, 1))
    sd["a.weight"] = qa
    sd.append("b.bias", qb)
    assert len(sd) == 2
    assert sd.arch == (8, 12, 3, 2, 1)
    assert sd.names == ["a.weight", "b.bias"]
    assert "a.weight" in sd and "missing" not in sd

    path = tmp_path / "tiny.nimq"
    sd.save(path)
    assert path.stat().st_size > 0

    for mapped in (False, True):
        loaded = ctx.load_state_dict(path, mapped=mapped)
        assert isinstance(loaded, StateDict)
        assert len(loaded) == 2
        assert loaded.arch == (8, 12, 3, 2, 1)
        np.testing.assert_allclose(loaded["a.weight"].dequantize(), a, atol=0.05)
        np.testing.assert_allclose(loaded["b.bias"].dequantize(), b, atol=0.0)
        d = loaded.to_dict()
        assert set(d) == {"a.weight", "b.bias"}
        loaded.close()
    sd.close()
    qa.close()
    qb.close()


def test_state_dict_errors(ctx, tmp_path):
    with pytest.raises(NimformersError):
        ctx.load_state_dict(tmp_path / "missing.nimq")
    sd = ctx.create_state_dict((1, 2, 3, 4, 5))
    with pytest.raises(KeyError):
        _ = sd["missing"]
    with pytest.raises(TypeError):
        sd["x"] = np.zeros(4, np.float32)
    sd.close()


def test_real_checkpoint(ctx, repo_root):
    path = repo_root / "test/spectron/models/finetune.nimq"
    if not path.is_file():
        pytest.skip("spectron checkpoint not downloaded")
    sd = ctx.load_state_dict(path, mapped=True)
    assert len(sd) == 40
    assert sd.arch == (756, 64, 4, 2, 4)
    assert "embed.weight" in sd
    first = sd["embed.weight"]
    assert first.shape == (756, 64)
    assert first.nbytes == 756 * 64 * 4
    sd.close()
