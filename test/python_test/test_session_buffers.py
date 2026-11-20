"""Stateful Buffer / Session API (resident buffers, JIT kernels, step inputs)."""
from contextlib import ExitStack

import numpy as np
import pytest

from nimformers import Buffer, Context, NimformersError, Op, Session


@pytest.fixture
def metal_ctx(ctx):
    if ctx.backend != "metal":
        pytest.skip("stateful session dispatch needs the Metal backend")
    return ctx


def _double_program(ctx):
    p = ctx.vm_program()
    p.emit(Op.U_LID, 0)
    p.emit(Op.U_LOAD_IN, 1, 0, value=0)
    p.mul_const(1, 1, 2.0)
    p.emit(Op.U_STORE_F32, 1, 0, value=0)
    p.halt()
    return p


def test_buffer_roundtrip(metal_ctx):
    data = np.arange(16, dtype=np.float32)
    buf = Buffer(metal_ctx, data)
    np.testing.assert_array_equal(buf.read(0, 16), data)
    buf.close()


def test_session_dispatch_doubles(metal_ctx):
    x = Buffer(metal_ctx, np.arange(64, dtype=np.float32))
    y = Buffer(metal_ctx, size=64)
    sess = Session(metal_ctx)
    kernel = sess.compile(_double_program(metal_ctx), "test_double")
    with sess:
        sess.dispatch(kernel, threads=64, tg_size=64, mem=y, input=x)
    np.testing.assert_array_equal(y.read(0, 64), 2 * np.arange(64, dtype=np.float32))


def test_session_state_persists_across_sessions(metal_ctx):
    x = Buffer(metal_ctx, np.ones(8, dtype=np.float32))
    acc = Buffer(metal_ctx, size=8)
    p = metal_ctx.vm_program()
    p.emit(Op.U_LID, 0)
    p.emit(Op.U_LOAD_IN, 1, 0, value=0)
    p.emit(Op.U_LOAD_F32, 2, 0, value=0)
    p.add(1, 1, 2)
    p.emit(Op.U_STORE_F32, 1, 0, value=0)
    p.halt()
    sess = Session(metal_ctx)
    kernel = sess.compile(p, "test_accumulate")
    for _ in range(3):
        with sess.step(token=1, position=0), sess:
            sess.dispatch(kernel, threads=8, tg_size=8, mem=acc, input=x)
    np.testing.assert_array_equal(acc.read(0, 8), np.full(8, 3.0, dtype=np.float32))


def test_step_inputs_land_at_offsets(metal_ctx):
    arena = Buffer(metal_ctx, size=128)
    sess = Session(metal_ctx)
    with sess:
        sess.set_step_inputs(arena, 0, 64, token=9707, position=3)
    out = arena.read(0, 65)
    assert out[0] == 9707.0 and out[64] == 3.0


def test_native_resident_operations():
    a = np.array([[1, 2, -1], [2, -1, 0]], dtype=np.float32)
    b = np.array([[1, 2], [0, 1], [2, 0]], dtype=np.float32)
    gain = np.array([1, 2], dtype=np.float32)
    bias = np.array([3, -1], dtype=np.float32)
    with Context("cpu") as ctx, ExitStack() as buffers:
        x = buffers.enter_context(ctx.buffer(a))
        w = buffers.enter_context(ctx.buffer(b))
        wt = buffers.enter_context(ctx.buffer(b.T))
        gamma = buffers.enter_context(ctx.buffer(gain))
        beta = buffers.enter_context(ctx.buffer(bias))
        product, product_nt, rms, layer, probs, summed, squared = [
            buffers.enter_context(ctx.buffer(size=4)) for _ in range(7)
        ]
        dst = buffers.enter_context(ctx.buffer(size=6))
        with ctx.session() as sess:
            sess.matmul(x, w, product, m=2, k=3, n=2)
            sess.matmul(x, wt, product_nt, m=2, k=3, n=2, transpose_b=True)
            sess.rmsnorm(product, gamma, rms, rows=2, cols=2)
            sess.layernorm(product, gamma, beta, layer, rows=2, cols=2)
            sess.softmax(product, probs, rows=2, cols=2)
            sess.vec_op("add", product, product_nt, summed)
            sess.activation("pow", summed, squared, p=2.0)
            sess.copy_into(squared, dst, 2, src_offset=1, dst_offset=2)

        expected = a @ b
        np.testing.assert_allclose(product.read().reshape(2, 2), expected)
        np.testing.assert_allclose(product_nt.read().reshape(2, 2), expected)
        norm = expected / np.sqrt((expected**2).mean(axis=1, keepdims=True) + 1e-5)
        np.testing.assert_allclose(rms.read().reshape(2, 2), norm * gain, rtol=1e-5)
        centered = expected - expected.mean(axis=1, keepdims=True)
        norm = centered / np.sqrt((centered**2).mean(axis=1, keepdims=True) + 1e-5)
        np.testing.assert_allclose(layer.read().reshape(2, 2), norm * gain + bias, rtol=1e-5)
        exps = np.exp(expected)
        np.testing.assert_allclose(probs.read().reshape(2, 2),
                                   exps / exps.sum(axis=1, keepdims=True), rtol=1e-6)
        np.testing.assert_allclose(summed.read(), (expected * 2).ravel())
        np.testing.assert_allclose(squared.read(), (expected * 2).ravel() ** 2)
        np.testing.assert_array_equal(dst.read(), [0, 0, 64, 16, 0, 0])


def test_native_resident_operations_validate_buffers():
    with Context("cpu") as ctx, Context("cpu") as other, ExitStack() as buffers:
        src = buffers.enter_context(ctx.buffer([1, 2, 3, 4]))
        out = buffers.enter_context(ctx.buffer(size=4))
        wrong_size = buffers.enter_context(ctx.buffer(size=5))
        foreign = buffers.enter_context(other.buffer(size=4))
        sess = ctx.session()
        with pytest.raises(NimformersError, match="open session"):
            sess.softmax(src, out, rows=2, cols=2)
        with sess:
            with pytest.raises(NimformersError, match="out of bounds"):
                sess.copy_into(src, out, 4, src_offset=1)
            with pytest.raises(NimformersError, match="another context"):
                sess.copy_into(src, foreign, 4)
            with pytest.raises(NimformersError, match="overlap"):
                sess.copy_into(src, src, 3, dst_offset=1)
            with pytest.raises(NimformersError, match="wrong size"):
                sess.softmax(src, wrong_size, rows=2, cols=2)
            with pytest.raises(NimformersError, match="distinct"):
                sess.softmax(src, src, rows=2, cols=2)
            with pytest.raises(NimformersError, match="unsupported"):
                sess.vec_op("unknown", src, src, out)
            with pytest.raises(NimformersError, match="invalid session dimensions"):
                sess.matmul(src, src, out, m=1 << 32, k=1 << 32, n=4)


def test_resident_q4_feeds_attention(rng):
    x = rng.standard_normal((2, 6)).astype(np.float32)
    w = rng.standard_normal((4, 6)).astype(np.float32)
    with Context("cpu") as ctx, ExitStack() as resources:
        quant = resources.enter_context(ctx.quantize(w, "int4_asym"))
        weight = resources.enter_context(ctx.resident_q4(quant))
        src = resources.enter_context(ctx.buffer(x))
        projected = resources.enter_context(ctx.buffer(size=8))
        out = resources.enter_context(ctx.buffer(size=8))
        scores = resources.enter_context(ctx.buffer(size=4))
        with ctx.session() as sess:
            sess.matmul_q4(src, weight, projected, rows=2)
            sess.attention(projected, projected, projected, out, scores,
                           batch=1, heads=1, steps=2, dim=4)

        expected_projection = x @ quant.dequantize().T
        np.testing.assert_allclose(projected.read().reshape(2, 4), expected_projection,
                                   rtol=2e-5, atol=2e-4)
        logits = expected_projection @ expected_projection.T / 2
        logits[0, 1] = -np.inf
        exps = np.exp(logits - logits.max(axis=1, keepdims=True))
        expected_scores = exps / exps.sum(axis=1, keepdims=True)
        np.testing.assert_allclose(scores.read().reshape(2, 2), expected_scores, atol=2e-4)
        np.testing.assert_allclose(out.read().reshape(2, 4), expected_scores @ expected_projection,
                                   rtol=2e-5, atol=2e-4)
