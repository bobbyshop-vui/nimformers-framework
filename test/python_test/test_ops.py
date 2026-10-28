"""Kernel / math operation tests, cross-checked against NumPy."""
import math

import numpy as np
import pytest


def _softmax(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _layernorm(x, gamma, beta, eps=1e-5, axis=-1):
    mean = x.mean(axis=axis, keepdims=True)
    var = x.var(axis=axis, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def _causal_attention(q, k, v, scale):
    scores = np.matmul(q, np.swapaxes(k, -1, -2)) * scale
    s = scores.shape[-1]
    mask = np.triu(np.ones((s, s), dtype=bool), 1)
    scores = np.where(mask, -np.inf, scores)
    w = _softmax(scores.astype(np.float32))
    out = np.matmul(w, v)
    w = np.where(mask, 0.0, w).astype(np.float32)
    return out.astype(np.float32), w


# ---------------------------------------------------------------- elementwise

def test_binary_ops(ctx, rng):
    a = rng.standard_normal((3, 4)).astype(np.float32)
    b = rng.standard_normal((3, 4)).astype(np.float32) + 2.0
    np.testing.assert_allclose(ctx.add(a, b), a + b, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ctx.sub(a, b), a - b, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ctx.mul(a, b), a * b, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ctx.div(a, b), a / b, rtol=1e-6, atol=1e-6)


def test_binary_broadcast(ctx, rng):
    a = rng.standard_normal((3, 4)).astype(np.float32)
    b = rng.standard_normal((4,)).astype(np.float32)
    y = ctx.add(a, b)
    assert y.shape == (3, 4)
    np.testing.assert_allclose(y, a + b, rtol=1e-6, atol=1e-6)


def test_unary_ops(ctx, rng):
    x = np.abs(rng.standard_normal((3, 4)).astype(np.float32)) + 0.1
    np.testing.assert_allclose(ctx.relu(x - 1.0), np.maximum(x - 1.0, 0), atol=1e-6)
    np.testing.assert_allclose(ctx.sqrt(x), np.sqrt(x), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ctx.exp(x), np.exp(x), rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(ctx.log(x), np.log(x), rtol=1e-6, atol=1e-6)
    sig = 1.0 / (1.0 + np.exp(-x))
    np.testing.assert_allclose(ctx.sigmoid(x), sig, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ctx.tanh(x), np.tanh(x), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(ctx.neg(x), -x, atol=1e-6)
    np.testing.assert_allclose(ctx.pow(x, 2.5), np.power(x, 2.5), rtol=1e-5, atol=1e-5)


def test_scalar_ops(ctx, rng):
    x = rng.standard_normal((2, 5)).astype(np.float32)
    np.testing.assert_allclose(ctx.add_scalar(x, 1.25), x + 1.25, atol=1e-6)
    np.testing.assert_allclose(ctx.mul_scalar(x, -2.0), x * -2.0, atol=1e-6)


def test_apflu(ctx, rng):
    x = rng.standard_normal((2, 5)).astype(np.float32) * 2
    y = ctx.apflu(x, 0.1, 0.1)
    dy = ctx.apflu_backward(x, np.ones_like(x), 0.1, 0.1)
    assert y.shape == x.shape and dy.shape == x.shape
    assert np.all(np.isfinite(y)) and np.all(np.isfinite(dy))


# ---------------------------------------------------------------- matmul

def test_matmul(ctx, rng):
    a = rng.standard_normal((3, 5)).astype(np.float32)
    b = rng.standard_normal((5, 7)).astype(np.float32)
    np.testing.assert_allclose(ctx.matmul(a, b), a @ b, rtol=1e-4, atol=1e-4)


def test_matmul_batched(ctx, rng):
    a = rng.standard_normal((2, 3, 5)).astype(np.float32)
    b = rng.standard_normal((2, 5, 7)).astype(np.float32)
    y = ctx.matmul(a, b)
    assert y.shape == (2, 3, 7)
    np.testing.assert_allclose(y, np.matmul(a, b), rtol=1e-4, atol=1e-4)


def test_matmul_nt(ctx, rng):
    a = rng.standard_normal((3, 5)).astype(np.float32)
    b = rng.standard_normal((7, 5)).astype(np.float32)  # [N, K]
    np.testing.assert_allclose(ctx.matmul_nt(a, b), a @ b.T, rtol=1e-4, atol=1e-4)


def test_matmul2(ctx, rng):
    a1 = rng.standard_normal((2, 3)).astype(np.float32)
    b1 = rng.standard_normal((3, 4)).astype(np.float32)
    a2 = rng.standard_normal((5, 2)).astype(np.float32)
    b2 = rng.standard_normal((2, 6)).astype(np.float32)
    y1, y2 = ctx.matmul2(a1, b1, a2, b2)
    np.testing.assert_allclose(y1, a1 @ b1, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(y2, a2 @ b2, rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------- norms

def test_softmax(ctx, rng):
    x = rng.standard_normal((4, 9)).astype(np.float32)
    np.testing.assert_allclose(ctx.softmax(x), _softmax(x), rtol=1e-5, atol=1e-6)
    x3 = rng.standard_normal((2, 3, 5)).astype(np.float32)
    np.testing.assert_allclose(ctx.softmax(x3, axis=1), _softmax(x3, axis=1),
                               rtol=1e-5, atol=1e-6)


def test_layernorm(ctx, rng):
    x = rng.standard_normal((4, 8)).astype(np.float32)
    g = rng.standard_normal(8).astype(np.float32)
    b = rng.standard_normal(8).astype(np.float32)
    np.testing.assert_allclose(ctx.layernorm(x, g, b), _layernorm(x, g, b),
                               rtol=1e-5, atol=1e-5)


def test_add_layernorm(ctx, rng):
    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((4, 8)).astype(np.float32)
    g = rng.standard_normal(8).astype(np.float32)
    beta = rng.standard_normal(8).astype(np.float32)
    summed, normed = ctx.add_layernorm(a, b, g, beta)
    np.testing.assert_allclose(summed, a + b, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(normed, _layernorm(a + b, g, beta),
                               rtol=1e-5, atol=1e-5)


def test_layernorm_backward(ctx, rng):
    x = rng.standard_normal((2, 4)).astype(np.float32)
    dy = rng.standard_normal((2, 4)).astype(np.float32)
    g = rng.standard_normal(4).astype(np.float32)
    b = rng.standard_normal(4).astype(np.float32)
    dx, dg, db = ctx.layernorm_backward(dy, x, g, b)

    eps = 1e-3

    def loss(xx, gg, bb):
        return float((ctx.layernorm(xx, gg, bb) * dy).sum())

    num_dx = np.zeros_like(x)
    for idx in np.ndindex(x.shape):
        xp, xm = x.copy(), x.copy()
        xp[idx] += eps
        xm[idx] -= eps
        num_dx[idx] = (loss(xp, g, b) - loss(xm, g, b)) / (2 * eps)
    np.testing.assert_allclose(dx, num_dx, rtol=3e-2, atol=3e-2)

    num_dg = np.zeros_like(g)
    num_db = np.zeros_like(b)
    for i in range(4):
        gp, gm = g.copy(), g.copy()
        gp[i] += eps
        gm[i] -= eps
        num_dg[i] = (loss(x, gp, b) - loss(x, gm, b)) / (2 * eps)
        bp, bm = b.copy(), b.copy()
        bp[i] += eps
        bm[i] -= eps
        num_db[i] = (loss(x, g, bp) - loss(x, g, bm)) / (2 * eps)
    np.testing.assert_allclose(dg, num_dg, rtol=3e-2, atol=3e-2)
    np.testing.assert_allclose(db, num_db, rtol=3e-2, atol=3e-2)


# ---------------------------------------------------------------- attention

def test_attention(ctx, rng):
    q = rng.standard_normal((1, 2, 4, 4)).astype(np.float32)
    k = rng.standard_normal((1, 2, 4, 4)).astype(np.float32)
    v = rng.standard_normal((1, 2, 4, 4)).astype(np.float32)
    scale = 1.0 / math.sqrt(4)
    out, scores = ctx.attention(q, k, v, scale=scale)
    ref_out, ref_scores = _causal_attention(q, k, v, scale)
    assert out.shape == q.shape
    assert scores.shape == (1, 2, 4, 4)
    np.testing.assert_allclose(out, ref_out, rtol=2e-4, atol=2e-4)
    np.testing.assert_allclose(scores, ref_scores, rtol=2e-4, atol=2e-4)


def test_attention_backward(ctx, rng):
    q = rng.standard_normal((1, 1, 3, 4)).astype(np.float32)
    out, scores = ctx.attention(q, q, q)
    dq, dk, dv = ctx.attention_backward(q, q, q, scores, out)
    assert dq.shape == q.shape and dk.shape == q.shape and dv.shape == q.shape
    assert np.all(np.isfinite(dq))


# ---------------------------------------------------------------- data movement

def test_embedding_lookup(ctx, rng):
    table = rng.standard_normal((8, 4)).astype(np.float32)
    ids = np.array([[1, 2], [3, 0]], dtype=np.int32)
    np.testing.assert_allclose(ctx.embedding_lookup(table, ids), table[ids])
    with pytest.raises(IndexError):
        ctx.embedding_lookup(table, np.array([9], np.int32))


def test_gather(ctx, rng):
    src = rng.standard_normal(16).astype(np.float32)
    idx = np.array([1, 4, 7], dtype=np.int32)
    np.testing.assert_allclose(ctx.gather(src, idx), src[idx])


def test_compare(ctx, rng):
    a = rng.standard_normal((4, 5)).astype(np.float32)
    b = rng.standard_normal((4, 5)).astype(np.float32)
    pairs = {"gt": a > b, "lt": a < b, "eq": a == b, "ge": a >= b, "le": a <= b}
    for op, ref in pairs.items():
        np.testing.assert_array_equal(ctx.compare(a, b, op), ref)


def test_dropout(ctx, rng):
    x = rng.standard_normal((8, 16)).astype(np.float32)
    p = 0.25
    out = ctx.dropout(x, p, training=True, seed=7)
    active = out != 0
    assert active.any()
    np.testing.assert_allclose(out[active], x[active] / (1 - p), rtol=1e-5, atol=1e-6)
    np.testing.assert_array_equal(ctx.dropout(x, p, training=False), x)
    out2 = ctx.dropout(x, p, training=True, seed=7)
    np.testing.assert_array_equal(out, out2)
    mask = ctx.dropout_mask(64, p, seed=7)
    assert set(np.unique(mask).tolist()) <= {0.0, float(np.float32(1 / (1 - p)))}
    np.testing.assert_allclose(
        ctx.dropout(np.ones(64, np.float32), p, training=True, seed=7),
        mask, rtol=1e-6)


def test_argmax(ctx, rng):
    x = rng.standard_normal((3, 7)).astype(np.float32)
    np.testing.assert_array_equal(ctx.argmax(x, axis=1), x.argmax(axis=1))
    np.testing.assert_array_equal(ctx.argmax(x, axis=0), x.argmax(axis=0))
    # Power-of-two reduction width exercises the GPU tree-reduction path.
    p = rng.standard_normal((2, 4, 3)).astype(np.float32)
    np.testing.assert_array_equal(ctx.argmax(p, axis=1), p.argmax(axis=1))
    np.testing.assert_array_equal(ctx.argmax(p, axis=0), p.argmax(axis=0))
    vals, idx = ctx.argmax_with_values(x, axis=1)
    np.testing.assert_array_equal(idx, x.argmax(axis=1))
    np.testing.assert_allclose(vals, x.max(axis=1), rtol=1e-6)
    with pytest.raises(ValueError):
        ctx.argmax(np.zeros((3, 0), np.float32), axis=1)


def test_masked_fill_where(ctx, rng):
    x = rng.standard_normal((4, 6)).astype(np.float32)
    y = rng.standard_normal((4, 6)).astype(np.float32)
    mask = x > 0
    np.testing.assert_array_equal(ctx.masked_fill(x, mask, -1.0),
                                  np.where(mask, -1.0, x))
    np.testing.assert_array_equal(ctx.where(mask, x, y), np.where(mask, x, y))


# ---------------------------------------------------------------- reductions

def test_reductions(ctx, rng):
    x = rng.standard_normal((3, 4, 5)).astype(np.float32)
    for axis in (0, 1, 2, -1):
        np.testing.assert_allclose(ctx.sum(x, axis=axis), x.sum(axis=axis),
                                   rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ctx.mean(x, axis=axis), x.mean(axis=axis),
                                   rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ctx.max(x, axis=axis), x.max(axis=axis),
                                   rtol=1e-6, atol=1e-6)


def test_scans(ctx, rng):
    x = rng.standard_normal((2, 3, 4)).astype(np.float32)
    for axis in (0, 1, 2):
        np.testing.assert_allclose(ctx.cumsum(x, axis=axis), x.cumsum(axis=axis),
                                   rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ctx.cumprod(x, axis=axis), x.cumprod(axis=axis),
                                   rtol=1e-4, atol=1e-5)
    # Non-contiguous input must not corrupt the output layout.
    noncontig = rng.standard_normal((3, 4)).astype(np.float32).T
    np.testing.assert_allclose(ctx.cumsum(noncontig, axis=-1),
                               np.cumsum(noncontig, axis=-1), rtol=1e-5, atol=1e-5)


def test_affine_scan(ctx, rng):
    a = rng.uniform(0.2, 0.9, (2, 5)).astype(np.float32)
    b = rng.standard_normal((2, 5)).astype(np.float32)
    out = ctx.affine_scan(a, b, axis=1)
    ref = np.zeros_like(b)
    prev = np.zeros(2, dtype=np.float32)
    for t in range(5):
        prev = a[:, t] * prev + b[:, t]
        ref[:, t] = prev
    np.testing.assert_allclose(out, ref, rtol=1e-5, atol=1e-6)
    # Non-last axis
    a0 = rng.uniform(0.2, 0.9, (5, 2)).astype(np.float32)
    b0 = rng.standard_normal((5, 2)).astype(np.float32)
    out0 = ctx.affine_scan(a0, b0, axis=0)
    ref0 = np.zeros_like(b0)
    prev0 = np.zeros(2, dtype=np.float32)
    for t in range(5):
        prev0 = a0[t] * prev0 + b0[t]
        ref0[t] = prev0
    np.testing.assert_allclose(out0, ref0, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------- shape ops

def test_permute(ctx, rng):
    x = rng.standard_normal((2, 3, 4)).astype(np.float32)
    for dims in ([2, 0, 1], [1, 2, 0], [0, 2, 1]):
        np.testing.assert_array_equal(ctx.permute(x, dims), x.transpose(dims))


def test_split_and_slice(ctx, rng):
    x = rng.standard_normal((2, 7, 3)).astype(np.float32)
    pieces = ctx.split(x, 3, axis=1)
    assert [p.shape for p in pieces] == [(2, 3, 3), (2, 3, 3), (2, 1, 3)]
    np.testing.assert_array_equal(np.concatenate(pieces, axis=1), x)
    np.testing.assert_array_equal(ctx.split_slice(x, 1, 2, 4), x[:, 2:6, :])


def test_concat(ctx, rng):
    a = rng.standard_normal((2, 3)).astype(np.float32)
    b = rng.standard_normal((4, 3)).astype(np.float32)
    np.testing.assert_array_equal(ctx.concat([a, b], axis=0),
                                  np.concatenate([a, b], axis=0))
    c = rng.standard_normal((2, 5)).astype(np.float32)
    np.testing.assert_array_equal(ctx.concat([a, c], axis=1),
                                  np.concatenate([a, c], axis=1))
    x3 = rng.standard_normal((2, 3, 4)).astype(np.float32)
    y3 = rng.standard_normal((2, 2, 4)).astype(np.float32)
    np.testing.assert_array_equal(ctx.concat([x3, y3], axis=1),
                                  np.concatenate([x3, y3], axis=1))
    np.testing.assert_array_equal(ctx.stack([x3, x3 + 1.0], axis=0),
                                  np.stack([x3, x3 + 1.0], axis=0))
    np.testing.assert_array_equal(ctx.stack([a, a + 1.0], axis=1),
                                  np.stack([a, a + 1.0], axis=1))
