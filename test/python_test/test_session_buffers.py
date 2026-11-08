"""Stateful Buffer / Session API (resident buffers, JIT kernels, step inputs)."""
import numpy as np
import pytest

from nimformers import Buffer, Op, Session


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
