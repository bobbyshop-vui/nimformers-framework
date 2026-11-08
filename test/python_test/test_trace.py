"""nf.trace: record session steps into a FIR trace file."""
import numpy as np
import pytest

import nimformers as nf
from nimformers import Buffer, Op, Session


@pytest.fixture
def metal_ctx(ctx):
    if ctx.backend != "metal":
        pytest.skip("stateful session dispatch needs the Metal backend")
    return ctx


def _accumulate(ctx, sess):
    p = ctx.vm_program()
    p.emit(Op.U_LID, 0)
    p.emit(Op.U_LOAD_IN, 1, 0, value=0)
    p.emit(Op.U_LOAD_F32, 2, 0, value=0)
    p.add(1, 1, 2)
    p.emit(Op.U_STORE_F32, 1, 0, value=0)
    p.halt()
    return sess.compile(p, "trace_test_accumulate")


def test_trace_records_steps(metal_ctx, tmp_path):
    x = Buffer(metal_ctx, np.ones(8, dtype=np.float32))
    acc = Buffer(metal_ctx, size=8)
    sess = Session(metal_ctx)
    kernel = _accumulate(metal_ctx, sess)
    path = tmp_path / "run.fprobe"
    assert not nf.trace_active()
    with nf.trace(path, steps="decode:2") as t:
        assert t.active and nf.trace_active()
        assert nf.trace_file() == str(path)
        for i in range(3):
            with sess.step(token=i, position=i), sess:
                sess.dispatch(kernel, threads=8, tg_size=8, mem=acc, input=x)
    assert not nf.trace_active() and not t.active
    text = path.read_text()
    assert text.startswith("fprobe")
    assert text.count("\nstep decode") == 2  # three steps ran, steps="decode:2" keeps two
    assert text.rstrip().endswith("end")
    np.testing.assert_array_equal(acc.read(0, 8), np.full(8, 3.0, dtype=np.float32))


def test_trace_data_flag_controls_sidecar(metal_ctx, tmp_path):
    x = Buffer(metal_ctx, np.ones(8, dtype=np.float32))
    acc = Buffer(metal_ctx, size=8)
    sess = Session(metal_ctx)
    kernel = _accumulate(metal_ctx, sess)
    for data in (True, False):
        path = tmp_path / f"data{int(data)}.fprobe"
        with nf.trace(path, steps="decode:2", data=data) as t:
            for i in range(2):
                with sess.step(token=i, position=i), sess:
                    sess.set_step_inputs(acc, 0, 1, i, i)  # a host upload inside the step
                    sess.dispatch(kernel, threads=8, tg_size=8, mem=acc, input=x)
        assert (path.parent / (path.name + ".data")).exists() == data
        assert t.data_path == (str(path) + ".data" if data else None)


def test_trace_manual_stop_and_restart(metal_ctx, tmp_path):
    t = nf.trace(tmp_path / "a.fprobe")
    assert t.active
    t.stop()
    t.stop()  # harmless twice
    assert not nf.trace_active()
    nf.trace(tmp_path / "b.fprobe")
    nf.trace_stop()
    assert not nf.trace_active()


def test_trace_rejects_missing_directory(tmp_path):
    with pytest.raises(nf.NimformersError):
        nf.trace(tmp_path / "missing" / "run.fprobe")
    assert not nf.trace_active()


def test_trace_rejects_bad_steps(tmp_path):
    with pytest.raises(nf.NimformersError):
        nf.trace(tmp_path / "run.fprobe", steps="decode")
    assert not nf.trace_active()


def test_kernel_compiled_before_trace_keeps_its_bytecode(metal_ctx, tmp_path):
    x = Buffer(metal_ctx, np.ones(8, dtype=np.float32))
    acc = Buffer(metal_ctx, size=8)
    sess = Session(metal_ctx)
    kernel = _accumulate(metal_ctx, sess)  # compiled before the recording starts
    path = tmp_path / "late.fprobe"
    with nf.trace(path, steps="decode:2"):
        for i in range(2):
            with sess.step(token=i, position=i), sess:
                sess.dispatch(kernel, threads=8, tg_size=8, mem=acc, input=x)
    text = path.read_text()
    assert "missing_words" not in text
    assert "disp d0 prog=p0" in text
