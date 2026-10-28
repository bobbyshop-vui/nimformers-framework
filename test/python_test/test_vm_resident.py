"""Micro-VM bytecode and resident-weight tests."""
import numpy as np
import pytest

from nimformers import NimformersError, Op, VMProgram


def _double_program(ctx):
    prog = ctx.vm_program()
    prog.load_f32(1, 0, 0)       # R1 = mem[R0]
    prog.add(1, 1, 1)            # R1 = R1 + R1
    prog.store_f32(1, 0, 0)      # mem[R0] = R1
    prog.halt()
    return prog


def test_vm_run_double(ctx):
    prog = _double_program(ctx)
    mem = np.arange(1, 65, dtype=np.float32)
    out = ctx.vm_run(prog, mem, tg_size=64)
    np.testing.assert_array_equal(out, 2 * np.arange(1, 65, dtype=np.float32))
    assert len(prog) == 6
    prog.close()


def test_vm_run_preserves_shape(ctx):
    prog = _double_program(ctx)
    mem = np.arange(1, 17, dtype=np.float32).reshape(4, 4)
    expected = mem * 2
    out = prog.run(ctx, mem, tg_size=16)
    assert out.shape == (4, 4)
    np.testing.assert_array_equal(out, expected)
    prog.close()


def test_vm_words_roundtrip(ctx):
    prog = _double_program(ctx)
    words = prog.words
    assert words.dtype == np.uint32
    assert words.size == 6
    other = ctx.vm_program()
    other.set_words(words)
    np.testing.assert_array_equal(other.words, words)
    # Raw emit produces the same bytes.
    raw = ctx.vm_program()
    for w in words:
        raw.emit_word(int(w))
    np.testing.assert_array_equal(raw.words, words)
    raw.close()
    other.close()
    prog.close()


def test_vm_save_load(ctx, tmp_path):
    prog = _double_program(ctx)
    path = tmp_path / "double.bvm"
    prog.save(path)
    assert path.stat().st_size == 6 * 4
    loaded = VMProgram.load(ctx.lib, path)
    np.testing.assert_array_equal(loaded.words, prog.words)
    mem = np.ones(32, np.float32)
    np.testing.assert_array_equal(ctx.vm_run(loaded, mem, tg_size=32),
                                  np.full(32, 2.0, np.float32))
    loaded.close()
    prog.close()


def test_vm_lowering_and_key(ctx):
    prog = _double_program(ctx)
    msl = prog.lower_msl("double_kernel")
    glsl = prog.lower_glsl("double_kernel")
    assert "double_kernel" in msl
    assert "void main" in glsl or "main()" in glsl
    key = prog.key("test")
    assert isinstance(key, str) and len(key) > 0
    assert key == prog.key("test")
    prog.close()


def test_vm_merge(ctx):
    a = _double_program(ctx)
    b = _double_program(ctx)
    merged = VMProgram.merge(ctx.lib, a, b)
    # mergePrograms drops the HALT at the end of every part and appends one,
    # so 5 + 5 + 1 = 11 words for two 6-word programs.
    assert len(merged) == len(a) + len(b) - 1
    np.testing.assert_array_equal(merged.words[:len(a) - 1], a.words[:-1])
    np.testing.assert_array_equal(
        merged.words[len(a) - 1:len(a) + len(b) - 2], b.words[:-1])
    assert int(merged.words[-1]) == int(Op.HALT)
    merged.close()
    a.close()
    b.close()


def test_vm_invalid_opcode(ctx):
    prog = ctx.vm_program()
    with pytest.raises(NimformersError):
        prog.emit(0xFE)
    prog.close()


def test_vm_raw_dispatch(ctx):
    prog = _double_program(ctx)
    words = prog.words
    prog.close()
    mem = np.arange(1, 9, dtype=np.float32)
    out = ctx.micro_vm_dispatch(words, mem, tg_size=8)
    np.testing.assert_array_equal(out, 2 * np.arange(1, 9, dtype=np.float32))


def test_resident_f32(ctx, rng):
    w = rng.standard_normal((6, 5)).astype(np.float32)  # [K, N]
    x = rng.standard_normal((3, 6)).astype(np.float32)
    with ctx.resident_f32(w) as resident:
        np.testing.assert_allclose(resident.linear(x), x @ w,
                                   rtol=1e-4, atol=1e-4)


def test_resident_q4(ctx, rng):
    w = rng.standard_normal((5, 6)).astype(np.float32)  # [N, K]
    x = rng.standard_normal((3, 6)).astype(np.float32)
    q = ctx.quantize(w, "int4_asym")
    with ctx.resident_q4(q) as resident:
        np.testing.assert_allclose(resident.linear(x), x @ q.dequantize().T,
                                   rtol=2e-5, atol=2e-4)
    q.close()


def test_resident_q4_rejects_symmetric(ctx, rng):
    q = ctx.quantize(rng.standard_normal((4, 5)).astype(np.float32), "int4")
    with pytest.raises(ValueError):
        ctx.resident_q4(q)
    q.close()
