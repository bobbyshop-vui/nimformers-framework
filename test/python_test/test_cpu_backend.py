"""CPU backend tests: no GPU required (SIMD kernels + CPU micro-VM)."""
import json

import numpy as np
import pytest

from nimformers import Context, cpu_info


@pytest.fixture(scope="module")
def cpu_ctx():
    try:
        ctx = Context(backend="cpu")
    except Exception as exc:  # pragma: no cover - environment dependent
        pytest.skip(f"CPU backend unavailable: {exc}")
    yield ctx
    ctx.close()


def test_cpu_identity(cpu_ctx):
    assert cpu_ctx.backend == "cpu"
    assert cpu_ctx.backend_id == 4

    info = cpu_info()
    assert info["isa"] in range(6)
    assert info["isa_name"] in {"scalar", "sse4", "avx", "avx2", "avx512", "neon"}
    assert info["l1_bytes"] > 0 and info["l2_bytes"] > 0 and info["l3_bytes"] > 0
    assert info["block_mc"] > 0 and info["block_nc"] > 0 and info["block_kc"] > 0
    if info["has_avx2"]:
        assert info["isa"] in (3, 4) or not info["has_avx"]


def test_cpu_elementwise(cpu_ctx, rng):
    a = rng.standard_normal((4, 8)).astype(np.float32)
    b = rng.standard_normal((4, 8)).astype(np.float32)
    np.testing.assert_allclose(cpu_ctx.add(a, b), a + b, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cpu_ctx.sub(a, b), a - b, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cpu_ctx.mul(a, b), a * b, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cpu_ctx.div(a, b + 3), a / (b + 3),
                               rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(cpu_ctx.sigmoid(a), 1 / (1 + np.exp(-a)),
                               rtol=1e-6, atol=1e-6)


def test_cpu_matmul_and_nn(cpu_ctx, rng):
    a = rng.standard_normal((3, 5)).astype(np.float32)
    b = rng.standard_normal((5, 7)).astype(np.float32)
    np.testing.assert_allclose(cpu_ctx.matmul(a, b), a @ b, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(cpu_ctx.matmul_nt(a, b.T), a @ b,
                               rtol=1e-4, atol=1e-4)

    x = rng.standard_normal((4, 8)).astype(np.float32)
    z = x - x.max(-1, keepdims=True)
    ref = np.exp(z) / np.exp(z).sum(-1, keepdims=True)
    np.testing.assert_allclose(cpu_ctx.softmax(x), ref, rtol=1e-5, atol=1e-6)

    g = np.ones(8, np.float32)
    be = np.zeros(8, np.float32)
    ref_ln = (x - x.mean(-1, keepdims=True)) / np.sqrt(x.var(-1, keepdims=True) + 1e-5)
    np.testing.assert_allclose(cpu_ctx.layernorm(x, g, be), ref_ln,
                               rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(cpu_ctx.sum(x, axis=1), x.sum(axis=1),
                               rtol=1e-5, atol=1e-5)


def test_cpu_attention(cpu_ctx, rng):
    q = rng.standard_normal((1, 2, 4, 4)).astype(np.float32)
    out, scores = cpu_ctx.attention(q, q, q)
    assert out.shape == q.shape
    assert scores.shape == (1, 2, 4, 4)
    # Causal: upper triangle of each score row is zero.
    for i in range(4):
        assert np.all(scores[..., i, i + 1:] == 0)


def test_cpu_quant_q4_resident(cpu_ctx, rng):
    w = rng.standard_normal((16, 32)).astype(np.float32)
    x = rng.standard_normal((5, 32)).astype(np.float32)
    q = cpu_ctx.quantize(w, "int4_asym")
    y = cpu_ctx.linear_q4(x, q)
    np.testing.assert_allclose(y, x @ q.dequantize().T, rtol=2e-5, atol=2e-4)
    with cpu_ctx.resident_q4(q) as resident:
        np.testing.assert_allclose(resident.linear(x), y, rtol=2e-5, atol=2e-4)
    q.close()

    wf = rng.standard_normal((6, 5)).astype(np.float32)  # [K, N]
    xf = rng.standard_normal((3, 6)).astype(np.float32)
    with cpu_ctx.resident_f32(wf) as resident:
        np.testing.assert_allclose(resident.linear(xf), xf @ wf, rtol=1e-4, atol=1e-4)


def test_cpu_micro_vm(cpu_ctx):
    prog = cpu_ctx.vm_program()
    prog.load_f32(1, 0, 0)
    prog.add(1, 1, 1)
    prog.store_f32(1, 0, 0)
    prog.halt()
    mem = np.arange(1, 65, dtype=np.float32)
    np.testing.assert_array_equal(cpu_ctx.vm_run(prog, mem, tg_size=64),
                                  2 * np.arange(1, 65, dtype=np.float32))
    prog.close()


def test_cpu_micro_vm_uops(cpu_ctx):
    from nimformers import Op
    prog = cpu_ctx.vm_program()
    prog.emit(Op.U_LID, 0)
    prog.emit(Op.U_LOAD_IN, 1, 0, 0)
    prog.emit(Op.ADD, 1, 1, 1)
    prog.emit(Op.U_STORE_F32, 1, 0, 0)
    prog.emit(Op.HALT)
    mem = np.arange(1, 33, dtype=np.float32)
    np.testing.assert_array_equal(cpu_ctx.vm_run(prog, mem, tg_size=32),
                                  2 * np.arange(1, 33, dtype=np.float32))
    prog.close()


def test_tiny_llama_cpu(cpu_ctx, repo_root, tmp_path):
    tok_path = repo_root / "test/codestral/models/hf"
    if not (tok_path / "tokenizer.json").is_file():
        pytest.skip("HF tokenizer assets not downloaded")
    rng = np.random.default_rng(11)
    tok = cpu_ctx.tokenizer(tok_path)
    vocab = tok.total_vocab_size
    hidden, inter, layers, heads, kv_heads = 8, 16, 1, 1, 1
    cfg = dict(vocab_size=vocab, hidden_size=hidden, intermediate_size=inter,
               num_hidden_layers=layers, num_attention_heads=heads,
               num_key_value_heads=kv_heads, max_position_embeddings=32,
               rms_norm_eps=1e-6, rope_theta=10000.0)
    sd = cpu_ctx.create_state_dict((vocab, hidden, inter, layers, heads))
    quantized = []

    def add(name, arr):
        q = cpu_ctx.quantize(arr.astype(np.float32), "fp32")
        quantized.append(q)
        sd[name] = q

    add("model.embed_tokens.weight", rng.standard_normal((vocab, hidden)) * 0.02)
    add("model.norm.weight", np.ones(hidden, np.float32))
    add("model.layers.0.input_layernorm.weight", np.ones(hidden, np.float32))
    add("model.layers.0.post_attention_layernorm.weight", np.ones(hidden, np.float32))
    for p in ("q_proj", "k_proj", "v_proj", "o_proj"):
        add(f"model.layers.0.self_attn.{p}.weight", rng.standard_normal((hidden, hidden)) * 0.02)
    for p in ("gate_proj", "up_proj"):
        add(f"model.layers.0.mlp.{p}.weight", rng.standard_normal((inter, hidden)) * 0.02)
    add("model.layers.0.mlp.down_proj.weight", rng.standard_normal((hidden, inter)) * 0.02)
    add("lm_head.weight", rng.standard_normal((vocab, hidden)) * 0.02)

    config_path = tmp_path / "config.json"
    weights_path = tmp_path / "model.nimq"
    config_path.write_text(json.dumps(cfg))
    sd.save(weights_path)
    sd.close()
    for q in quantized:
        q.close()
    model = cpu_ctx.load_llama(config_path, weights_path, upload_resident=True)
    try:
        out = model.generate(tok, "Hello", max_new_tokens=2, temperature=0.0,
                             use_chat_template=False)
        assert isinstance(out, str) and "Hello" in out
    finally:
        model.close()
        tok.close()


def test_cpu_data_and_shape_ops(cpu_ctx, rng):
    tab = rng.standard_normal((8, 4)).astype(np.float32)
    ids = np.array([[1, 2], [3, 0]], dtype=np.int32)
    np.testing.assert_allclose(cpu_ctx.embedding_lookup(tab, ids), tab[ids])
    flat = tab.reshape(-1)
    gidx = np.array([1, 4, 7], dtype=np.int32)
    np.testing.assert_allclose(cpu_ctx.gather(flat, gidx), flat[gidx])

    a = rng.standard_normal((4, 5)).astype(np.float32)
    b = rng.standard_normal((4, 5)).astype(np.float32)
    np.testing.assert_array_equal(cpu_ctx.compare(a, b, "gt"), a > b)
    np.testing.assert_array_equal(cpu_ctx.argmax(a, axis=1), a.argmax(axis=1))
    np.testing.assert_array_equal(cpu_ctx.masked_fill(a, a > 0, 0.0),
                                  np.where(a > 0, 0.0, a))
    np.testing.assert_array_equal(cpu_ctx.where(a > 0, a, b), np.where(a > 0, a, b))

    x = rng.standard_normal((2, 3, 4)).astype(np.float32)
    np.testing.assert_array_equal(cpu_ctx.permute(x, [2, 0, 1]), x.transpose(2, 0, 1))
    np.testing.assert_array_equal(cpu_ctx.concat([x, x], axis=1),
                                  np.concatenate([x, x], axis=1))
    pieces = cpu_ctx.split(x, 2, axis=1)
    np.testing.assert_array_equal(np.concatenate(pieces, axis=1), x)
    np.testing.assert_allclose(cpu_ctx.cumsum(x, axis=2), x.cumsum(axis=2),
                               rtol=1e-5, atol=1e-5)
    ac = np.full((2, 3), 0.9, np.float32)
    bc = rng.standard_normal((2, 3)).astype(np.float32)
    ref = np.zeros_like(bc)
    prev = np.zeros(2, np.float32)
    for t in range(3):
        prev = ac[:, t] * prev + bc[:, t]
        ref[:, t] = prev
    np.testing.assert_allclose(cpu_ctx.affine_scan(ac, bc, axis=1), ref,
                               rtol=1e-5, atol=1e-5)

    g = np.ones(5, np.float32)
    be = np.zeros(5, np.float32)
    dx, dg, db = cpu_ctx.layernorm_backward(a, a, g, be)
    assert dx.shape == a.shape and dg.shape == (5,) and db.shape == (5,)

    out = cpu_ctx.dropout(a, 0.25, training=True, seed=3)
    assert out.shape == a.shape
    np.testing.assert_array_equal(cpu_ctx.dropout(a, 0.25, training=False), a)


def test_cpu_micro_vm_threadgroup_barrier(cpu_ctx):
    from nimformers import Op
    prog = cpu_ctx.vm_program()
    prog.emit(Op.U_LID, 0)                 # U0 = lane
    prog.emit(Op.U_CONST, 1, value=1)      # U1 = 1
    prog.emit(Op.U_ADD, 2, 0, 1)           # U2 = lane+1
    prog.emit(Op.U_TO_R, 2, 2)             # R2 = lane+1
    prog.emit(Op.U_TO_R, 1, 0)             # R1 = lane
    prog.emit(Op.TG_STORE_F32, 2, 1, 0)    # tg[lane] = lane+1
    prog.emit(Op.BARRIER_TG)
    prog.emit(Op.LOAD_CONST, 3, f32=0.0)
    prog.emit(Op.TG_LOAD_F32, 4, 3, 0)     # R4 = tg[0]
    prog.emit(Op.TG_LOAD_F32, 5, 3, 1)     # R5 = tg[1]
    prog.emit(Op.ADD, 4, 4, 5)             # 3
    prog.emit(Op.U_STORE_F32, 4, 0, 0)     # mem[lane] = 3
    prog.emit(Op.HALT)
    mem = np.zeros(4, np.float32)
    out = cpu_ctx.vm_run(prog, mem, tg_floats=2, tg_size=2)
    np.testing.assert_array_equal(out[:2], np.array([3.0, 3.0], np.float32))
    prog.close()


def test_cpu_attention_backward(cpu_ctx, rng):
    q = rng.standard_normal((1, 1, 3, 4)).astype(np.float32)
    out, scores = cpu_ctx.attention(q, q, q)
    dq, dk, dv = cpu_ctx.attention_backward(q, q, q, scores, out)
    assert dq.shape == q.shape and dk.shape == q.shape and dv.shape == q.shape
    assert np.all(np.isfinite(dq))
