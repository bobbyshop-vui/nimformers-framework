"""Tokenizer and end-to-end Llama inference tests (skip without assets)."""
import json

import numpy as np
import pytest

from nimformers import LlamaConfig, LlamaModel, NimformersError, Tokenizer


def _hf_dir(repo_root):
    return repo_root / "test/codestral/models/hf"


def test_hf_tokenizer_roundtrip(ctx, repo_root):
    path = _hf_dir(repo_root)
    if not (path / "tokenizer.json").is_file():
        pytest.skip("HF tokenizer assets not downloaded")
    tok = ctx.tokenizer(path)
    assert isinstance(tok, Tokenizer)
    assert tok.vocab_size > 0
    assert tok.total_vocab_size >= tok.vocab_size
    for text in ("Hello world", "Tiếng Việt có dấu", "def f(x):\n    return x + 1"):
        ids = tok.encode(text)
        assert ids.dtype == np.int32 and ids.size > 0
        assert tok.decode(ids) == text
    assert tok.encode("hello", add_special_tokens=False).size > 0
    tok.close()


def test_sentencepiece_roundtrip(ctx, repo_root):
    path = _hf_dir(repo_root) / "tokenizer.model"
    if not path.is_file():
        pytest.skip("SentencePiece model not downloaded")
    sp = ctx.sentencepiece(path)
    ids = sp.encode("Hello world")
    assert ids.size > 0
    assert sp.decode(ids) == "Hello world"
    assert sp.piece_id("▁Hello") >= 0
    assert isinstance(sp.normalizer, str)
    sp.close()


def test_llama_config(ctx, repo_root):
    path = repo_root / "test/deepseek-inference/models/config.nim.json"
    if not path.is_file():
        pytest.skip("Llama config not downloaded")
    cfg = ctx.llama_config(path)
    assert isinstance(cfg, LlamaConfig)
    assert cfg.vocab_size == 32256
    assert cfg.hidden_size == 4096
    assert cfg.num_layers == 32
    assert cfg.num_heads == 32


@pytest.fixture(scope="module")
def tiny_llama(ctx, repo_root, tmp_path_factory):
    tok_path = _hf_dir(repo_root)
    if not (tok_path / "tokenizer.json").is_file():
        pytest.skip("HF tokenizer assets not downloaded")
    rng = np.random.default_rng(7)
    tok = ctx.tokenizer(tok_path)
    vocab = tok.total_vocab_size
    hidden, inter, layers, heads, kv_heads = 8, 16, 1, 1, 1
    cfg = dict(
        vocab_size=vocab, hidden_size=hidden, intermediate_size=inter,
        num_hidden_layers=layers, num_attention_heads=heads,
        num_key_value_heads=kv_heads, max_position_embeddings=32,
        rms_norm_eps=1e-6, rope_theta=10000.0)
    sd = ctx.create_state_dict((vocab, hidden, inter, layers, heads))
    quantized = []

    def add(name, arr):
        q = ctx.quantize(arr.astype(np.float32), "fp32")
        quantized.append(q)
        sd[name] = q

    add("model.embed_tokens.weight", rng.standard_normal((vocab, hidden)) * 0.02)
    add("model.norm.weight", np.ones(hidden, np.float32))
    add("model.layers.0.input_layernorm.weight", np.ones(hidden, np.float32))
    add("model.layers.0.post_attention_layernorm.weight", np.ones(hidden, np.float32))
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        add(f"model.layers.0.self_attn.{proj}.weight",
            rng.standard_normal((hidden, hidden)) * 0.02)
    for proj in ("gate_proj", "up_proj"):
        add(f"model.layers.0.mlp.{proj}.weight",
            rng.standard_normal((inter, hidden)) * 0.02)
    add("model.layers.0.mlp.down_proj.weight",
        rng.standard_normal((hidden, inter)) * 0.02)
    add("lm_head.weight", rng.standard_normal((vocab, hidden)) * 0.02)

    work = tmp_path_factory.mktemp("tiny-llama")
    config_path = work / "config.json"
    weights_path = work / "model.nimq"
    config_path.write_text(json.dumps(cfg))
    sd.save(weights_path)
    sd.close()
    for q in quantized:
        q.close()
    model = ctx.load_llama(config_path, weights_path, upload_resident=True)
    try:
        yield model, tok
    finally:
        model.close()
        tok.close()


def test_tiny_llama_generate(tiny_llama):
    model, tok = tiny_llama
    assert isinstance(model, LlamaModel)
    assert model.config.vocab_size == tok.total_vocab_size
    out = model.generate(tok, "Hello", max_new_tokens=2, temperature=0.0,
                         use_chat_template=False)
    assert isinstance(out, str)
    assert "Hello" in out


