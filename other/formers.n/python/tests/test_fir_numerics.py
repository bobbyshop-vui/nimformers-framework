"""Numerical validation of traced, optimized and emitted FIR decode graphs.

Tiny random-weight GPT-2, Llama and Mamba models (2 layers) are decoded step by
step and compared, at 1e-4, across:

* the raw traced graph and the optimized graph (``fir.compare_graphs``),
* the reference graph evaluator and independent straight-line Python decoders
  written below without the tracer,
* the emitted FIR files executed by the Python session_vm interpreter
  (``fir.FirModel``), for the optimized and the unoptimized export,
* the hand-written models/gpt2-125m/model.fir on a tiny config.

Each run decodes two sequences without clearing the arena between them, so the
reset of persistent state at position 0 is exercised as well.
"""

import json
import math
import shutil
import tempfile
import unittest
from unittest import mock
from pathlib import Path

from formers import fir
from formers.fir.examples import gpt2, llama, mamba


TOLERANCE = 1e-4
ROOT = Path(__file__).resolve().parents[4]
HAND_GPT2 = ROOT / "models" / "gpt2-125m" / "model.fir"
SEQUENCES = ([1, 5, 2, 7], [3, 3, 0, 6])


# ---- straight-line decoders ------------------------------------------------------

def matvec_io(x, w, n):
    """y = x @ W for W [len(x), n] row-major (HF Conv1D layout)."""
    return [sum(x[i] * w[i * n + j] for i in range(len(x))) for j in range(n)]


def matvec_oi(x, w, n):
    """y = W @ x for W [n, len(x)] row-major (nn.Linear layout)."""
    k = len(x)
    return [sum(w[j * k + i] * x[i] for i in range(k)) for j in range(n)]


def layernorm(x, w, b, eps):
    mean = sum(x) / len(x)
    var = sum((v - mean) ** 2 for v in x) / len(x)
    r = 1 / math.sqrt(var + eps)
    return [(v - mean) * r * w[i] + b[i] for i, v in enumerate(x)]


def rmsnorm(x, w, eps):
    r = 1 / math.sqrt(sum(v * v for v in x) / len(x) + eps)
    return [v * r * w[i] for i, v in enumerate(x)]


def silu(v):
    return v / (1 + math.exp(-v))


def softplus(v):
    return math.log1p(math.exp(v))


def gelu_new(v):
    return 0.5 * v * (1 + math.tanh(math.sqrt(2 / math.pi) * (v + 0.044715 * v ** 3)))


def attend(q, keys, values, scale):
    scores = [scale * sum(a * b for a, b in zip(q, k)) for k in keys]
    m = max(scores)
    e = [math.exp(s - m) for s in scores]
    total = sum(e)
    return [sum(e[j] * values[j][d] for j in range(len(keys))) / total for d in range(len(values[0]))]


def rope(x, position, base=10000.0):
    half = len(x) // 2
    out = list(x)
    for i in range(half):
        angle = position * base ** (-i / half)
        c, s = math.cos(angle), math.sin(angle)
        out[i] = x[i] * c - x[i + half] * s
        out[i + half] = x[i + half] * c + x[i] * s
    return out


class GPT2Decoder:
    def __init__(self, p, c):
        self.p, self.c = p, c
        self.reset()

    def reset(self):
        self.keys = [[] for _ in range(self.c["n_layer"])]
        self.values = [[] for _ in range(self.c["n_layer"])]

    def step(self, token, position):
        p, c = self.p, self.c
        E, H = c["n_embd"], c["n_head"]
        D, eps = E // H, c["layer_norm_epsilon"]
        x = [a + b for a, b in zip(p["transformer.wte.weight"][token * E:(token + 1) * E],
                                   p["transformer.wpe.weight"][position * E:(position + 1) * E])]
        for layer in range(c["n_layer"]):
            w = lambda name: p["transformer.h.%d.%s" % (layer, name)]
            h = layernorm(x, w("ln_1.weight"), w("ln_1.bias"), eps)
            qkv = [a + b for a, b in zip(matvec_io(h, w("attn.c_attn.weight"), 3 * E), w("attn.c_attn.bias"))]
            q, k, v = qkv[:E], qkv[E:2 * E], qkv[2 * E:]
            self.keys[layer].append(k)
            self.values[layer].append(v)
            out = []
            for head in range(H):
                sl = slice(head * D, (head + 1) * D)
                out += attend(q[sl], [r[sl] for r in self.keys[layer]], [r[sl] for r in self.values[layer]],
                              1 / math.sqrt(D))
            x = [a + b + bb for a, b, bb in zip(x, matvec_io(out, w("attn.c_proj.weight"), E), w("attn.c_proj.bias"))]
            h = layernorm(x, w("ln_2.weight"), w("ln_2.bias"), eps)
            f = [gelu_new(a + b) for a, b in zip(matvec_io(h, w("mlp.c_fc.weight"), 4 * E), w("mlp.c_fc.bias"))]
            x = [a + b + bb for a, b, bb in zip(x, matvec_io(f, w("mlp.c_proj.weight"), E), w("mlp.c_proj.bias"))]
        x = layernorm(x, p["transformer.ln_f.weight"], p["transformer.ln_f.bias"], eps)
        return matvec_oi(x, p["transformer.wte.weight"], c["vocab_size"])


class LlamaDecoder:
    def __init__(self, p, c):
        self.p, self.c = p, c
        self.reset()

    def reset(self):
        self.keys = [[] for _ in range(self.c["num_hidden_layers"])]
        self.values = [[] for _ in range(self.c["num_hidden_layers"])]

    def step(self, token, position):
        p, c = self.p, self.c
        E, H, KV = c["hidden_size"], c["num_attention_heads"], c["num_key_value_heads"]
        D, M, eps = E // H, c["intermediate_size"], c["rms_norm_eps"]
        x = p["model.embed_tokens.weight"][token * E:(token + 1) * E]
        for layer in range(c["num_hidden_layers"]):
            w = lambda name: p["model.layers.%d.%s" % (layer, name)]
            h = rmsnorm(x, w("input_layernorm.weight"), eps)
            q = matvec_oi(h, w("self_attn.q_proj.weight"), E)
            k = matvec_oi(h, w("self_attn.k_proj.weight"), KV * D)
            v = matvec_oi(h, w("self_attn.v_proj.weight"), KV * D)
            if c.get("use_rope", True):
                q = sum((rope(q[i * D:(i + 1) * D], position) for i in range(H)), [])
                k = sum((rope(k[i * D:(i + 1) * D], position) for i in range(KV)), [])
            self.keys[layer].append(k)
            self.values[layer].append(v)
            out = []
            for head in range(H):
                g = head // (H // KV)
                sl = slice(g * D, (g + 1) * D)
                out += attend(q[head * D:(head + 1) * D], [r[sl] for r in self.keys[layer]],
                              [r[sl] for r in self.values[layer]], 1 / math.sqrt(D))
            x = [a + b for a, b in zip(x, matvec_oi(out, w("self_attn.o_proj.weight"), E))]
            h = rmsnorm(x, w("post_attention_layernorm.weight"), eps)
            gate = matvec_oi(h, w("mlp.gate_proj.weight"), M)
            up = matvec_oi(h, w("mlp.up_proj.weight"), M)
            x = [a + b for a, b in zip(x, matvec_oi([silu(g_) * u for g_, u in zip(gate, up)],
                                                    w("mlp.down_proj.weight"), E))]
        x = rmsnorm(x, p["model.norm.weight"], eps)
        return matvec_oi(x, p["lm_head.weight"], c["vocab_size"])


class MambaDecoder:
    def __init__(self, p, c):
        self.p, self.c = p, c
        self.reset()

    def reset(self):
        c = self.c
        self.conv = [[[0.0] * (c["d_conv"] - 1) for _ in range(c["d_inner"])] for _ in range(c["n_layer"])]
        self.ssm = [[[0.0] * c["d_state"] for _ in range(c["d_inner"])] for _ in range(c["n_layer"])]

    def step(self, token, position):
        p, c = self.p, self.c
        E, d, n, K, r = c["d_model"], c["d_inner"], c["d_state"], c["d_conv"], c["dt_rank"]
        eps = c["rms_norm_eps"]
        x = p["backbone.embedding.weight"][token * E:(token + 1) * E]
        for layer in range(c["n_layer"]):
            w = lambda name: p["backbone.layers.%d.%s" % (layer, name)]
            h = rmsnorm(x, w("norm.weight"), eps)
            xz = matvec_oi(h, w("mixer.in_proj.weight"), 2 * d)
            xs, z = xz[:d], xz[d:]
            cw, cb = w("mixer.conv1d.weight"), w("mixer.conv1d.bias")
            u = []
            for ch in range(d):
                window = self.conv[layer][ch] + [xs[ch]]
                u.append(silu(sum(cw[ch * K + i] * window[i] for i in range(K)) + cb[ch]))
                self.conv[layer][ch] = window[1:]
            proj = matvec_oi(u, w("mixer.x_proj.weight"), r + 2 * n)
            dt_raw, B, C = proj[:r], proj[r:r + n], proj[r + n:]
            dt = [softplus(a + b) for a, b in zip(matvec_oi(dt_raw, w("mixer.dt_proj.weight"), d),
                                                  w("mixer.dt_proj.bias"))]
            A_log, Dp = w("mixer.A_log"), w("mixer.D")
            y = []
            for ch in range(d):
                state = self.ssm[layer][ch]
                for s in range(n):
                    A = -math.exp(A_log[ch * n + s])
                    state[s] = state[s] * math.exp(dt[ch] * A) + dt[ch] * B[s] * u[ch]
                y.append(sum(state[s] * C[s] for s in range(n)) + Dp[ch] * u[ch])
            gated = [a * silu(b) for a, b in zip(y, z)]
            x = [a + b for a, b in zip(x, matvec_oi(gated, w("mixer.out_proj.weight"), E))]
        x = rmsnorm(x, p["backbone.norm_f.weight"], eps)
        return matvec_oi(x, p["backbone.embedding.weight"], c["vocab_size"])


# ---- helpers ---------------------------------------------------------------------

def checkpoint(graph, seed):
    """Random synthetic weights by checkpoint name, plus offline-flattened aliases."""
    params = fir.random_parameters(graph, seed=seed, scale=0.5)
    params.update({k + ".fir_flat": v for k, v in list(params.items())})
    return params


def largest_difference(a, b):
    assert len(a) == len(b), (len(a), len(b))
    return max(abs(x - y) for x, y in zip(a, b))


CASES = (("gpt2", gpt2, gpt2.GPT2, GPT2Decoder),
         ("llama", llama, llama.Llama, LlamaDecoder),
         ("mamba", mamba, mamba.Mamba, MambaDecoder))


class DecodeNumericsTests(unittest.TestCase):
    def traces(self, source, model):
        raw = fir.trace(model(), source.INPUTS, config=source.CONFIG, name="tiny", optimize=False)
        optimized = fir.trace(model(), source.INPUTS, config=source.CONFIG, name="tiny")
        return raw, optimized

    def test_raw_and_optimized_graphs_agree(self):
        for name, source, model, _ in CASES:
            with self.subTest(model=name):
                raw, optimized = self.traces(source, model)
                self.assertLess(len(optimized.nodes), len(raw.nodes))
                steps = [{"token": [t], "position": [p]} for seq in SEQUENCES for p, t in enumerate(seq)]
                diff = fir.compare_graphs(raw, optimized, steps, checkpoint(raw, 7), rel_tol=0, abs_tol=TOLERANCE)
                self.assertLessEqual(diff, TOLERANCE)

    def test_traced_graphs_match_straight_line_decoders(self):
        for name, source, model, decoder in CASES:
            raw, optimized = self.traces(source, model)
            params = checkpoint(raw, 11)
            for label, graph in (("raw", raw), ("optimized", optimized)):
                with self.subTest(model=name, graph=label):
                    reference = decoder(params, graph.config)
                    states = {}
                    for seq in SEQUENCES:
                        reference.reset()
                        for position, token in enumerate(seq):
                            got = fir.evaluate(graph, {"token": [token], "position": [position]}, params, states)
                            want = reference.step(token, position)
                            self.assertLessEqual(largest_difference(got, want), TOLERANCE)

    def test_emitted_fir_runs_in_python_vm_like_straight_line_decoders(self):
        for name, source, model, decoder in CASES:
            raw, optimized = self.traces(source, model)
            params = checkpoint(raw, 13)
            for label, graph in (("raw", raw), ("optimized", optimized)):
                with self.subTest(model=name, export=label), tempfile.TemporaryDirectory() as directory:
                    path = fir.export(graph, directory)
                    vm = fir.FirModel(path, weights=params)
                    reference = decoder(params, graph.config)
                    worst = 0.0
                    for seq in SEQUENCES:
                        vm.reset()
                        reference.reset()
                        for position, token in enumerate(seq):
                            worst = max(worst, largest_difference(vm.step(token), reference.step(token, position)))
                    self.assertLessEqual(worst, TOLERANCE)

    def test_attention_variants_mha_gqa_with_and_without_rope(self):
        variants = ({"use_rope": False}, {"num_key_value_heads": 2}, {"num_key_value_heads": 2, "use_rope": False},
                    {"hidden_size": 8, "num_attention_heads": 4, "num_key_value_heads": 2, "intermediate_size": 8})
        for change in variants:
            config = dict(llama.CONFIG, **change)
            with self.subTest(**change), tempfile.TemporaryDirectory() as directory:
                graph = fir.trace(llama.Llama(config), llama.INPUTS, config=config, name="tiny")
                attention = [n for n in graph.nodes if n.op == "attention_decode"]
                self.assertEqual(len(attention), 2)
                self.assertEqual(bool(attention[0].attrs.get("rope")), config["use_rope"])
                self.assertEqual(attention[0].attrs["group"],
                                 config["num_attention_heads"] // config["num_key_value_heads"])
                params = checkpoint(graph, 19)
                vm = fir.FirModel(fir.export(graph, directory), weights=params)
                reference = LlamaDecoder(params, config)
                for seq in SEQUENCES:
                    vm.reset()
                    reference.reset()
                    for position, token in enumerate(seq):
                        self.assertLessEqual(largest_difference(vm.step(token), reference.step(token, position)),
                                             TOLERANCE)

    @unittest.skipUnless(HAND_GPT2.is_file(), "hand-written GPT-2 FIR is not present")
    def test_hand_written_gpt2_fir_matches_traced_example_on_tiny_config(self):
        config = dict(gpt2.CONFIG)
        graph = fir.trace(gpt2.GPT2(), gpt2.INPUTS, config=config, name="tiny")
        params = checkpoint(graph, 17)
        with tempfile.TemporaryDirectory() as directory:
            shutil.copy(HAND_GPT2, Path(directory) / "model.fir")
            (Path(directory) / "config.json").write_text(json.dumps(graph.config))
            hand = fir.FirModel(Path(directory) / "model.fir", weights=params)
            emitted = fir.FirModel(fir.export(graph, Path(directory) / "emitted"), weights=params)
        reference = GPT2Decoder(params, graph.config)
        for seq in SEQUENCES:
            hand.reset()
            emitted.reset()
            reference.reset()
            for position, token in enumerate(seq):
                want = reference.step(token, position)
                self.assertLessEqual(largest_difference(hand.step(token), want), TOLERANCE)
                self.assertLessEqual(largest_difference(emitted.step(token), want), TOLERANCE)
        # Same per-layer cache buffers: [n_positions * n_embd] key and value caches.
        self.assertEqual(hand.prog["buffers"]["key_cache"]["elements"],
                         emitted.prog["buffers"]["key_cache"]["elements"])
        self.assertTrue(emitted.prog["buffers"]["key_cache"]["per_layer"])


class PatternTests(unittest.TestCase):
    def ops(self, source, model):
        graph = fir.trace(model(), source.INPUTS, config=source.CONFIG, name="tiny")
        return graph, [n.op for n in graph.nodes]

    def test_gpt2_lowers_to_attention_norm_and_pointwise_kernels(self):
        graph, ops = self.ops(gpt2, gpt2.GPT2)
        attention = [n for n in graph.nodes if n.op == "attention_decode"]
        self.assertEqual(len(attention), 2)
        self.assertEqual((attention[0].attrs["heads"], attention[0].attrs["group"]), (2, 1))
        self.assertIsNone(attention[0].attrs.get("rope"))
        self.assertIn("norm", ops)  # residual add folded into the second LayerNorm
        self.assertNotIn("softmax", ops)
        self.assertNotIn("matmul", ops)
        text = fir.emit(graph)
        self.assertIn("kernel script.attention ", text)
        self.assertIn("kernel script.add_gelu_new ", text)

    def test_llama_fuses_rotary_into_grouped_attention(self):
        graph, ops = self.ops(llama, llama.Llama)
        attention = next(n for n in graph.nodes if n.op == "attention_decode")
        self.assertEqual((attention.attrs["heads"], attention.attrs["group"]), (2, 2))
        self.assertEqual(attention.attrs["rope"]["q"], True)
        self.assertNotIn("rotary", ops)
        self.assertIn("kernel script.silu_mul ", fir.emit(graph))

    def test_mamba_lowers_conv_state_and_selective_scan_step(self):
        graph, ops = self.ops(mamba, mamba.Mamba)
        self.assertEqual(ops.count("conv1d_step"), 2)
        self.assertEqual(ops.count("state_update_reduce"), 2)
        self.assertNotIn("cat", ops)
        text = fir.emit(graph)
        self.assertIn("kernel script.conv_step ", text)
        self.assertIn("kernel script.ssm_step ", text)
        self.assertIn("buf conv_state f32 24 per_layer", text)
        self.assertIn("buf ssm_state f32 32 per_layer", text)


class VmTests(unittest.TestCase):
    CONFIG = {"vocab_size": 4, "n_positions": 2, "layer_norm_epsilon": 1e-5}
    HEADER = ("fir 1\nmodel t\nruntime graph\nlayout script\nsym layers = 1\nsym positions = n_positions\n"
              "sym vocab = vocab_size\nbuf token f32 1 input:token\nbuf position f32 1 input:position\n"
              "buf other f32 4\nbuf logits f32 4 output:logits\n")

    def model(self, body, node):
        return fir.FirModel(self.HEADER + body + node, config=self.CONFIG)

    def test_stores_outside_output_ports_are_rejected(self):
        body = "kernel k in=1 out=1 threads=1\n  uconst 0 4\n  loadconstf32 1 2\n  ustoref32 1 0 @out0\nendkernel\n"
        model = self.model(body, "node n micro k token -> logits\n")
        with self.assertRaisesRegex(fir.FirVmError, "outside its output buffers"):
            model.step(0)

    def test_thread_ids_loops_and_float_rounding(self):
        body = ("kernel k in=1 out=1 threads=out0.size\n  rtou 0 0\n  utor 1 0\n  loadconstf32 2 3\n"
                "  loadconstf32 3 0\n  forbegin 2\n    addconst 3 3 0.1\n  forend\n  add 1 1 3\n"
                "  ustoref32 1 0 @out0\nendkernel\n")
        logits = self.model(body, "node n micro k token -> logits\n").step(1)
        acc = 0.0
        for _ in range(3):
            acc = fir.vm.f32(acc + fir.vm.f32(0.1))
        self.assertEqual(logits, [fir.vm.f32(i + acc) for i in range(4)])

    def test_unsupported_mnemonics_and_step_limit_fail_clearly(self):
        body = "kernel k in=1 out=1 threads=1\n  barriertg\nendkernel\n"
        with self.assertRaisesRegex(fir.FirVmError, "not supported by the Python VM"):
            self.model(body, "node n micro k token -> logits\n")
        body = "kernel k in=1 out=1 threads=1\n  loadconstf32 1 100000\n  forbegin 1\n  forend\nendkernel\n"
        model = self.model(body, "node n micro k token -> logits\n")
        with mock.patch.object(fir.vm, "MAX_STEPS", 1000), \
                self.assertRaisesRegex(fir.FirVmError, "VM steps per dispatch"):
            model.step(0)


if __name__ == "__main__":
    unittest.main()
