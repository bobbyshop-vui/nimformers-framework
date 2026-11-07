"""The torch-shaped front end of fir: tensor API, modules, torch_compat, re-fusion, from_torch.

Tiny synthetic tensors and random weights only. HF-style Llama and GPT-2 code
(tests/hf_style_models.py) is traced and checked to lower into the same
kernels as the hand-written examples, and decoded against the straight-line
reference decoders, the reference evaluator and the emitted FIR in the Python VM.
"""

import json
import math
import sys
import tempfile
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock

from formers import fir
from formers.fir import coverage_table
from formers.fir import torch_compat as torch
from formers.fir.torch_compat import nn
from formers.fir.torch_compat.nn import functional as F
from formers.fir.torch_compat.transformers import ACT2FN, PretrainedConfig
from formers.fir.examples import gpt2 as gpt2_example, llama as llama_example

from tests import hf_style_models as H
from tests.test_fir import fir_check
from tests.test_fir_numerics import GPT2Decoder, LlamaDecoder, SEQUENCES, checkpoint, largest_difference


TOLERANCE = 1e-4
INPUTS = (fir.TensorSpec((1, 1), "token"), fir.TensorSpec((1, 1), "position"))
ROOT = Path(__file__).resolve().parents[4]


def close(test, got, want, tol=1e-6):
    got, want = _flat(got), _flat(want)
    test.assertEqual(len(got), len(want))
    for a, b in zip(got, want):
        test.assertLessEqual(abs(a - b), tol * max(1.0, abs(b)), (got, want))


def _flat(x):
    if isinstance(x, (list, tuple)): return [y for v in x for y in _flat(v)]
    return [float(x)]


class Probe(fir.nn.Module):
    """Wraps ``fn(x)`` so it can be traced on one runtime input."""

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x)


def traced(fn, shape, values, optimize=True):
    """Trace ``fn`` on a runtime f32 input and evaluate it with ``values``."""
    graph = fir.trace(Probe(fn), (fir.TensorSpec(shape),), config={}, optimize=optimize)
    return fir.evaluate(graph, {"input0": list(values)}), graph


# ---- tensor API ----------------------------------------------------------------------------------

class TensorApiTests(unittest.TestCase):
    X = [[1.0, -2.0, 3.0], [4.0, 5.0, -6.0]]

    def x(self): return torch.tensor(self.X)

    def test_size_shape_and_metadata(self):
        x = self.x()
        self.assertEqual(x.size(), torch.Size([2, 3]))
        self.assertEqual(x.size(-1), 3)
        self.assertEqual(tuple(x.shape[:-1]) + (5,), (2, 5))
        self.assertEqual((x.ndim, x.dim(), x.numel()), (2, 2, 6))
        self.assertIs(x.dtype, torch.float32)
        self.assertEqual(x.device.type, "cpu")

    def test_max_min_return_named_tuples(self):
        r = self.x().max(dim=1)
        self.assertEqual((r.values.tolist(), r.indices.tolist()), ([3.0, 5.0], [2, 1]))
        self.assertIs(r.indices.dtype, torch.int64)
        values, indices = torch.min(self.x(), 0, keepdim=True)
        self.assertEqual((values.tolist(), indices.tolist()), ([[1.0, -2.0, -6.0]], [[0, 0, 1]]))
        self.assertEqual(torch.max(self.x()).item(), 5.0)
        self.assertEqual(self.x().argmax(-1).tolist(), [2, 1])

    def test_arange_uses_torch_argument_order_and_dtype(self):
        self.assertEqual(torch.arange(4).tolist(), [0, 1, 2, 3])
        self.assertEqual(torch.arange(1, 7, 2).tolist(), [1, 3, 5])
        self.assertEqual(torch.arange(0, 1, 0.25).tolist(), [0.0, 0.25, 0.5, 0.75])
        self.assertIs(torch.arange(3).dtype, torch.int64)
        self.assertIs(torch.arange(3, dtype=torch.float32).dtype, torch.float32)

    def test_indexing_with_none_ellipsis_negative_steps_and_masks(self):
        x = self.x()
        self.assertEqual(x[None, ..., 1].tolist(), [[-2.0, 5.0]])
        self.assertEqual(x[:, None].shape, torch.Size([2, 1, 3]))
        self.assertEqual(x[..., ::-1].tolist(), [[3.0, -2.0, 1.0], [-6.0, 5.0, 4.0]])
        self.assertEqual(x[x > 0].tolist(), [1.0, 3.0, 4.0, 5.0])
        self.assertEqual(x[:, torch.tensor([2, 0])].tolist(), [[3.0, 1.0], [-6.0, 4.0]])

    def test_boolean_operators(self):
        b = self.x() > 0
        self.assertEqual((b & ~(self.x() > 4)).tolist(), [[True, False, True], [True, False, False]])
        self.assertEqual((b | (self.x() < -5)).tolist(), [[True, False, True], [True, True, True]])
        self.assertEqual((b ^ True).tolist(), [[False, True, False], [False, False, True]])
        self.assertIs(b.dtype, torch.bool)

    def test_creation_and_combination_functions(self):
        x = self.x()
        self.assertEqual(torch.where(x > 0, x, torch.zeros_like(x)).tolist(), [[1.0, 0.0, 3.0], [4.0, 5.0, 0.0]])
        self.assertEqual(torch.stack([x, x]).shape, torch.Size([2, 2, 3]))
        self.assertEqual(torch.full((2,), 7).tolist(), [7, 7])
        self.assertEqual(torch.ones(2, 2).tolist(), [[1.0, 1.0], [1.0, 1.0]])
        self.assertEqual(torch.full_like(x, 2.0)[0].tolist(), [2.0, 2.0, 2.0])
        self.assertEqual(torch.tril(torch.ones(3, 3)).tolist(), [[1, 0, 0], [1, 1, 0], [1, 1, 1]])
        self.assertEqual(torch.triu(torch.ones(3, 3), 1).tolist(), [[0, 1, 1], [0, 0, 1], [0, 0, 0]])
        self.assertEqual(torch.cat([x, x], -1).shape, torch.Size([2, 6]))
        self.assertEqual([t.tolist() for t in x.split([1, 2], dim=1)], [[[1.0], [4.0]], [[-2.0, 3.0], [5.0, -6.0]]])
        self.assertEqual([t.shape for t in torch.chunk(x, 2)], [torch.Size([1, 3])] * 2)
        self.assertEqual(torch.einsum("ij,kj->ik", x, x).tolist(), [[14.0, -24.0], [-24.0, 77.0]])
        self.assertEqual(torch.einsum("...i,...i->...", x, x).tolist(), [14.0, 77.0])
        self.assertEqual(torch.outer(torch.tensor([1., 2.]), torch.tensor([3., 4.])).tolist(), [[3., 4.], [6., 8.]])
        self.assertEqual(torch.cumsum(x, 1).tolist(), [[1.0, -1.0, 2.0], [4.0, 9.0, 3.0]])
        self.assertEqual(torch.flip(x, [1]).tolist(), [[3.0, -2.0, 1.0], [-6.0, 5.0, 4.0]])
        self.assertEqual(torch.roll(x, 1, 1).tolist(), [[3.0, 1.0, -2.0], [-6.0, 4.0, 5.0]])

    def test_method_forms_and_in_place_methods(self):
        x = self.x()
        self.assertEqual(x.add(1, alpha=2).tolist(), [[3.0, 0.0, 5.0], [6.0, 7.0, -4.0]])
        self.assertEqual(torch.tensor([7, -7]).div(2, rounding_mode="floor").tolist(), [3, -4])
        self.assertEqual(torch.tensor([7, -7]).div(2, rounding_mode="trunc").tolist(), [3, -3])
        self.assertEqual(x.clamp(min=0).sum(-1).tolist(), [4.0, 9.0])
        self.assertEqual(x.transpose(0, 1).contiguous().view(-1).tolist(), [1.0, 4.0, -2.0, 5.0, 3.0, -6.0])
        self.assertEqual(x.unsqueeze(0).expand(2, -1, -1).shape, torch.Size([2, 2, 3]))
        y = x.clone()
        y.mul_(2).add_(1)
        self.assertEqual(y.tolist(), [[3.0, -3.0, 7.0], [9.0, 11.0, -11.0]])
        y[0, 1:] = 0.0
        self.assertEqual(y.tolist(), [[3.0, 0.0, 0.0], [9.0, 11.0, -11.0]])
        self.assertEqual(x.float().type_as(x).to(torch.float32).tolist(), self.X)

    def test_dtype_promotion(self):
        i = torch.tensor([1, 2])
        self.assertIs(i.dtype, torch.int64)
        self.assertIs((i / 2).dtype, torch.float32)
        self.assertIs((i + 1).dtype, torch.int64)
        self.assertIs((torch.tensor([1.0], dtype=torch.bfloat16) + 1.0).dtype, torch.bfloat16)
        self.assertIs((torch.tensor([1.0], dtype=torch.bfloat16) + torch.tensor([1.0])).dtype, torch.float32)
        self.assertIs(torch.result_type(i, 1.5), torch.float32)
        self.assertEqual(torch.finfo(torch.float16).max, 65504.0)

    def test_traced_tensor_functions_match_python(self):
        x = [0.5, -1.0, 2.0, 3.0, -0.25, 1.5]
        fn = lambda t: (torch.cumsum(t.view(2, 3), -1).flip(-1) + torch.tril(torch.ones(2, 3))
                        - t.view(2, 3).roll(1, 1) * (t.view(2, 3) > 0)).max(-1).values
        got, graph = traced(fn, (6,), x)
        rows = [x[:3], x[3:]]
        want = []
        for r, row in enumerate(rows):
            c = [sum(row[:j + 1]) for j in range(3)][::-1]
            tril = [1.0 if j <= r else 0.0 for j in range(3)]
            rolled = [row[-1]] + row[:-1]
            want.append(max(c[j] + tril[j] - rolled[j] * (row[j] > 0) for j in range(3)))
        close(self, got, want)
        self.assertEqual(graph.values[graph.output].shape, (2,))

    def test_traced_einsum_stack_and_split(self):
        x = [1.0, 2.0, 3.0, 4.0]
        fn = lambda t: torch.einsum("i,j->ij", *t.split(2)) + torch.stack(t.chunk(2), 0)
        got, _ = traced(fn, (4,), x)
        close(self, got, [1 * 3 + 1, 1 * 4 + 2, 2 * 3 + 3, 2 * 4 + 4])

    def test_data_dependent_operations_raise_unsupported(self):
        with self.assertRaises(fir.Unsupported):
            traced(lambda t: t.nonzero(), (3,), [1.0, 0.0, 1.0])
        with self.assertRaises(fir.Unsupported):
            traced(lambda t: torch.randn(3) + t, (3,), [1.0, 0.0, 1.0])


# ---- modules -----------------------------------------------------------------------------------------

class Buffered(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("scale", torch.tensor([2.0, 3.0]))
        self.register_buffer("table", torch.arange(2, dtype=torch.float32), persistent=False)
        self.blocks = nn.ModuleDict({"a": nn.Linear(2, 2, bias=False), "b": nn.Identity()})
        self.gains = nn.ParameterList([nn.Parameter(torch.ones(2)) for _ in range(2)])

    def forward(self, x):
        y = self.blocks["a"](x * self.scale + self.table)
        for g in self.gains: y = y * g
        return self.blocks["b"](y)


class ModuleTests(unittest.TestCase):
    def test_buffers_are_graph_constants(self):
        m = Buffered().eval()
        graph = fir.trace(m, (fir.TensorSpec((2,)),), config={}, optimize=False)
        constants = sorted(v.constant for v in graph.values.values() if v.constant is not None and len(v.constant) == 2)
        self.assertIn((0.0, 1.0), constants)
        self.assertIn((2.0, 3.0), constants)
        weights = sorted(v.weight.name for v in graph.values.values() if v.weight)
        self.assertEqual(weights, ["blocks.a.weight", "gains.0", "gains.1"])
        params = {"blocks.a.weight": [1.0, 2.0, 3.0, 4.0], "gains.0": [1.0, 2.0], "gains.1": [3.0, 1.0]}
        got = fir.evaluate(graph, {"input0": [1.0, 1.0]}, params)
        h = [2.0, 4.0]
        close(self, got, [(1 * h[0] + 2 * h[1]) * 3, (3 * h[0] + 4 * h[1]) * 2])

    def test_registry_state_dict_and_load_state_dict(self):
        m = Buffered()
        self.assertEqual([k for k, _ in m.named_buffers()], ["scale", "table"])
        self.assertEqual(list(m.state_dict()), ["scale", "blocks.a.weight", "gains.0", "gains.1"])
        result = m.load_state_dict({"scale": torch.tensor([5.0, 6.0]), "blocks.a.weight": torch.zeros(2, 2)},
                                   strict=False)
        self.assertEqual(sorted(result.missing_keys), ["gains.0", "gains.1"])
        self.assertEqual(m.scale.tolist(), [5.0, 6.0])
        with self.assertRaises(RuntimeError):
            m.load_state_dict({"blocks.a.weight": torch.zeros(3, 2)}, strict=False)
        seen = []
        m.apply(lambda module: seen.append(type(module).__name__))
        self.assertIn("ModuleDict", seen)
        self.assertTrue(m.train().training and m.blocks["a"].training)
        self.assertFalse(m.eval().training)
        self.assertIs(m.to(torch.bfloat16).to("cpu"), m)
        self.assertEqual(m.get_submodule("blocks.a").weight.shape, (2, 2))

    def test_config_from_json_and_activation_registry(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(json.dumps({"n_embd": 8, "n_head": 2, "n_layer": 3, "layer_norm_epsilon": 1e-5}))
            config = H.GPT2Config.from_pretrained(directory)
        self.assertEqual((config.hidden_size, config.num_attention_heads, config.num_hidden_layers), (8, 2, 3))
        self.assertEqual((config.head_dim, config.num_key_value_heads), (4, 2))
        self.assertEqual(config.to_dict()["n_embd"], 8)
        self.assertFalse(config.output_attentions)
        generic = PretrainedConfig(hidden_size=6, num_attention_heads=3)
        self.assertEqual(generic.head_dim, 2)
        x = torch.tensor([-1.0, 0.5, 2.0])
        for name, ref in (("gelu_new", lambda v: 0.5 * v * (1 + math.tanh(math.sqrt(2 / math.pi) * (v + 0.044715 * v ** 3)))),
                          ("gelu", lambda v: 0.5 * v * (1 + math.erf(v / math.sqrt(2)))),
                          ("silu", lambda v: v / (1 + math.exp(-v))),
                          ("quick_gelu", lambda v: v / (1 + math.exp(-1.702 * v)))):
            close(self, ACT2FN[name](x).tolist(), [ref(v) for v in x.tolist()], 1e-6)

    def test_dropout_in_training_mode_needs_rng(self):
        m = nn.Dropout(0.1)
        self.assertEqual(m(torch.ones(2)).tolist(), [1.0, 1.0])
        m.train()
        with self.assertRaises(fir.Unsupported):
            m(torch.ones(2))


# ---- torch.nn.functional ----------------------------------------------------------------------------

class FunctionalTests(unittest.TestCase):
    def test_norms_linear_and_activations(self):
        x = [[1.0, -2.0, 3.0], [0.5, 0.0, -1.0]]
        t = torch.tensor(x)
        w, b = torch.tensor([[1.0, 0.0, 1.0], [0.0, 2.0, 0.0]]), torch.tensor([0.5, -0.5])
        close(self, F.linear(t, w, b).tolist(), [[4.5, -4.5], [0.0, -0.5]])
        for row, got in zip(x, F.rms_norm(t, (3,), eps=1e-6).tolist()):
            r = math.sqrt(sum(v * v for v in row) / 3 + 1e-6)
            close(self, got, [v / r for v in row])
        for row, got in zip(x, F.layer_norm(t, (3,), eps=1e-5).tolist()):
            m = sum(row) / 3
            s = math.sqrt(sum((v - m) ** 2 for v in row) / 3 + 1e-5)
            close(self, got, [(v - m) / s for v in row])
        close(self, F.softmax(torch.tensor([1.0, 2.0, 3.0]), -1).tolist(),
              [math.exp(v) / sum(math.exp(u) for u in (1, 2, 3)) for v in (1, 2, 3)])
        close(self, F.silu(torch.tensor([1.0])).tolist(), [1 / (1 + math.exp(-1))])
        close(self, F.gelu(torch.tensor([1.0]), approximate="tanh").tolist(),
              [0.5 * (1 + math.tanh(math.sqrt(2 / math.pi) * 1.044715))])

    def test_scaled_dot_product_attention_options(self):
        q = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 0.0]]]])
        out = F.scaled_dot_product_attention(q, q, q, is_causal=True).tolist()
        a = math.exp(1 / math.sqrt(2))
        close(self, out[0][0][1], [1 / (1 + a), a / (1 + a)])
        close(self, out[0][1][1], [0.5, 0.5])
        k = q[:, :1]
        grouped = F.scaled_dot_product_attention(q, k, k, enable_gqa=True)
        expanded = F.scaled_dot_product_attention(q, k.expand(1, 2, 2, 2), k.expand(1, 2, 2, 2))
        close(self, grouped.tolist(), expanded.tolist())
        with self.assertRaises(fir.Unsupported):
            F.scaled_dot_product_attention(q, q, q, dropout_p=0.1)

    def test_conv1d_padding_and_stride_match_direct_convolution(self):
        x = [[[1.0, 2.0, -1.0, 0.5, 3.0]]]
        w = [[[0.5, -1.0, 2.0]]]
        def direct(pad, stride):
            row = [0.0] * pad + x[0][0] + [0.0] * pad
            return [sum(row[s + j] * w[0][0][j] for j in range(3)) for s in range(0, len(row) - 2, stride)]
        for pad, stride in ((0, 1), (1, 1), (2, 1), (1, 2), (3, 2)):
            with self.subTest(padding=pad, stride=stride):
                got = F.conv1d(torch.tensor(x), torch.tensor(w), padding=pad, stride=stride).tolist()
                close(self, got, direct(pad, stride))

    def test_grad_modes_and_autocast_are_no_ops(self):
        @torch.no_grad()
        def f(t): return t * 2
        with torch.inference_mode(), torch.autocast("cpu", enabled=False):
            self.assertEqual(f(torch.ones(1)).tolist(), [2.0])
        with self.assertRaises(fir.Unsupported):
            torch.autocast("cpu", dtype=torch.bfloat16)


# ---- tracer inputs and outputs -------------------------------------------------------------------------

class TwoInputs(fir.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = fir.nn.Linear(2, 2, bias=False)

    def forward(self, token, position, bias, scale=2.0):
        h = self.proj(token.float().expand(1, 2)) + bias
        return h * scale, h.sum(-1)


class TracerInterfaceTests(unittest.TestCase):
    def test_extra_inputs_and_multiple_outputs_evaluate_but_do_not_export(self):
        inputs = {"token": fir.TensorSpec((1,), "token"), "position": fir.TensorSpec((1,), "position"),
                  "bias": fir.TensorSpec((1, 2)), "scale": 3.0}
        graph = fir.trace(TwoInputs(), inputs, config={})
        self.assertEqual(len(graph.output_names()), 2)
        params = {"proj.weight": [1.0, 0.0, 0.0, 2.0]}
        out = fir.evaluate(graph, {"token": [2.0], "position": [0.0], "bias": [1.0, -1.0]}, params, all_outputs=True)
        close(self, out, [[9.0, 9.0], [6.0]])
        with tempfile.TemporaryDirectory() as directory, self.assertRaisesRegex(fir.LoweringError, "runner ABI"):
            fir.export(graph, directory)


# ---- HF-style models ---------------------------------------------------------------------------------------

def kernels(path):
    """Kernel blocks and node operations of an emitted FIR file (buffer names stripped)."""
    text = Path(path).read_text(encoding="utf-8")
    blocks = text.split("\nkernel ")[1:]
    bodies = sorted("kernel " + b.split("\nendkernel")[0] for b in blocks)
    nodes = [line.split()[2:4] for line in text.splitlines() if line.startswith("node ")]
    repeats = text.count("\nrepeat ")
    return bodies, nodes, repeats


class HfStyleModelTests(unittest.TestCase):
    def export(self, graph, directory):
        path = fir.export(graph, directory)
        fir_check(self, path)
        return path

    def test_llama_traces_into_the_kernels_of_the_hand_written_example(self):
        model, config = H.llama()
        graph = fir.trace(model, INPUTS, config=config, name="hf_llama")
        ops = Counter(n.op for n in graph.nodes)
        self.assertEqual(ops["attention_decode"], 2)
        self.assertTrue(all(n.attrs.get("rope") for n in graph.nodes if n.op == "attention_decode"))
        for op in ("softmax", "matmul", "rotary", "cat", "sin", "cos", "rsqrt", "mean", "where"):
            self.assertNotIn(op, ops)
        example = fir.trace(llama_example.Llama(), llama_example.INPUTS, config=llama_example.CONFIG, name="tiny")
        with tempfile.TemporaryDirectory() as directory:
            hf = kernels(self.export(graph, Path(directory) / "hf"))
            ex = kernels(fir.export(example, Path(directory) / "example"))
        self.assertEqual(hf, ex)

    def test_gpt2_traces_into_the_kernels_of_the_hand_written_example(self):
        model, config = H.gpt2()
        graph = fir.trace(model, INPUTS, config=config, name="hf_gpt2")
        example = fir.trace(gpt2_example.GPT2(), gpt2_example.INPUTS, config=gpt2_example.CONFIG, name="tiny")
        with tempfile.TemporaryDirectory() as directory:
            hf = kernels(self.export(graph, Path(directory) / "hf"))
            ex = kernels(fir.export(example, Path(directory) / "example"))
        name = lambda body: body.split()[1]
        self.assertEqual([name(b) for b in hf[0]], [name(b) for b in ex[0]])
        self.assertEqual(hf[1], ex[1])
        self.assertEqual(hf[2], ex[2])
        # The pointwise and norm kernels are the same programs; attention differs in
        # the StaticCache layout ([kv_heads, positions, head_dim] instead of [positions, width]).
        same = {"script.add", "script.layernorm", "script.add_gelu_new"}
        self.assertEqual([b for b in hf[0] if name(b) in same], [b for b in ex[0] if name(b) in same])

    def test_variants_lower_to_compact_kernels(self):
        cases = (("llama sdpa", H.llama, {"_attn_implementation": "sdpa"}, "silu"),
                 ("llama mha", H.llama, {"num_key_value_heads": 2}, "silu"),
                 ("gpt2 gelu_fast", H.gpt2, {"activation_function": "gelu_fast"}, "gelu_new"),
                 ("gpt2 gelu_pytorch_tanh", H.gpt2, {"activation_function": "gelu_pytorch_tanh"}, "gelu_new"),
                 ("gpt2 quick_gelu", H.gpt2, {"activation_function": "quick_gelu"}, "gelu_quick"),
                 ("gpt2 gelu", H.gpt2, {"activation_function": "gelu"}, "gelu"))
        for label, make, change, activation in cases:
            with self.subTest(label):
                model, config = make(**change)
                graph = fir.trace(model, INPUTS, config=config)
                ops = Counter(n.op for n in graph.nodes)
                self.assertEqual(ops["attention_decode"], 2)
                self.assertNotIn("softmax", ops)
                self.assertNotIn("matmul", ops)
                trees = repr([n.attrs.get("tree") for n in graph.nodes if n.op == "fused"])
                self.assertIn("'%s'" % activation, trees)

    def decode(self, make, decoder, change=None):
        model, config = make(**(change or {}))
        raw = fir.trace(model, INPUTS, config=config, name="hf", optimize=False)
        model, config = make(**(change or {}))
        optimized = fir.trace(model, INPUTS, config=config, name="hf")
        params = checkpoint(raw, 23)
        steps = [{"token": [t], "position": [p]} for seq in SEQUENCES for p, t in enumerate(seq)]
        self.assertLessEqual(fir.compare_graphs(raw, optimized, steps, params, rel_tol=0, abs_tol=TOLERANCE),
                             TOLERANCE)
        reference_config = dict(config.to_dict())
        reference_config.setdefault("n_layer", config.num_hidden_layers)
        reference_config.setdefault("num_hidden_layers", config.num_hidden_layers)
        reference_config.setdefault("num_key_value_heads", config.num_key_value_heads)
        with tempfile.TemporaryDirectory() as directory:
            vm = fir.FirModel(self.export(optimized, directory), weights=params)
            reference = decoder(params, reference_config)
            states = {}
            for seq in SEQUENCES:
                vm.reset()
                reference.reset()
                for position, token in enumerate(seq):
                    want = reference.step(token, position)
                    got = fir.evaluate(optimized, {"token": [token], "position": [position]}, params, states)
                    self.assertLessEqual(largest_difference(got, want), TOLERANCE)
                    self.assertLessEqual(largest_difference(vm.step(token), want), TOLERANCE)

    def test_llama_decodes_like_the_straight_line_decoder(self):
        self.decode(H.llama, LlamaDecoder)
        self.decode(H.llama, LlamaDecoder, {"_attn_implementation": "sdpa", "num_key_value_heads": 2})

    def test_gpt2_decodes_like_the_straight_line_decoder(self):
        self.decode(H.gpt2, GPT2Decoder)

    def test_checkpoint_names_are_the_hf_names(self):
        model, config = H.llama()
        graph = fir.trace(model, INPUTS, config=config)
        names = {v.weight.name for v in graph.values.values() if v.weight}
        self.assertIn("model.layers.0.self_attn.q_proj.weight", names)
        self.assertIn("lm_head.weight", names)
        model, config = H.gpt2()
        graph = fir.trace(model, INPUTS, config=config)
        names = {v.weight.name for v in graph.values.values() if v.weight}
        self.assertIn("transformer.h.1.attn.c_attn.weight", names)


# ---- from_torch against a fake torch.fx object model --------------------------------------------------------

class _Node:
    def __init__(self, op, target, args=(), kwargs=None, name=""):
        self.op, self.target, self.args, self.kwargs, self.name = op, target, tuple(args), kwargs or {}, name

    def __hash__(self): return id(self)


class _Aten:
    def __init__(self, text): self.text = text
    def __str__(self): return self.text


class _DType:
    def __init__(self, name): self.name = name
    def __str__(self): return "torch." + self.name


class _Tensor:
    def __init__(self, shape, data=None, dtype="float32"):
        self.shape, self.data, self.dtype = tuple(shape), data, _DType(dtype)
    def detach(self): return self
    def cpu(self): return self
    def tolist(self): return self.data


class _Leaf:
    def __init__(self, **attrs): self.__dict__.update(attrs)


def _fake_torch():
    fake = types.ModuleType("torch")
    fx = types.ModuleType("torch.fx")

    class GraphModule:
        def __init__(self, nodes, parameters, buffers, modules):
            self.graph = types.SimpleNamespace(nodes=nodes)
            self._p, self._b, self._m = parameters, buffers, modules
        def named_parameters(self): return iter(self._p.items())
        def named_buffers(self): return iter(self._b.items())
        def named_modules(self): return iter([("", self)] + list(self._m.items()))
        def get_submodule(self, name): return self._m[name]

    fx.GraphModule = GraphModule
    fx.symbolic_trace = lambda m: m
    fake.fx = fx
    return fake, fx


def _named(kind, **attrs):
    return type(kind, (_Leaf,), {})(**attrs)


class FromTorchTests(unittest.TestCase):
    def convert(self, nodes, parameters=None, buffers=None, modules=None, inputs=((1, 3),), optimize=True):
        fake, fx = _fake_torch()
        gm = fx.GraphModule(nodes, parameters or {}, buffers or {}, modules or {})
        with mock.patch.dict(sys.modules, {"torch": fake, "torch.fx": fx}):
            return fir.from_torch(gm, tuple(fir.TensorSpec(s) for s in inputs), config={}, optimize=optimize)

    def test_modules_aten_functions_methods_and_buffers(self):
        import operator
        x = _Node("placeholder", "x")
        fc = _Node("call_module", "fc", (x,))
        norm = _Node("call_module", "norm", (fc,))
        act = _Node("call_module", "act", (norm,))
        inv = _Node("get_attr", "inv")
        scaled = _Node("call_function", operator.mul, (act, inv))
        soft = _Node("call_function", _Aten("aten._softmax.default"), (scaled, -1, False))
        floor = _Node("call_function", _Aten("aten.div.Tensor_mode"), (soft, 0.25), {"rounding_mode": None})
        view = _Node("call_method", "view", (floor, -1))
        out = _Node("output", "output", (view,))
        parameters = {"fc.weight": _Tensor((2, 3)), "fc.bias": _Tensor((2,)), "norm.weight": _Tensor((2,))}
        modules = {"fc": _named("Linear", in_features=3, out_features=2, bias=parameters["fc.bias"],
                                weight=parameters["fc.weight"]),
                   "norm": _named("RMSNorm", normalized_shape=(2,), eps=1e-6, weight=parameters["norm.weight"]),
                   "act": _named("SiLU")}
        graph = self.convert([x, fc, norm, act, inv, scaled, soft, floor, view, out], parameters,
                             {"inv": _Tensor((2,), [1.0, 2.0])}, modules)
        ops = Counter(n.op for n in graph.nodes)
        self.assertIn("linear_t", ops)
        self.assertTrue(ops["rmsnorm"] or ops["norm"], ops)
        self.assertNotIn("rsqrt", ops)
        params = {"fc.weight": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0], "fc.bias": [0.0, 1.0], "norm.weight": [1.0, 0.5]}
        got = fir.evaluate(graph, {"input0": [1.0, 2.0, 3.0]}, params)
        h = [4.0, 3.0]
        r = math.sqrt((16 + 9) / 2 + 1e-6)
        a = [v / r * g for v, g in zip(h, (1.0, 0.5))]
        s = [v / (1 + math.exp(-v)) * m for v, m in zip(a, (1.0, 2.0))]
        e = [math.exp(v - max(s)) for v in s]
        close(self, got, [v / sum(e) / 0.25 for v in e], 1e-5)

    def test_rounding_division_and_conv1d(self):
        x = _Node("placeholder", "x")
        conv = _Node("call_module", "conv", (x,))
        out = _Node("output", "output", (conv,))
        parameters = {"conv.weight": _Tensor((1, 1, 2))}
        modules = {"conv": _named("Conv1d", stride=(1,), padding=(1,), dilation=(1,), groups=1, bias=None,
                                  padding_mode="zeros", weight=parameters["conv.weight"])}
        graph = self.convert([x, conv, out], parameters, {}, modules, inputs=((1, 1, 3),))
        got = fir.evaluate(graph, {"input0": [1.0, 2.0, 3.0]}, {"conv.weight": [1.0, 10.0]})
        close(self, got, [10.0, 21.0, 32.0, 3.0])
        y = _Node("placeholder", "y")
        div = _Node("call_function", _Aten("aten.div.Tensor_mode"), (y, 2.0), {"rounding_mode": "floor"})
        with self.assertRaises(fir.Unsupported):
            self.convert([y, div, _Node("output", "output", (div,))])

    def test_unknown_targets_fail_by_name(self):
        x = _Node("placeholder", "x")
        bad = _Node("call_function", _Aten("aten.unknown_op.default"), (x,))
        with self.assertRaisesRegex(ValueError, "aten.unknown_op"):
            self.convert([x, bad, _Node("output", "output", (bad,))])
        mod = _Node("call_module", "lstm", (x,))
        with self.assertRaisesRegex(ValueError, "lstm"):
            self.convert([x, mod, _Node("output", "output", (mod,))], modules={"lstm": _named("LSTM")})


# ---- coverage document --------------------------------------------------------------------------------------

class CoverageDocumentTests(unittest.TestCase):
    def test_coverage_document_matches_the_table(self):
        path = ROOT / "docs" / "fir-script-coverage.md"
        text = path.read_text(encoding="utf-8")
        self.assertIn(coverage_table.markdown_summary(), text)
        for surface in coverage_table.surfaces():
            self.assertIn(coverage_table.markdown_table(surface), text)


if __name__ == "__main__":
    unittest.main()
