"""FIR text/graph checks against docs/fir.md and docs/fir-script.md.

Emitted files are checked for grammar, assembled by ``formers-cli --fir-check``
when the CLI is built, and executed by the Python VM on random synthetic
weights against the reference graph evaluator. No checkpoint is read.
"""

import json
import importlib.util
import math
import random
import subprocess
import tempfile
import unittest
from pathlib import Path

from formers import fir
from formers.fir.example import CONFIG, INPUTS, TinyBlock
from formers.fir.vm import SIGNATURES


CLI = Path(__file__).resolve().parents[2] / "formers-cli"
TOLERANCE = 1e-4


def fir_check(test, path):
    if not CLI.is_file():
        return
    result = subprocess.run([str(CLI), "--fir-check", str(path)], capture_output=True, text=True, timeout=30)
    test.assertEqual(result.returncode, 0, result.stdout + result.stderr)
    test.assertIn("all micro/builtin programs built", result.stdout)


def run_against_reference(test, graph, steps=3, seed=5):
    """Export, then execute the FIR file in the Python VM and compare with fir.evaluate."""
    params = fir.random_parameters(graph, seed=seed, scale=0.5)
    params.update({k + ".fir_flat": v for k, v in list(params.items())})
    with tempfile.TemporaryDirectory() as directory:
        path = fir.export(graph, directory)
        fir_check(test, path)
        model = fir.FirModel(path, weights=params)
    rng = random.Random(seed)
    states = {}
    for position in range(min(steps, graph.config["n_positions"])):
        token = rng.randrange(graph.config["vocab_size"])
        want = fir.evaluate(graph, {"token": [token], "position": [position]}, params, states)
        got = model.step(token)
        test.assertEqual(len(got), len(want))
        for a, b in zip(got, want):
            test.assertLessEqual(abs(a - b), TOLERANCE * max(1.0, abs(b)), (graph.name, position))
    return model


class FirScriptTests(unittest.TestCase):
    def trace(self, model=None, optimize=True):
        return fir.script(model or TinyBlock(), INPUTS, config=CONFIG,
                          name="tiny_gpt2_block", optimize=optimize)

    def assert_grammar(self, text):
        """Check declarations, ports, arities and VM signatures of the emitted subset."""
        buffers, kernels, nodes, scopes = {}, {}, [], []
        kernel = None
        for line in text.splitlines():
            words = line.split()
            if not words or words[0].startswith("#"):
                continue
            if words[0] == "endkernel":
                self.assertIsNotNone(kernel)
                self.assertEqual(scopes, [])
                kernel = None
            elif kernel:
                sig = SIGNATURES[words[0]]
                self.assertEqual(len(words) - 1, len(sig), line)
                for kind, arg in zip(sig, words[1:]):
                    if kind == "r":
                        self.assertRegex(arg, r"^\d+$")
                        self.assertLessEqual(int(arg), 31)
                if words[0] in ("forbegin", "ifbegin"):
                    scopes.append(words[0])
                elif words[0] in ("forend", "ifend"):
                    self.assertEqual(scopes.pop(), words[0].replace("end", "begin"))
                    self.assertLessEqual(scopes.count("forbegin"), 8)
            elif words[0] == "kernel":
                kernel = words[1]
                self.assertRegex(kernel, r"^script\.\w+$")
                self.assertNotIn(kernel, kernels)
                options = dict(item.split("=", 1) for item in words[2:])
                self.assertEqual(set(options), {"in", "out", "threads", "tg", "tgbytes"})
                self.assertEqual((options["tg"], options["tgbytes"]), ("1", "16"))
                kernels[kernel] = options
            elif words[0] == "buf":
                self.assertRegex(line, r"^buf \w+ f32 [1-9]\d*( per_layer)?( (input:(token|position)|output:logits))?$")
                self.assertNotIn(words[1], buffers)
                buffers[words[1]] = (int(words[3]), None)
            elif words[0] == "weight":
                self.assertRegex(line, r"^weight \w+ (f32|q4) [\w.{}]+ tensor \[[1-9]\d*(,[1-9]\d*)*\]$")
                self.assertNotIn(words[1], buffers)
                dims = tuple(map(int, words[5][1:-1].split(",")))
                buffers[words[1]] = (None, dims)
            elif words[0] == "node":
                self.assertRegex(line, r"^node \w+ (micro|ext) [\w.]+( \w+)+ ->( \w+)+$")
                nodes.append(words)
            else:
                self.assertRegex(line, r"^(fir 1|model \w+|config config.json|runtime graph|layout script|"
                                 r"epsilon \w+|prefix \S+|require config \w+ \S+|sym \w+ = \w+|"
                                 r"require sym \w+ = \d+|repeat layers|endrepeat)$")
        self.assertIsNone(kernel)
        for node in nodes:
            arrow = node.index("->")
            ins, outs = node[4:arrow], node[arrow + 1:]
            self.assertTrue(set(ins + outs) <= set(buffers), node)
            if node[2] == "micro":
                options = kernels[node[3]]
                self.assertEqual((len(ins), len(outs)), (int(options["in"]), int(options["out"])))
                self.assertTrue(all(buffers[o][1] is None for o in outs), node)
            else:
                self.assertIn(node[3], ("fir.embedding", "fir.linear", "fir.linear_t"))
                self.assertEqual((len(ins), len(outs)), (2, 1))
                size_in, dims = buffers[ins[0]][0], buffers[ins[1]][1]
                size_out = buffers[outs[0]][0]
                if node[3] == "fir.embedding":
                    self.assertEqual(size_in, 1)
                    self.assertEqual(dims[1], size_out)
                else:
                    self.assertEqual(dims, (size_in, size_out) if node[3] == "fir.linear"
                                     else (size_out, size_in))

    def test_tiny_block_emits_documented_graph_and_kernel_blocks(self):
        graph = self.trace()
        text = fir.emit(graph)
        self.assertTrue(text.startswith("fir 1\nmodel tiny_gpt2_block\nconfig config.json\nruntime graph\nlayout script\n"))
        self.assertIn("epsilon layer_norm_epsilon\n", text)
        self.assertIn("sym layers = 1\nsym positions = n_positions\nsym vocab = vocab_size\n", text)
        self.assertIn("require sym vocab_size = 16\n", text)
        self.assertIn("weight wte_weight f32 wte.weight tensor [16,4]", text)
        # The embedding sum is folded into the LayerNorm, which also stores the residual.
        self.assertIn("kernel script.add_layernorm in=4 out=2 threads=1", text)
        self.assertIn("kernel script.add_gelu_new in=2 out=1 threads=out0.size", text)
        self.assertIn("mulconst 2 2 2*sqrt(2/pi)", text)
        self.assertIn("node norm_norm micro script.add_layernorm tokens_embed positions_embed ln_1_weight "
                      "ln_1_bias -> norm_norm norm_residual\n", text)
        self.assertEqual(text.count("output:logits"), 1)
        self.assertNotIn("builtin", text)
        self.assert_grammar(text)
        run_against_reference(self, graph)

    def test_fusion_removes_intermediates(self):
        raw, optimized = self.trace(optimize=False), self.trace()
        self.assertLess(len(optimized.nodes), len(raw.nodes))
        fused = next(node for node in optimized.nodes if node.op == "fused")
        self.assertEqual(fused.attrs["tree"][0], "gelu_new")
        gelu = next(node for node in raw.nodes if node.op == "gelu_new")
        self.assertEqual(fused.output, gelu.output)
        add = next(node for node in raw.nodes if node.output == gelu.inputs[0])
        self.assertNotIn(add.output, optimized.values)
        self.assert_grammar(fir.emit(raw))
        run_against_reference(self, raw)

    def test_dead_operations_and_their_weights_are_removed(self):
        class WithDeadBranch(TinyBlock):
            def forward(self, token, position):
                fir.Linear(4, 4, "unused")(self.tokens(token))
                return super().forward(token, position)
        raw = self.trace(WithDeadBranch(), optimize=False)
        graph = self.trace(WithDeadBranch())
        self.assertGreater(len(raw.nodes), len(graph.nodes))
        self.assertNotIn("unused", fir.emit(graph))
        self.assert_grammar(fir.emit(graph))

    def test_shared_add_is_kept_as_its_own_buffer(self):
        class Shared(TinyBlock):
            def forward(self, token, position):
                x = self.tokens(token) + self.positions(position)
                return self.tokens.project(x + x.gelu())
        graph = self.trace(Shared())
        text = fir.emit(graph)
        self.assert_grammar(text)
        run_against_reference(self, graph)

    def test_graph_output_is_a_fusion_boundary(self):
        def forward(token, position):
            x = fir.Embedding(16, 16, "table")(token)
            y = x + x
            y.gelu()
            return y
        graph = self.trace(forward, optimize=False)
        output = graph.output
        fir.fuse_pointwise(graph)
        self.assertIn(output, graph.values)
        self.assertEqual(graph.nodes[-1].output, output)

    def test_weight_layout_and_q4_selection(self):
        for layout, dtype, expected in (("io", "q4", "linear"), ("oi", "f32", "linear_t")):
            with self.subTest(layout=layout, dtype=dtype):
                model = TinyBlock()
                model.fc = fir.Linear(4, 16, "fc", layout=layout, dtype=dtype)
                graph = self.trace(model)
                weight_name, value = next((k, v) for k, v in graph.values.items()
                                          if v.weight and v.weight.name == "fc.weight")
                self.assertEqual(value.weight.dtype, dtype)
                self.assertTrue(any(n.op == expected and n.inputs[1] == weight_name
                                    for n in graph.nodes))
                text = fir.emit(graph)
                self.assertIn(f"ext fir.{expected} ", text)
                self.assert_grammar(text)
        with self.assertRaises(ValueError):
            fir.Linear(4, 16, "fc", layout="oi", dtype="q4")

    def test_rejects_invalid_trace_contracts(self):
        with self.assertRaisesRegex(TypeError, "control flow"):
            self.trace(lambda token, position: token if token else position)
        with self.assertRaisesRegex(TypeError, "comparisons"):
            self.trace(lambda token, position: token if token == position else position)
        with self.assertRaisesRegex(ValueError, "equal element counts"):
            self.trace(lambda token, position: fir.Embedding(16, 4, "a")(token)
                       + fir.Embedding(8, 8, "b")(position))
        with self.assertRaisesRegex(ValueError, "logits"):
            fir.emit(self.trace(lambda token, position: fir.Embedding(16, 4, "a")(token)))
        with self.assertRaisesRegex(ValueError, "conflicting descriptor"):
            self.trace(lambda token, position: fir.Embedding(16, 4, "same")(token)
                       + fir.Embedding(8, 4, "same")(position))
        with self.assertRaisesRegex(ValueError, "rows"):
            self.trace(lambda token, position: fir.Embedding(15, 16, "a")(token))
        with self.assertRaises(ValueError):
            fir.emit(fir.trace(lambda t, p: t, (fir.TensorSpec(2, "token"), fir.TensorSpec(1, "position")), config=CONFIG))
        for eps in (0, -1, float("nan"), float("inf")):
            with self.subTest(eps=eps), self.assertRaises(ValueError):
                fir.script(TinyBlock(), INPUTS, config=dict(CONFIG, layer_norm_epsilon=eps))
        with self.assertRaisesRegex(ValueError, "unoptimized graph"):
            fir.export(self.trace(), tempfile.gettempdir() + "/fir-never-written", optimize=False)

    def test_export_is_deterministic_and_preserves_graph(self):
        graph = self.trace()
        before = repr(graph)
        with tempfile.TemporaryDirectory() as directory:
            path = fir.export(graph, directory)
            self.assertEqual(path.read_text(), fir.emit(self.trace()))
            self.assertEqual(json.loads(path.with_name("config.json").read_text()), CONFIG)
            self.assertEqual({p.name for p in Path(directory).iterdir()}, {"model.fir", "config.json"})
        self.assertEqual(repr(graph), before)

    def test_export_traces_modules_and_optimizes_raw_graphs(self):
        with tempfile.TemporaryDirectory() as directory:
            direct = fir.export(TinyBlock(), Path(directory) / "a", example_inputs=INPUTS, config=CONFIG,
                                name="tiny_gpt2_block").read_text()
            raw = fir.export(self.trace(optimize=False), Path(directory) / "b", optimize=True).read_text()
            unoptimized = fir.export(self.trace(optimize=False), Path(directory) / "c", optimize=False).read_text()
        self.assertEqual(direct, fir.emit(self.trace()))
        self.assertEqual(raw, direct)
        self.assertNotEqual(unoptimized, direct)

    def test_prefix_and_config_requirements(self):
        graph = self.trace()
        text = fir.emit(graph, prefix="mlp.", require_config={"model_type": "gpt2", "tie": True})
        self.assertIn("prefix mlp.\n", text)
        self.assertIn("require config model_type gpt2\nrequire config tie true\n", text)
        self.assertIn(" c_fc.weight tensor", text)
        self.assertNotIn(" mlp.c_fc.weight", text)
        with self.assertRaisesRegex(fir.LoweringError, "absent from config"):
            fir.emit(graph, require_config=["missing_key"])

    def test_report_counts_kernels_nodes_and_instructions(self):
        graph = self.trace()
        text = fir.emit(graph)
        stats = fir.fir_stats(text, graph.config)
        self.assertEqual(stats["nodes"], sum(line.startswith("node ") for line in text.splitlines()))
        self.assertEqual(set(stats["kernels"]), {line.split()[1] for line in text.splitlines() if line.startswith("kernel ")})
        self.assertGreater(stats["kernel_instructions"], 0)
        self.assertIn("kernels", fir.report(text, graph))

    @unittest.skipUnless(CLI.is_file(), "formers-cli is not built")
    def test_cli_builds_programs_without_weights(self):
        for optimized in (False, True):
            with self.subTest(optimized=optimized), tempfile.TemporaryDirectory() as directory:
                fir_check(self, fir.export(self.trace(optimize=optimized), directory))


class GeneralFirTests(unittest.TestCase):
    """Synthetic operators: reference evaluation, assembly and Python VM execution."""

    def fixture(self, forward, shapes, data, expected, *, config=None):
        specs = tuple(fir.TensorSpec(s) for s in shapes)
        raw = fir.trace(forward, specs, config=config, optimize=False)
        optimized = fir.trace(forward, specs, config=config)
        inputs = {"input"+str(i): x for i, x in enumerate(data)}
        self.assertEqual(fir.compare_graphs(raw, optimized, [inputs]), 0)
        actual = fir.evaluate(optimized, inputs)
        self.assertEqual(len(actual), len(expected))
        for x, y in zip(actual, expected): self.assertAlmostEqual(x, y, places=6)
        return raw, optimized

    def check(self, graph):
        before = repr(graph)
        text = fir.emit(graph)
        self.assertEqual(repr(graph), before)
        weights = {}
        for line in text.splitlines():
            words = line.split()
            if words and words[0] == "weight":
                dims = tuple(map(int, words[-1][1:-1].split(",")))
                self.assertGreaterEqual(len(dims), 1)
                weights[words[1]] = dims
        for line in text.splitlines():
            words = line.split()
            if len(words) > 3 and words[:1] == ["node"] and words[2] == "micro":
                for k in words[4:words.index("->")]:
                    if k in weights: self.assertNotEqual(len(weights[k]), 2, line)
        self.assertEqual(fir.emit(graph), text)  # deterministic
        run_against_reference(self, graph)
        return text

    def decode(self, model, size, optimize=True):
        return fir.trace(model, INPUTS, config=dict(CONFIG, vocab_size=size), optimize=optimize)

    def test_higher_rank_weights_keep_checkpoint_name_and_logical_shape(self):
        class Weights(fir.Module):
            def __init__(self):
                super().__init__()
                self.qkv = fir.Parameter((3, 2, 2))
            def forward(self, token, position):
                q, k, v = self.qkv.permute(1, 0, 2).chunk(3, 1)
                return fir.cat((v, k, q), 1).reshape(12) + position
        for optimize in (False, True):
            graph = self.decode(Weights(), 12, optimize)
            weight = next(v for v in graph.values.values() if v.weight)
            self.assertEqual(weight.shape, (3, 2, 2))
            text = self.check(graph)
            self.assertIn("layout script\n", text)
            self.assertIn("weight qkv f32 qkv tensor [3,2,2]", text)
            self.assertNotIn(".fir_flat", text)
            self.assertNotIn("checkpoint-map", text)
            self.assertNotIn("offline reshape", text)

    def test_script_layout_preserves_rank1_and_rank4_checkpoint_descriptors(self):
        class Weights(fir.Module):
            def __init__(self):
                super().__init__()
                self.volume = fir.Parameter((1, 2, 1, 3), name="source")
                self.bias = fir.Parameter((6,), name="source.fir_flat")
            def forward(self, token, position):
                return self.volume.reshape(6) + self.bias + position
        text = self.check(self.decode(Weights(), 6))
        self.assertIn("weight source f32 source tensor [1,2,1,3]", text)
        self.assertIn("weight source_fir_flat f32 source.fir_flat tensor [6]", text)
        self.assertNotIn("checkpoint-map", text)
        self.assertNotIn("offline reshape", text)

    def test_tied_matrix_has_separate_resident_and_arena_descriptors(self):
        class Shared(fir.Module):
            def __init__(self):
                super().__init__()
                self.table = fir.Embedding(8, 4)
            def forward(self, token, position):
                return self.table.project(self.table(token) + self.table.weight[0])
        text = self.check(self.decode(Shared(), 8))
        self.assertIn("table.weight tensor [8,4]", text)
        self.assertIn("table.weight.fir_flat tensor [32]", text)
        self.assertIn("fir.embedding token table_weight ->", text)
        self.assertIn("# checkpoint-map table.weight [8,4]", text)

    def test_matrix_micro_shortcuts_also_use_vector_bindings(self):
        class Matrix(fir.Module):
            def __init__(self):
                super().__init__()
                self.weight = fir.Parameter((2, 2))
            def forward(self, token, position):
                return (self.weight + fir.ones((2, 2))).reshape(4)
        text = self.check(self.decode(Matrix(), 4))
        self.assertIn("weight.fir_flat tensor [4]", text)
        self.assertNotIn("weight tensor [2,2]", text)

    def test_batched_linear_uses_vector_weight_and_scalar_plan(self):
        class Rows(fir.Module):
            def __init__(self):
                super().__init__()
                self.fc = fir.Linear(2, 2, bias=False)
            def forward(self, token, position):
                return self.fc(token + fir.ones((2, 2))).reshape(4)
        text = self.check(self.decode(Rows(), 4))
        self.assertIn("fc.weight.fir_flat tensor [4]", text)
        self.assertNotIn(" ext ", text)

    def test_repeat_falls_back_when_later_checkpoint_escapes(self):
        class Block(fir.Module):
            def __init__(self):
                super().__init__()
                self.bias = fir.Parameter((2,))
            def forward(self, x): return x + self.bias
        class Stack(fir.Module):
            def __init__(self):
                super().__init__()
                self.blocks = fir.ModuleList((Block(), Block()))
            def forward(self, t, p):
                x = t + fir.ones((2,))
                for block in self.blocks: x = block(x)
                return x + self.blocks[1].bias
        text = self.check(self.decode(Stack(), 2))
        self.assertNotIn("repeat layers", text)
        self.assertIn("blocks.1.bias tensor [2]", text)

    def test_repeated_higher_rank_weights_keep_layer_checkpoint_mapping(self):
        class Stack(fir.Module):
            def __init__(self):
                super().__init__()
                self.blocks = fir.ModuleList(fir.Conv1d(2, 2, 2, groups=2) for _ in range(2))
            def forward(self, t, p):
                x = t + fir.ones((1, 2, 2))
                for block in self.blocks: x = block(x)
                return x.reshape(4)
        text = self.check(self.decode(Stack(), 4))
        self.assertIn("repeat layers", text)
        self.assertIn("blocks.{i}.weight tensor [2,1,2]", text)
        self.assertNotIn(".fir_flat", text)
        self.assertNotIn("checkpoint-map", text)

    def test_repeat_preserves_residual_from_before_stack(self):
        class Stack(fir.Module):
            def __init__(self):
                super().__init__()
                self.blocks = fir.ModuleList(fir.Linear(2, 2) for _ in range(2))
            def forward(self, t, p):
                residual = t + fir.ones((2,))
                x = residual
                for block in self.blocks: x = block(x)
                return residual + x
        text = self.check(self.decode(Stack(), 2))
        self.assertNotIn("repeat layers", text)

    def test_scan_cumsum_and_transcendental_kernels_assemble(self):
        def forward(t, p):
            x = t + fir.tensor([1, 2])
            scanned, final = fir.scan(lambda s, y: s+y, x, fir.tensor(0))
            y = (scanned.cumsum(0) + final).sqrt().log().exp().tanh().sigmoid()
            return fir.where(x != p, y, x.rsqrt())
        self.check(self.decode(forward, 2))

    def test_activation_variants_and_standalone_rotary_run_in_vm(self):
        activations = {
            "gelu_erf": lambda x: x.gelu("none"),
            "gelu_tanh": lambda x: x.gelu(),
            "gelu_quick": lambda x: x.gelu("quick"),
            "softplus": lambda x: x.softplus(),
            "sin_cos": lambda x: x.sin() * (x * 3).cos(),
            "abs_max_min": lambda x: x.abs().maximum(x * 0.5).minimum(fir.ones((4,))),
            "silu_relu_tanh": lambda x: x.silu() + x.relu() - x.tanh(),
        }
        for name, fn in activations.items():
            with self.subTest(activation=name):
                self.check(self.decode(lambda t, p, fn=fn: fn((t - 3) * fir.tensor([0.5, -1.25, 2.0, 0.1])), 4))
        rotary = lambda t, p: fir.rotary_embedding((t + fir.arange(8)).reshape(1, 2, 4) * 0.25, p).reshape(8)
        text = self.check(self.decode(rotary, 8))
        self.assertIn("kernel script.rotary ", text)

    def test_checkpoint_mapping_collision_and_direct_matrix_lowering_rejected(self):
        class Collision(fir.Module):
            def __init__(self):
                super().__init__()
                self.first = fir.Parameter((2, 2), name="tensor")
                self.second = fir.Parameter((4,), name="tensor.fir_flat")
            def forward(self, t, p): return self.first.reshape(4) + self.second
        with self.assertRaisesRegex(fir.LoweringError, "conflicts with a source parameter"):
            fir.emit(self.decode(Collision(), 4))
        # A matrix read by a micro-kernel is bound through its flattened checkpoint copy.
        text = self.check(self.decode(lambda t, p: (t + fir.ones((2,))) @ fir.Parameter((2, 2), name="matrix"), 2))
        self.assertIn("weight matrix_flat f32 matrix.fir_flat tensor [4]", text)

    def test_symbol_bindings_guard_config_and_reserved_runtime_symbols(self):
        with self.assertRaisesRegex(ValueError, "conflicts with config"):
            fir.trace(lambda x: x, (fir.TensorSpec(fir.Dim("width", 2)),), config={"width": 3})
        graph = self.decode(lambda t, p: t+fir.ones((2,)), 2)
        graph.symbols["layers"] = 3
        with self.assertRaisesRegex(fir.LoweringError, "conflicts with runtime"):
            fir.emit(graph)

    def test_conv_grouped_and_depthwise_scalar_offsets(self):
        for groups, expected in ((1, [5, 12, 16, 5, 12, 16]), (2, [1, 4, 7, 4, 13, 16])):
            conv = fir.Conv1d(2, 2, 2, groups=groups, bias=False)
            raw = fir.trace(conv, (fir.TensorSpec((1, 2, 3)),), optimize=False)
            parameters = {"weight": [1, 1] * (4//groups)}
            if groups == 2: parameters["weight"] = [2, 1, 2, 1]
            actual = fir.evaluate(raw, {"input0": [1, 2, 3, 4, 5, 6]}, parameters)
            self.assertEqual(actual, expected)
            class DecodeConv(fir.Module):
                def __init__(self):
                    super().__init__()
                    self.conv = fir.Conv1d(2, 2, 2, groups=groups)
                def forward(self, token, position):
                    return self.conv(token + fir.ones((1, 2, 3))).reshape(6)
            self.assertIn(f"conv.weight tensor [2,{2//groups},2]", self.check(self.decode(DecodeConv(), 6)))

    def test_broadcast_and_long_fusion_chain(self):
        raw, optimized = self.fixture(lambda x, y: ((x+y)*2-3).relu(),
                                     ((2, 2), (2,)), ([1, 2, -3, -4], [2, 3]), [3, 7, 0, 0])
        self.assertLess(len(optimized.nodes), len(raw.nodes))
        available = {k for k, v in optimized.values.items() if v.role or v.constant is not None}
        for node in optimized.nodes:
            self.assertTrue(set(node.inputs) <= available)
            available.add(node.output)
        self.assertEqual(optimized.passes, [fn.__name__ for fn in fir.PASSES])

    def test_matmul_vector_matrix_and_batched_shapes(self):
        fixtures = [(((2,), (2,)), ([2, 3], [4, 5]), [23]),
                    (((2,), (2, 2)), ([2, 3], [1, 2, 3, 4]), [11, 16]),
                    (((2, 2), (2,)), ([1, 2, 3, 4], [2, 3]), [8, 18]),
                    (((2, 1, 2), (1, 2, 1)), ([1, 2, 3, 4], [2, 3]), [8, 18])]
        for shapes, data, expected in fixtures:
            with self.subTest(shapes=shapes): self.fixture(fir.matmul, shapes, data, expected)
        self.check(self.decode(lambda t, p: (t + fir.ones((2, 2))) @ fir.tensor([1, 2]), 2))

    def test_views_slicing_split_chunk_cat_and_inverse_permutations(self):
        def forward(x):
            y = x.permute(1, 0).permute(1, 0).reshape(2, 3)
            a, b = y.split((1, 2), 1)
            return fir.cat((b[:, ::-1], a), 1)
        raw, optimized = self.fixture(forward, ((2, 3),), ([1, 2, 3, 4, 5, 6],), [3, 2, 1, 6, 5, 4])
        self.assertLess(len(optimized.nodes), len(raw.nodes))

    def test_constant_fold_cse_and_kernel_dedup(self):
        def forward(t, p):
            c = (fir.tensor([1, 2, 3, 4]) * 2).reshape(4)
            a, b = t+c, t+c
            return (a+b) + (p+c)
        raw, optimized = self.decode(forward, 4, False), self.decode(forward, 4)
        self.assertLess(len(optimized.nodes), len(raw.nodes))
        self.assertTrue(any(v.constant == (2., 4., 6., 8.) for v in optimized.values.values()))
        self.assertEqual(fir.compare_graphs(raw, optimized, [{"token": [2], "position": [1]}]), 0)
        self.check(optimized)
        dedup = self.decode(lambda t, p: fir.cat(((t+fir.ones((2,))).relu(),
                                                (p+fir.ones((2,))).relu())), 4)
        text = self.check(dedup)
        calls = [line.split()[3] for line in text.splitlines() if line.startswith("node ") and " micro " in line]
        self.assertLess(len(set(calls)), len(calls))

    def test_softmax_norms_activations_and_masking(self):
        self.fixture(lambda x: x.softmax(-1), ((2,),), ([1000, 1000],), [0.5, 0.5])
        self.fixture(lambda x: fir.where(x > 0, x.silu(), x.relu()), ((2,),), ([-2, 0],), [0, 0])
        for norm in (fir.LayerNorm(2, elementwise_affine=False),):
            self.fixture(norm, ((1, 2),), ([1, 3],), [-1/math.sqrt(1+1e-5), 1/math.sqrt(1+1e-5)])
        norm = fir.RMSNorm(2)
        graph = fir.trace(norm, (fir.TensorSpec((2,)),))
        expected = [1/math.sqrt(5+1e-6), 3/math.sqrt(5+1e-6)]
        for actual, want in zip(fir.evaluate(graph, {"input0": [1, 3]}, {"weight": [1, 1]}), expected):
            self.assertAlmostEqual(actual, want)

    def test_rope_gqa_and_fully_masked_rows(self):
        self.fixture(lambda x, p: fir.rotary_embedding(x, p), ((1, 1, 2), (1,)),
                     ([1, 2], [math.pi/2]), [-2, 1])
        def attention(q, k, v):
            return fir.scaled_dot_product_attention(q, k, v, enable_gqa=True, is_causal=True)
        self.fixture(attention, ((2, 2, 2), (1, 2, 2), (1, 2, 1)),
                     ([0]*8, [0]*4, [2, 4]), [2, 3, 2, 3])
        self.fixture(lambda q, k, v: fir.scaled_dot_product_attention(q, k, v,
                     attn_mask=fir.tensor([[False, False]], "bool")),
                     ((1, 1, 2), (1, 2, 2), (1, 2, 1)), ([0, 0], [0]*4, [2, 4]), [0])

    def test_cumsum_scan_and_state_ordering_and_reset(self):
        self.fixture(lambda x: x.cumsum(0), ((3,),), ([1, 2, 3],), [1, 3, 6])
        self.fixture(lambda x: fir.scan(lambda s, y: s+y, x, fir.tensor(0))[0],
                     ((3,),), ([1, 2, 3],), [1, 3, 6])
        class Recurrent(fir.Module):
            def __init__(self):
                super().__init__()
                self.register_state("carry", (fir.Dim("width", 2),))
            def forward(self, token, position):
                old = self.carry.read()
                new = self.carry.update(old + token)
                again = self.carry.read()
                self.carry.update(again + 1)
                return new + old
        raw, optimized = self.decode(Recurrent(), 2, False), self.decode(Recurrent(), 2)
        steps = [{"token": [x], "position": [p]} for x, p in ((2, 0), (3, 1), (4, 0))]
        self.assertEqual(fir.compare_graphs(raw, optimized, steps), 0)
        states = {}
        self.assertEqual(fir.evaluate(raw, steps[0], states=states), [2, 2])
        self.assertEqual(fir.evaluate(raw, steps[1], states=states), [9, 9])
        self.assertEqual(fir.evaluate(raw, steps[2], states=states), [4, 4])
        self.check(optimized)

    def test_symbolic_indexed_state_and_effect_roots(self):
        class Cache(fir.Module):
            def __init__(self):
                super().__init__()
                self.register_state("cache", (fir.Dim("slots", 4), 2))
            def forward(self, t, p):
                self.cache.update(t + fir.ones((2,)), p, dim=0)
                return fir.ones((2,)) + t
        graph = self.decode(Cache(), 2)
        self.assertTrue(any(n.effectful for n in graph.nodes))
        self.assertEqual(graph.symbols["slots"], 4)
        self.check(graph)

    def test_examples_export_raw_and_optimized_with_repeat_and_per_layer_cache(self):
        from formers.fir.examples import gpt2, llama
        for source, model in ((gpt2, gpt2.GPT2), (llama, llama.Llama)):
            for optimize in (False, True):
                with self.subTest(model=model.__name__, optimize=optimize):
                    graph = fir.trace(model(), source.INPUTS, config=source.CONFIG, optimize=optimize)
                    text = self.check(graph)
                    self.assertRegex(text, r"sym layers = (n_layer|num_hidden_layers)\n")
                    self.assertRegex(text, r"require sym (n_layer|num_hidden_layers) = 2\n")
                    self.assertEqual(text.count("repeat layers\n"), 1)
                    self.assertIn(" per_layer", text)
                    self.assertIn(".{i}.", text)
                    if optimize: self.assertIn("q4_candidates", graph.hints)

    def test_clear_errors_for_unrepresentable_exports(self):
        class Q4Vector(fir.Module):
            def __init__(self):
                super().__init__()
                self.weight = fir.Parameter((2, 2, 2), dtype="q4")
            def forward(self, t, p): return self.weight.reshape(8) + t
        with self.assertRaisesRegex(fir.LoweringError, "q4 is supported only by fir.linear"):
            fir.emit(self.decode(Q4Vector(), 8))
        graph = self.decode(lambda t, p: (t+fir.ones((2,))) ** 1.5, 2)
        with self.assertRaisesRegex(fir.LoweringError, "constant integer exponent"):
            fir.emit(graph)
        for invalid in (0, -1, True, 1.5):
            graph = self.decode(lambda t, p: t+fir.ones((2,)), 2)
            graph.config["n_positions"] = invalid
            with self.assertRaisesRegex(fir.LoweringError, "positive context length"):
                fir.emit(graph)
        if importlib.util.find_spec("torch") is None:
            with self.assertRaisesRegex(ImportError, "optional torch"):
                fir.from_torch(None, ())


if __name__ == "__main__":
    unittest.main()
