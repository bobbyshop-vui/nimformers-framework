"""GPT-2 125M trace example: real-size emission and drop-in compatibility with
models/gpt2-125m/model.fir, plus tiny-config numerics (no checkpoint is read)."""

import json
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from formers import fir
from formers.fir import vm
from formers.fir.examples import gpt2_125m
from formers.fir.examples.gpt2_125m import GPT2LMHead, INPUTS, REQUIRE_CONFIG

from tests.test_fir_numerics import GPT2Decoder, SEQUENCES, TOLERANCE, checkpoint, largest_difference


CLI = Path(__file__).resolve().parents[2] / "formers-cli"
HAND = gpt2_125m.HAND_WRITTEN
TINY = {
    "model_type": "gpt2",
    "activation_function": "gelu_new",
    "vocab_size": 16,
    "n_positions": 4,
    "n_embd": 8,
    "n_head": 2,
    "n_layer": 2,
    "layer_norm_epsilon": 1e-5,
}


def declarations(text, config):
    prog = vm.parse(text, config)
    weights = {(b["tensor"], b["dtype"], tuple(b["dims"])) for b in prog["buffers"].values() if b["kind"] == "weight"}
    caches = {name: (b["elements"], b["per_layer"]) for name, b in prog["buffers"].items()
              if name in ("key_cache", "value_cache")}
    return prog, weights, caches


class GPT2RealSizeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        start = time.perf_counter()
        cls.path, cls.graph = gpt2_125m.export_model(Path(cls.directory.name) / "optimized")
        cls.seconds = time.perf_counter() - start
        cls.text = cls.path.read_text()

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_real_size_emission_is_fast_and_deterministic(self):
        self.assertLess(self.seconds, 10)
        again, _ = gpt2_125m.export_model(Path(self.directory.name) / "again")
        self.assertEqual(again.read_text(), self.text)

    @unittest.skipUnless(HAND.is_file(), "hand-written GPT-2 FIR is not present")
    def test_drop_in_layout_matches_hand_written_file(self):
        config = gpt2_125m.CONFIG
        ours, our_weights, our_caches = declarations(self.text, config)
        hand_text = HAND.read_text()
        hand, hand_weights, hand_caches = declarations(hand_text, config)
        self.assertEqual(our_weights, hand_weights)
        self.assertEqual(our_caches, hand_caches)
        self.assertEqual(our_caches["key_cache"], (config["n_positions"] * config["n_embd"], True))
        self.assertEqual(ours["prefixes"], hand["prefixes"])
        self.assertEqual({l for l in self.text.splitlines() if l.startswith("require config")},
                         {l for l in hand_text.splitlines() if l.startswith("require config")})
        for name in ("layers", "positions", "vocab"):
            self.assertEqual(ours["symbols"][name], hand["symbols"][name])
        stats, hand_stats = fir.fir_stats(self.text, config), fir.fir_stats(hand_text, config)
        self.assertLessEqual(len(stats["kernels"]), 6)
        self.assertLess(stats["nodes"], hand_stats["nodes"])
        self.assertLess(stats["dispatches_per_token"]["micro"], hand_stats["dispatches_per_token"]["micro"])
        self.assertEqual(stats["dispatches_per_token"]["ext"], hand_stats["dispatches_per_token"]["ext"])

    def test_report_mode_prints_statistics(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                ["python3", "-m", "formers.fir.examples.gpt2_125m", directory, "--report"],
                capture_output=True, text=True, timeout=60, cwd=Path(__file__).resolve().parents[1])
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("kernels 6", result.stdout)
        self.assertIn("script.attention", result.stdout)

    @unittest.skipUnless(CLI.is_file(), "formers-cli is not built")
    def test_cli_builds_real_size_programs(self):
        with tempfile.TemporaryDirectory() as directory:
            raw, _ = gpt2_125m.export_model(Path(directory) / "raw", optimize=False)
            for path in (self.path, raw):
                result = subprocess.run([str(CLI), "--fir-check", str(path)], capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                self.assertIn("all micro/builtin programs built", result.stdout)


class GPT2TinyConfigTests(unittest.TestCase):
    def test_tiny_config_decodes_like_straight_line_gpt2(self):
        graph = fir.trace(GPT2LMHead(TINY), INPUTS, config=TINY, name="gpt2_tiny")
        params = checkpoint(graph, 23)
        with tempfile.TemporaryDirectory() as directory:
            path = fir.export(graph, directory, prefix="transformer.", require_config=REQUIRE_CONFIG)
            text = path.read_text()
            self.assertIn("prefix transformer.\n", text)
            self.assertIn("weight c_attn_weight q4 h.{i}.attn.c_attn.weight tensor [8,24]", text)
            self.assertIn("buf key_cache f32 32 per_layer", text)
            self.assertEqual(json.loads(path.with_name("config.json").read_text()), TINY)
            model = fir.FirModel(path, weights=params)
        reference = GPT2Decoder(params, TINY)
        for seq in SEQUENCES:
            model.reset()
            reference.reset()
            for position, token in enumerate(seq):
                self.assertLessEqual(largest_difference(model.step(token), reference.step(token, position)), TOLERANCE)


if __name__ == "__main__":
    unittest.main()
