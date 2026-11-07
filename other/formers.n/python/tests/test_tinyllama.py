import json
import re
import struct
import tempfile
import unittest
from pathlib import Path

from formers.fir.examples import tinyllama

ROOT = Path(__file__).resolve().parents[4]
SAFETENSORS = ROOT / "models" / "tinyllama" / "model.safetensors"


def checkpoint_names():
    with SAFETENSORS.open("rb") as handle:
        size = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(size))
    return {name: meta["shape"] for name, meta in header.items() if name != "__metadata__"}


class TinyLlamaTraceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.path, cls.graph = tinyllama.export_model(cls.temp.name)
        cls.text = cls.path.read_text(encoding="utf-8")

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_config_is_not_modified(self):
        exported = json.loads((self.path.parent / "config.json").read_text())
        self.assertEqual(exported, json.loads(tinyllama.CONFIG_PATH.read_text()))

    def test_header_uses_the_checkpoint_config_keys(self):
        self.assertIn("epsilon rms_norm_eps", self.text)
        self.assertIn("sym positions = max_position_embeddings", self.text)
        self.assertIn("sym layers = num_hidden_layers", self.text)
        self.assertIn("layout script", self.text)

    def test_layers_are_emitted_once(self):
        self.assertIn("per_layer", self.text)
        self.assertEqual(len(re.findall(r"^kernel script\.attention ", self.text, re.M)), 1)

    def test_weights_match_checkpoint_tensors(self):
        if not SAFETENSORS.exists():
            self.skipTest("checkpoint header not present")
        names = checkpoint_names()
        for match in re.finditer(r"^weight \S+ \S+ (\S+) tensor \[([\d,]+)\]", self.text, re.M):
            name = match.group(1)
            shape = [int(x) for x in match.group(2).split(",")]
            if "{i}" in name:
                name = name.replace("{i}", "0")
            self.assertIn(name, names, name)
            self.assertEqual(names[name], shape, name)


if __name__ == "__main__":
    unittest.main()
