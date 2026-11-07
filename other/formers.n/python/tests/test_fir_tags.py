import json
import re
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from formers.fir import vm
from formers.fir.emitter import KERNEL_TAGS, kernel_tag
from formers.fir.examples import gpt2

CLI = Path(__file__).resolve().parents[2] / "formers-cli"


def export(tags):
    from formers.fir import export, trace
    graph = trace(gpt2.GPT2(), gpt2.INPUTS, config=gpt2.CONFIG, name="tiny_gpt2")
    directory = tempfile.mkdtemp()
    return Path(export(graph, directory, tags=tags)), directory


class KernelTagTests(unittest.TestCase):
    def test_default_export_has_no_tags(self):
        path, directory = export(False)
        self.addCleanup(shutil.rmtree, directory, True)
        self.assertNotIn("tag=", path.read_text())

    def test_every_kernel_header_carries_a_tag(self):
        path, directory = export(True)
        self.addCleanup(shutil.rmtree, directory, True)
        headers = [l for l in path.read_text().splitlines() if l.startswith("kernel ")]
        self.assertTrue(headers)
        for line in headers:
            self.assertRegex(line, r" tag=[A-Za-z0-9_:+.]+$|tag=[A-Za-z0-9_:+.]+ ")
        # tags never appear outside kernel headers
        body = [l for l in path.read_text().splitlines() if not l.startswith("kernel ") and "tag=" in l]
        self.assertEqual(body, [])

    def test_python_vm_parser_keeps_the_tag(self):
        path, directory = export(True)
        self.addCleanup(shutil.rmtree, directory, True)
        program = vm.parse(path.read_text(), json.loads((Path(directory) / "config.json").read_text()))
        tagged = [k for k in program["kernels"].values() if k.get("tag")]
        self.assertEqual(len(tagged), len(program["kernels"]))

    def test_known_and_generic_families(self):
        self.assertEqual(kernel_tag("attention"), "aten::scaled_dot_product_attention")
        self.assertEqual(kernel_tag("silu_mul"), "aten::silu+aten::mul")
        self.assertEqual(kernel_tag("softmax"), "aten::softmax")
        self.assertTrue(all(re.fullmatch(r"[A-Za-z0-9_:+.]+", t) for t in KERNEL_TAGS.values()))

    @unittest.skipUnless(CLI.exists(), "formers-cli is not built")
    def test_runner_accepts_tags_only_with_script_layout(self):
        path, directory = export(True)
        self.addCleanup(shutil.rmtree, directory, True)
        ok = subprocess.run([str(CLI), "--fir-check", str(path)], capture_output=True, text=True)
        self.assertEqual(ok.returncode, 0, ok.stderr + ok.stdout)
        classic = Path(directory) / "classic.fir"
        classic.write_text(path.read_text().replace("layout script", "layout classic"))
        bad = subprocess.run([str(CLI), "--fir-check", str(classic)], capture_output=True, text=True)
        self.assertNotEqual(bad.returncode, 0)
        self.assertIn("tag needs 'layout script'", bad.stderr + bad.stdout)


if __name__ == "__main__":
    unittest.main()
