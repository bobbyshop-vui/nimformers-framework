"""Exercise the native process group through the Python compatibility API."""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

from nimformers import Context, NimformersError
from nimformers.distributed import init_process_group


def test_partition_many_devices():
    devices = ["cpu", *(f"vulkan:{i}" for i in range(100))]
    with init_process_group(devices) as group:
        assert group.partition(801) == (1, *((8,) * 100))
        assert group.partition(7, ratios=[1, *([0] * 100)]) == (7, *((0,) * 100))
    with pytest.raises(NimformersError, match="(?i)duplicate|distinct"):
        init_process_group(["vulkan", "vulkan:0"])


@pytest.mark.skipif(os.name != "posix", reason="Native worker transport uses POSIX")
def test_parallel_vm_worker_keeps_input_and_item_order():
    data = np.arange(16_384, dtype=np.float32).reshape(4096, 4)
    with Context("cpu") as ctx, ctx.vm_program() as program:
        program.load_f32(1, 0, 0)
        program.add(1, 1, 1)
        program.store_f32(1, 0, 0)
        program.halt()
        with init_process_group(["cpu", "vulkan:0"], ratios=[1, 0]) as group:
            actual = group.vm_run(program, data, tg_size=8)
            np.testing.assert_array_equal(actual, data * 2)
            np.testing.assert_array_equal(data, np.arange(16_384).reshape(4096, 4))
            with pytest.raises(NimformersError, match="divisible"):
                group.vm_run(program.words, data[:-1], tg_size=8)


@pytest.mark.skipif(os.name != "posix", reason="Native worker transport uses POSIX")
def test_parallel_fir_requests_overlap_and_keep_order(tmp_path: Path, monkeypatch):
    # Fake CLI fixture: the workers must reach the rendezvous concurrently.
    script = tmp_path / "fake_formers.py"
    script.write_text(
        f"#!{sys.executable}\n"
        "import os, sys, time\n"
        "from pathlib import Path\n"
        "args = sys.argv[1:]\n"
        "assert args[args.index('--fir') + 1] == 'model.fir'\n"
        "assert 'TEST_PARALLEL_UNSET' not in os.environ\n"
        "backend = args[args.index('--backend') + 1]\n"
        "if backend == 'vulkan': assert os.environ['BYBY_VULKAN_DEVICE'] == '6'\n"
        "root = Path(os.environ['PARALLEL_TEST_DIR'])\n"
        "(root / backend).touch()\n"
        "deadline = time.monotonic() + 5\n"
        "while not ((root / 'cpu').exists() and (root / 'vulkan').exists()):\n"
        "    if time.monotonic() > deadline: raise RuntimeError('workers did not overlap')\n"
        "    time.sleep(0.01)\n"
        "sys.stderr.write('x' * 100000)\n"
        "print(backend + ':' + args[args.index('-p') + 1])\n",
        encoding="utf-8",
    )
    script.chmod(0o700)
    monkeypatch.setenv("TEST_PARALLEL_UNSET", "inherited")
    with init_process_group(["cpu", "vulkan:6"], ratios=[1, 1]) as group:
        result = group.generate_batch(
            script, ["--fir", "model.fir"], ["{backend}", "two\nlines"],
            env={"PARALLEL_TEST_DIR": tmp_path, "BYBY_VULKAN_DEVICE": "0",
                 "TEST_PARALLEL_UNSET": None},
        )
    assert result == ["cpu:{backend}", "vulkan:two\nlines"]
