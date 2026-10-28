"""Small tour of the Nimformers Python API.

Run after building the shared library:

    make build-for-python
    python3 examples/python/basic.py
"""
import sys
from pathlib import Path

import numpy as np

# Allow running straight from a source checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nimformers import Context


def main() -> None:
    rng = np.random.default_rng(0)
    with Context() as ctx:
        print("backend:", ctx.backend, "abi:", ctx.abi_version)

        a = rng.standard_normal((4, 8)).astype(np.float32)
        b = rng.standard_normal((8, 16)).astype(np.float32)
        print("matmul:", ctx.matmul(a, b).shape)
        print("softmax sums:", ctx.softmax(a).sum(axis=-1))

        gamma = np.ones(8, dtype=np.float32)
        beta = np.zeros(8, dtype=np.float32)
        print("layernorm:", ctx.layernorm(a, gamma, beta).shape)

        q = rng.standard_normal((1, 2, 4, 4)).astype(np.float32)
        out, scores = ctx.attention(q, q, q)
        print("attention:", out.shape, scores.shape)

        # Quantization + dequantization round trip.
        weight = rng.standard_normal((32, 64)).astype(np.float32)
        q4 = ctx.quantize(weight, "int4_asym")
        print("int4 max error:", float(np.abs(q4.dequantize() - weight).max()))
        x64 = rng.standard_normal((4, 64)).astype(np.float32)
        print("q4 linear:", ctx.linear_q4(x64, q4).shape)
        q4.close()

        # Micro-VM: double every element in a buffer.
        prog = ctx.vm_program()
        prog.load_f32(1, 0, 0)
        prog.add(1, 1, 1)
        prog.store_f32(1, 0, 0)
        prog.halt()
        data = np.arange(1, 65, dtype=np.float32)
        print("vm doubled:", np.array_equal(ctx.vm_run(prog, data, tg_size=64), data))
        prog.close()

        # NIMQ state dicts (uncomment with a real checkpoint):
        # sd = ctx.load_state_dict("test/spectron/models/finetune.nimq")
        # print(sd, sd.names[:3])
        # sd.close()


if __name__ == "__main__":
    main()
