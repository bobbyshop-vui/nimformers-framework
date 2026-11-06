# FIR (formers IR)

[English](fir.md) | [Tiếng Việt](fir.vi.md)

FIR is a small text file that ships with a model and describes its decode graph: which
**micro-kernels** run, which **backend kernels** to call, and the tensor names and
dimensions (taken from the model's `config.json`). It does not add a second VM or
assembler. Micro-kernels are the framework's own (`session_vm` bytecode, the BybyLang
`micro kernel` DSL, JIT in `jit_vm`), referenced by name from a registry. A model can
therefore choose or add kernels without the framework hard-coding that architecture.

The first model is Mamba-3 SISO 1.5B: [`models/mamba3-1.5b/model.fir`](../models/mamba3-1.5b/model.fir).
The full format and the kernel ABI are in [`models/FIR.md`](../models/FIR.md).

## Node kinds

| Kind | Meaning | Example |
| --- | --- | --- |
| `micro` | a registered micro-kernel (VM bytecode, JIT-compiled on GPU) | `node scan micro m3.ssm_step ...` |
| `ext` | a backend kernel by name (Q4 GEMV, embedding) | `node project ext m3.gemv_q4 ...` |
| `builtin` | small ops (`rms_norm`, `add`) | `node r builtin add ...` |

## Use

```sh
make formers-cli
# check a .fir without loading weights: resolves every name, builds every micro program
other/formers.n/formers-cli --fir-check models/mamba3-1.5b/model.fir
# run Mamba-3 1.5B with its micro-kernels chosen by the FIR file
other/formers.n/formers-cli --models test/mamba3/models \
  -m test/mamba3/models/mamba3_siso_1.5b.nimq \
  --config test/mamba3/models/mamba3-siso-1.5b/config.json \
  --tokenizer test/mamba3/models/mamba3-siso-1.5b \
  --no-lora -p 'def fib(n):' -n 24 --temp 0 --backend metal \
  --fir models/mamba3-1.5b/model.fir
```

With `--fir`, formers.n prints `FIR: provider used for N programs`. Without it the
runtime keeps its direct builders; output is the same.

## Tests

```sh
make -C other/formers.n fir-test    # or: make formers-fir-test
```

It parses the 1.5B file, builds every micro program with the 1.5B dimensions, runs the
Mamba-3 prepare/scan/carry kernels on the CPU reference interpreter against a plain-float
reference (tolerance `1e-4`), and compares the FIR provider's programs with the direct
builders.

## Verified and not verified

- Verified: Mamba-3 1.5B (base, no LoRA), greedy 24 tokens, Metal and Vulkan (MoltenVK)
  on Intel Iris Plus 640; FIR and direct runs give identical text.
- Not verified: other GPUs, LoRA/Linear++ runs, long generations. `BYBY_ONE_DISPATCH=1` is
  incompatible with `--fir`.
- FIR chooses the registered operations for fixed activation roles. It does not reorder
  the schedule or replace the resident weight/projection implementation; a new
  architecture with a different schedule also needs a runtime adapter.
