# Mamba-3 SISO 1.5B FIR

`model.fir` describes a decode-step graph with symbolic dimensions from
`config.json`. It names the framework's existing Mamba-3 prepare, SSM, carry,
and MLP kernels. Quantized projections and embedding lookup resolve through
callable backend registry entries. No weights are included.

Run `make -C other/formers.n fir-test` for CPU reference checks and the
weight-free `--fir-check` command. See [FIR.md](../FIR.md) for memory bindings,
registry extension, and the `firMicroKernel` / JIT integration hook. A runtime
caller still supplies weights, per-layer bindings, scheduling, and GPU dispatch.
