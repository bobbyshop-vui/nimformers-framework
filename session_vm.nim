## Compatibility shim — implementation moved to vendor/bybylang/micro_kernel.
## Prefer: `import nimformers` (re-exports) or `import vendor/bybylang/micro_kernel`.
## It must stay a shim (no copied code): two copies => two different VmProgram types in Nim.
import vendor/bybylang/micro_kernel/session_vm
export session_vm
