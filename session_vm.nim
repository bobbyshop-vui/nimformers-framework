## Compatibility shim — implementation moved to vendor/bybylang/micro_kernel.
## Prefer: `import nimformers` (re-exports) or `import vendor/bybylang/micro_kernel`.
## Phải là shim (không copy code): hai bản copy => hai kiểu VmProgram khác nhau trong Nim.
import vendor/bybylang/micro_kernel/session_vm
export session_vm
