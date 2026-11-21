## Stable Nimble-facing module for the dynamic C ABI bindings.
##
## Consumers can `import nimformers_shared` without importing the framework's
## implementation modules. The wrapper resolves the installed platform
## library relative to this module, or uses `NIMFORMERS_LIB` when overridden.

import bindings/nimformers_shared

export nimformers_shared
