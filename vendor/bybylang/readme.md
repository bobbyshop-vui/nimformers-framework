# BybyLang - Cross-Platform GPU Programming Language

**BybyLang** is a high-level programming language with Python-like syntax, designed to run **directly on the GPU** **without writing shaders**. Just write simple code, and BybyLang automatically compiles and runs it on the GPU backend best suited to your machine.

---

## ✨ Key Features

- **Automatic GPU Offload** - Code runs on the GPU without writing shaders
- **Multi-backend** - Supports CUDA, Metal, OpenCL, TSIC (intermediate representation)
- **Automatic fallback** - Falls back to CPU automatically if no GPU is available
- **No build dependencies** - Kernels are compiled at runtime, no build tools required
- **Nested control flow support** - Unlimited nesting of if/while/for
- **High performance** - Optimized per backend (Tensor Core on CUDA, MPS on Metal)

---

## 📋 System Requirements

| Backend | Requirements | Platform |
|---|---|---|
| **CUDA** | NVIDIA driver, libcuda.so | Linux, Windows |
| **Metal** | macOS 10.13+, Xcode CLT | macOS |
| **OpenCL** | OpenCL 1.2+ driver | Linux, macOS, Windows |
| **TSIC** | None (intermediate representation) | Any platform |

---

## 🛠️ Installation & Build

### 1. Install the Nim compiler

```bash
# Install Nim (if not already installed)
curl https://nim-lang.org/choosenim/init.sh -sSf | sh

# Or use a package manager:
# sudo apt install nim  # Ubuntu/Debian
# brew install nim      # macOS
```

### 2. Build BybyLang

```bash
# Clone the repository
git clone https://github.com/bobbyshop-vui/bybylang
cd bybylang

# Build using the Makefile
make build

# Or build manually
nim c -d:release -o:bybylang bybylang.nim
```

### 3. Run tests and build your bybylang code

```bash
# Run the full test suite
make test
```
```bash
./bybylang (your bybylang code) --aot=(name the execute export and if you are using windows you need to add .exe)
```
### 4. Project directory structure

```
bybylang/
├── Makefile                  # Build script
├── bybylang.nim              # Main compiler
├── gpubackend.nim            # GPU dispatch layer
├── tsic_ir.nim               # TSIC Intermediate Representation
├── backends/
│   ├── cuda/
│   │   ├── cuda_driver.nim   # CUDA Driver API
│   │   ├── cuda_runtime.nim  # CUDA Runtime + cuBLAS
│   │   └── kernels/
│   │       └── vecop.ptx     # PTX kernels
│   ├── metal/
│   │   ├── metal_backend.nim # Nim wrapper
│   │   ├── metal_shim.m      # Objective-C shim
│   │   ├── metal_shim.h      # Header
│   │   └── kernels/
│   │       └── vecop_matmul.metal # MSL kernels
│   └── opencl/
│       ├── opencl_api.nim    # OpenCL bindings
│       └── kernels/
│           └── vecop_matmul.cl   # OpenCL C kernels
```

---

## 📝 BybyLang Syntax

### 1. Backend declaration

```
# Automatically select the best backend (default)
gpu backend is "auto"

# Select a specific backend
gpu backend is "cuda"      # NVIDIA GPU
gpu backend is "metal"     # macOS Apple Silicon/AMD
gpu backend is "opencl"    # AMD/Intel GPU
gpu backend is "tsic"      # Intermediate IR (for custom GPUs)
gpu backend is "cpu"       # Run on CPU (fallback)
```

### 2. Declaring GPU arrays

```
# 1D array
gpu array A = [1, 2, 3, 4, 5, 6, 7, 8]
gpu array B = [10, 20, 30, 40, 50, 60, 70, 80]

# Array with floating-point numbers
gpu array X = [1.5, 2.5, 3.5]
gpu array Y = [0.5, 1.5, 2.5]
```

### 3. Vector operations (element-wise)

```
# Basic operations
gpu add A, B -> C          # C = A + B
gpu sub A, B -> D          # D = A - B
gpu mul A, B -> E          # E = A * B
gpu div A, B -> F          # F = A / B

# Optionally add "size N" (descriptive only)
gpu add A, B -> C size 8
```

### 4. Matrix operations

```
# C(m x n) = A(m x k) * B(k x n)
# Dimensions must be declared
gpu matmul A, B -> C m 2 k 3 n 4

# Concrete example
gpu array M1 = [1, 2, 3, 4, 5, 6]    # 2x3
gpu array M2 = [7, 8, 9, 10, 11, 12] # 3x2
gpu matmul M1, M2 -> Result m 2 k 3 n 2
```

### 4b. Activations and NN kernels

These map 1:1 to the same `gpubackend.nim` kernels used by nimformer's
`backend.nim` (relu/sigmoid/tanh/softmax/layernorm/embedding lookup), across
every backend (cuda/metal/opencl/tsic/cpu) — adding a kernel there makes it
available here automatically, no separate shader needed.

```
# Elementwise activations
gpu relu X -> Y
gpu sigmoid X -> Y
gpu tanh X -> Y

# Row-wise softmax: X is rows*cols flattened, row-major
gpu softmax X -> Y rows 2 cols 4

# Layernorm over each row of X (gamma/beta length = cols), eps is the
# numerical-stability epsilon added before the sqrt
gpu layernorm X, Gamma, Beta -> Y rows 2 cols 4 eps 0.00001

# Embedding lookup: TABLE is vocab*dim flattened, INDICES is int32 ids.
# Note: "gpu array" only generates seq[float32], so Indices must come from a
# plain Nim seq[int32] variable (e.g. one you assign to directly), not "gpu array".
gpu embedding Table, Indices -> Y vocab 100 dim 16
```

### 5. Control flow

```
# Variables and assignment
x = 5
y = 10
z = x + y

# If-elif-else (supports deep nesting)
if x > 0:
    print "x is positive"
    if x > 3:
        print "x is greater than 3"
        if x == 5:
            print "x equals 5"
        else:
            print "x is not 5"
    else:
        print "x is between 1 and 3"
elif x == 0:
    print "x equals 0"
else:
    print "x is negative"

# While loop
i = 0
while i < 5:
    print "i =", i
    i = i + 1

# For loop (supports range)
for i in range(0, 10):
    print "i =", i
    if i == 5:
        print "reached 5"
```

### 6. Function definitions

```
# Define a function
function my_func:
    a = 10
    b = 20
    c = a + b
    print "Sum =", c
my_func

# Functions can call other functions
function greet:
    print "Hello from GPU!"
function main:
    print "Start"
    greet()
    print "End"
main
```

### 7. System commands (APU)

```
# APU Transfer
apu tran "data" with "payload"

# APU Memory
apu mem write "RAM0" with "100"
apu mem read "RAM0"

# APU Core
apu core with "run"

# APU Pin
apu pin 1 to high

# Bit operations
bit send "01010101"
bit recv

# Memory mapping
mem map "GPU_MEM_0"

# Memory push
mem push "BUFFER_0" with "data"

# Pulse transfer
tran pulse 4 width "100ns"
```

---

## 🚀 Detailed Examples

### Example 1: GPU Vector Addition

**File: examples/demo_gpu.bybylang**

```
# Select the backend automatically
gpu backend is "auto"

# Declare data
gpu array A = [1, 2, 3, 4, 5, 6, 7, 8]
gpu array B = [10, 20, 30, 40, 50, 60, 70, 80]

# GPU operations
gpu add A, B -> C
gpu mul A, B -> D

# Print results
print "Addition result:"
print C
print "Multiplication result:"
print D
```

### Example 2: Matrix Multiplication

**File: examples/demo_matmul.bybylang**

```
# Select the Metal backend (or CUDA/OpenCL)
gpu backend is "metal"

# 2x3 matrix
gpu array M1 = [1, 2, 3, 4, 5, 6]

# 3x2 matrix
gpu array M2 = [7, 8, 9, 10, 11, 12]

# Matrix multiplication: 2x3 * 3x2 = 2x2
gpu matmul M1, M2 -> Result m 2 k 3 n 2

print "Matrix result:"
print Result
```

### Example 3: Control Flow + GPU

**File: examples/demo_control.bybylang**

```
# Combine control flow and GPU
x = 5
gpu backend is "auto"

if x > 0:
    gpu array A = [1, 2, 3, 4]
    gpu array B = [5, 6, 7, 8]
    gpu add A, B -> C
    print "GPU result:", C
    if x == 5:
        print "x = 5, GPU ran successfully"
    else:
        print "x is not 5"
else:
    print "x is not positive"
```

---

## 🔧 Backend Details

### 1. CUDA Backend

- **File:** backends/cuda/cuda_driver.nim
- **Kernel:** PTX (loaded at runtime with cuModuleLoadData)
- **Matmul:** cuBLAS with Tensor Core (CUBLAS_TENSOR_OP_MATH)
- **Requirements:** libcuda.so, NVIDIA driver

### 2. Metal Backend

- **File:** backends/metal/metal_shim.m
- **Kernel:** MSL (compiled at runtime with newLibraryWithSource)
- **Matmul:** Naive kernel (can use MPSMatrixMultiplication)
- **Requirements:** macOS, Metal.framework

### 3. OpenCL Backend

- **File:** backends/opencl/opencl_api.nim
- **Kernel:** OpenCL C (compiled at runtime with clBuildProgram)
- **Matmul:** Naive kernel (runs on any GPU/CPU)
- **Requirements:** libOpenCL.so, OpenCL driver

### 3.1 Thư viện API cho framework nhúng (không qua cú pháp .bybylang)

Ngoài cú pháp `.bybylang`, mọi backend đều expose qua `gpubackend.nim` như
thư viện Nim thuần để framework khác (ví dụ framework transformer/AI) nhúng
làm lớp GPU của họ mà không cần tự viết shader:

```nim
import gpubackend

let c  = gpuMatmul(gbAuto, a, b, m, k, n)          # 1 phép matmul
let r  = gpuMatmul2(gbAuto, a1, b1, m1, k1, n1,     # 2 phép matmul ĐỘC LẬP,
                     a2, b2, m2, k2, n2)             # gộp 1 round-trip GPU
# r.c1, r.c2
```

`gpuMatmul2` trên backend Metal gộp cả 2 dispatch vào **1 command buffer**
(1 lần commit+wait) thay vì gọi `gpuMatmul` hai lần — hữu ích cho các phép
tính có 2 matmul độc lập cần chạy song song (vd. gradient trọng số và
gradient đầu vào trong lan truyền ngược của một lớp Linear).

### 4. TSIC Backend (Intermediate Representation)

- **File:** tsic_ir.nim
- **Role:** Hardware-independent intermediate representation
- **Lowering:** Can emit to PTX, MSL, OpenCL C, GLSL
- **Benefit:** Easy to add new backends, standardized kernels

---

## 🧪 Testing

### Running tests

```bash
# Build and run all tests
make build
make test
```

### Example test output

```
$ make test
[INFO] Generated Nim code to test_gpu.nim
[INFO] Built executable: test_gpu
Addition result:
@[11.0, 22.0, 33.0, 44.0, 55.0, 66.0, 77.0, 88.0]
Multiplication result:
@[10.0, 40.0, 90.0, 160.0, 250.0, 360.0, 490.0, 640.0]
GPU test PASSED

$ make test-tsic
[INFO] Built executable: test_tsic
TSIC IR test PASSED
```

---

## 📄 License

MIT License - See the LICENSE file for details.

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request on GitHub.

---

**Built with ❤️ by the BybyLang team**