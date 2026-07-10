# Makefile - BybyLang (tự detect OS: Linux / macOS / Windows)
ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    EXE := 
else
    DETECTED_OS := $(shell uname -s)
    EXE :=
endif

NIMC     = nim c -d:release
BIN      = bybylang$(EXE)
DEMO_SRC = demo/demo_gpu.bybylang
DEMO_OUT = demo/demo_gpu_out$(EXE)

# Cấu trúc thư mục:
#   bybylang.nim, gpubackend.nim, tsic_ir.nim   -> core, ở gốc dự án
#   backends/cuda/{cuda_driver,cuda_runtime}.nim -> backend CUDA
#   backends/opencl/opencl_api.nim               -> backend OpenCL
#   backends/metal/{metal_backend.nim,metal_shim.m,metal_shim.h} -> backend Metal (macOS)
#   demo/                                        -> file .bybylang mẫu + output AOT
SRC_CORE    = bybylang.nim gpubackend.nim tsic_ir.nim
SRC_CUDA    = backends/cuda/cuda_driver.nim backends/cuda/cuda_runtime.nim
SRC_OPENCL  = backends/opencl/opencl_api.nim
SRC_METAL   = backends/metal/metal_backend.nim backends/metal/metal_shim.m backends/metal/metal_shim.h

.PHONY: build test clean

build: $(SRC_CORE) $(SRC_CUDA) $(SRC_OPENCL) $(SRC_METAL)
	@echo "[Makefile] OS: $(DETECTED_OS)"
	$(NIMC) -o:$(BIN) bybylang.nim

test: build
	./$(BIN) $(DEMO_SRC) --aot=$(DEMO_OUT)
	./$(DEMO_OUT)

clean:
	rm -f $(BIN) $(DEMO_OUT) $(DEMO_OUT).nim
	rm -rf ~/.cache/nim/$(notdir $(BIN))_r ~/.cache/nim/$(notdir $(DEMO_OUT))_r