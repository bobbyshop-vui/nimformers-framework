# Makefile — build project Nimformer
#
#   make               -> build HẾT (test_nimformer + metal_ai, backend mặc định = auto) rồi chạy test_nimformer
#   make build         -> build HẾT, không chạy
#   make run           -> build + chạy test_nimformer (backend "auto": tự dò CUDA/Metal/CPU lúc runtime)
#   make run-cpu       -> build + chạy test_nimformer, ép cứng backend=cpu
#   make run-metal     -> build + chạy test_nimformer, ép cứng backend=metal (macOS)
#   make metal         -> build + chạy riêng metal_ai (macOS)
#   make cuda-lib      -> nvcc compile cuda_kernels.cu -> libcudakernels.a (cần nvcc + CUDA toolkit)
#   make run-cuda      -> build cuda-lib + nim build -d:withCuda -d:backend=cuda + chạy test_nimformer
#   make clean         -> xoá file build + checkpoint .nimq + lib CUDA

NIM      := nim
NIMFLAGS := -d:release --hints:off

TEST_SRC  := test_nimformer.nim
METAL_SRC := metal_ai.nim
CUDA_SRC  := cuda_kernels.cu
CUDA_LIB  := libcudakernels.a
NVCC      := nvcc

# CUDA 11.8 chỉ hỗ trợ tới gcc/g++ 11 — override nếu g++-11 của bạn nằm chỗ khác:
NVCC_HOSTCC := $(HOME)/toolchains/gcc11/extracted/usr/bin/g++-11

# ── Tự dò CUDA_HOME thay vì hardcode version/path ───────────────────
# Ưu tiên biến môi trường CUDA_HOME/CUDA_PATH nếu người dùng đã set sẵn
# (VD: export CUDA_HOME=/usr/local/cuda-12.4). Nếu chưa có, suy ra từ vị trí
# thực thi của $(NVCC) trong PATH: .../<cuda-home>/bin/nvcc -> <cuda-home>.
NVCC_BIN := $(shell command -v $(NVCC) 2>/dev/null)
ifndef CUDA_HOME
    ifdef CUDA_PATH
        CUDA_HOME := $(CUDA_PATH)
    else ifneq ($(NVCC_BIN),)
        CUDA_HOME := $(patsubst %/bin/$(NVCC),%,$(NVCC_BIN))
    else
        CUDA_HOME := /usr/local/cuda
    endif
endif

# Kiến trúc CPU hiện tại (x86_64, aarch64, ...) để build đúng đường dẫn
# targets/<arch>-linux/lib bên dưới $(CUDA_HOME) — thay vì hardcode x86_64-linux.
ARCH := $(shell uname -m)

# Thư viện CUDA (cudart/cublas) thường nằm ở 1 trong 2 chỗ tuỳ layout gói:
#   $(CUDA_HOME)/targets/$(ARCH)-linux/lib   (layout "targets", phổ biến trên .deb/.rpm)
#   $(CUDA_HOME)/lib64                       (layout truyền thống)
# Thêm cả 2 vào -L, linker sẽ tự bỏ qua đường dẫn không tồn tại.
CUDA_LIBDIRS := -L$(CUDA_HOME)/targets/$(ARCH)-linux/lib -L$(CUDA_HOME)/lib64

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    IS_MACOS := yes
else
    IS_MACOS := no
endif

.PHONY: all build run run-cpu run-metal metal cuda-lib run-cuda clean

all: build run

build:
	$(NIM) c $(NIMFLAGS) $(TEST_SRC)
ifeq ($(IS_MACOS),yes)
	$(NIM) c $(NIMFLAGS) $(METAL_SRC)
endif

run:
	$(NIM) c $(NIMFLAGS) $(TEST_SRC)
	./test_nimformer

run-cpu:
	$(NIM) c $(NIMFLAGS) -d:backend=cpu $(TEST_SRC)
	./test_nimformer

run-metal:
ifeq ($(IS_MACOS),yes)
	$(NIM) c $(NIMFLAGS) -d:backend=metal $(TEST_SRC)
	./test_nimformer
else
	@echo "Metal chỉ hỗ trợ trên macOS"
endif

metal:
ifeq ($(IS_MACOS),yes)
	$(NIM) c $(NIMFLAGS) $(METAL_SRC)
	./metal_ai
else
	@echo "Metal chỉ hỗ trợ trên macOS"
endif

cuda-lib: $(CUDA_LIB)

$(CUDA_LIB): $(CUDA_SRC)
	$(NVCC) -ccbin $(NVCC_HOSTCC) -c $(CUDA_SRC) -o cuda_kernels.o
	ar rcs $(CUDA_LIB) cuda_kernels.o

run-cuda: cuda-lib
	@echo ">> Dùng CUDA_HOME=$(CUDA_HOME) (arch=$(ARCH))"
	$(NIM) c $(NIMFLAGS) -d:withCuda -d:backend=cuda \
		--passL:"-L. $(CUDA_LIBDIRS) -lcudakernels -lcudart -lcublas -lstdc++" \
		$(TEST_SRC)
	./test_nimformer

clean:
	rm -f test_nimformer metal_ai
	rm -f test_nimformer.exe metal_ai.exe
	rm -rf nimcache
	rm -f model_*.nimq
	rm -f cuda_kernels.o $(CUDA_LIB)