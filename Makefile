# Makefile — build project Nimformer
#
#   make               -> build HẾT (test_nimformer + metal_ai, backend mặc định = auto) rồi chạy test_nimformer
#   make build         -> build HẾT, không chạy
#   make run           -> build + chạy test_nimformer (backend "auto": tự dò CUDA/Metal/CPU lúc runtime)
#   make run-cpu       -> build + chạy test_nimformer, ép cứng backend=cpu
#   make run-metal     -> build + chạy test_nimformer, ép cứng backend=metal (macOS)
#   make metal         -> build + chạy riêng metal_ai (macOS)
#   make cuda-lib      -> compile cuda_kernels.cu -> lib CUDA (Linux/macOS: libcudakernels.a qua nvcc+ar;
#                          Windows: cudakernels.lib qua nvcc+lib.exe, cần MSVC)
#   make run-cuda      -> build cuda-lib + nim build -d:withCuda -d:backend=cuda + chạy test_nimformer
#   make clean         -> xoá file build + checkpoint .nimq + lib CUDA
#
# YÊU CẦU RIÊNG CHO run-cuda TRÊN WINDOWS:
#   - Cài CUDA Toolkit (đặt sẵn biến môi trường CUDA_PATH, trình cài đặt tự set).
#   - Cài Visual Studio Build Tools (MSVC) — nvcc trên Windows dùng cl.exe làm
#     host compiler, không dùng gcc/clang được.
#   - Chạy "make run-cuda" TỪ "x64 Native Tools Command Prompt for VS" (hoặc
#     đã tự chạy vcvarsall.bat) để có sẵn cl.exe, lib.exe, link.exe trong PATH.
#   - Target này tự thêm --cc:vcc để Nim sinh code biên dịch bằng MSVC thay vì
#     MinGW gcc mặc định, vì thư viện CUDA (.lib) trên Windows là định dạng
#     COFF/MSVC, không tương thích tốt với linker của MinGW.

NIM      := nim
NIMFLAGS := -d:release --hints:off

TEST_SRC  := test_nimformer.nim
METAL_SRC := metal_ai.nim
CUDA_SRC  := cuda_kernels.cu
NVCC      := nvcc

# ── Nhận diện Windows TRƯỚC TIÊN qua biến môi trường $(OS) (luôn có sẵn trên
# Windows, kể cả cmd.exe thuần, không cần uname) ──
ifeq ($(OS),Windows_NT)
    IS_WINDOWS := yes
else
    IS_WINDOWS := no
endif

ifeq ($(IS_WINDOWS),yes)
    EXE_EXT    := .exe
    RUN_PREFIX :=
    RM_F        = del /Q /F
    RM_RF       = rmdir /S /Q
else
    EXE_EXT    :=
    RUN_PREFIX := ./
    RM_F        = rm -f
    RM_RF       = rm -rf
endif

TEST_BIN  := test_nimformer$(EXE_EXT)
METAL_BIN := metal_ai$(EXE_EXT)

# ═══════════════════════════════════════════════════════════════
# CUDA — 2 nhánh hoàn toàn khác nhau: Linux/macOS (nvcc+ar, gcc/clang link)
# và Windows (nvcc+cl.exe, lib.exe, link theo kiểu MSVC).
# ═══════════════════════════════════════════════════════════════

ifeq ($(IS_WINDOWS),yes)

# Windows: CUDA_PATH do bộ cài CUDA Toolkit tự set sẵn, VD:
# C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.4
CUDA_HOME    := $(CUDA_PATH)
CUDA_LIBDIR  := $(CUDA_HOME)\lib\x64
CUDA_OBJ     := cuda_kernels.obj
CUDA_LIB     := cudakernels.lib

else

NVCC_HOSTCC := $(HOME)/toolchains/gcc11/extracted/usr/bin/g++-11

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

ARCH := $(shell uname -m)
CUDA_LIBDIRS := -L$(CUDA_HOME)/targets/$(ARCH)-linux/lib -L$(CUDA_HOME)/lib64
CUDA_OBJ     := cuda_kernels.o
CUDA_LIB     := libcudakernels.a

UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Darwin)
    IS_MACOS := yes
else
    IS_MACOS := no
endif

endif

ifndef IS_MACOS
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
	$(RUN_PREFIX)$(TEST_BIN)

run-cpu:
	$(NIM) c $(NIMFLAGS) -d:backend=cpu $(TEST_SRC)
	$(RUN_PREFIX)$(TEST_BIN)

run-metal:
ifeq ($(IS_MACOS),yes)
	$(NIM) c $(NIMFLAGS) -d:backend=metal $(TEST_SRC)
	$(RUN_PREFIX)$(TEST_BIN)
else
	@echo "Metal chi ho tro tren macOS"
endif

metal:
ifeq ($(IS_MACOS),yes)
	$(NIM) c $(NIMFLAGS) $(METAL_SRC)
	$(RUN_PREFIX)$(METAL_BIN)
else
	@echo "Metal chi ho tro tren macOS"
endif

# ── cuda-lib: build thư viện CUDA tĩnh ───────────────────────────────
cuda-lib: $(CUDA_LIB)

ifeq ($(IS_WINDOWS),yes)
$(CUDA_LIB): $(CUDA_SRC)
	$(NVCC) -c $(CUDA_SRC) -o $(CUDA_OBJ)
	lib /OUT:$(CUDA_LIB) $(CUDA_OBJ)
else
$(CUDA_LIB): $(CUDA_SRC)
	$(NVCC) -ccbin $(NVCC_HOSTCC) -c $(CUDA_SRC) -o $(CUDA_OBJ)
	ar rcs $(CUDA_LIB) $(CUDA_OBJ)
endif

# ── run-cuda: build lib CUDA + build Nim link vào + chạy ─────────────
ifeq ($(IS_WINDOWS),yes)
run-cuda: cuda-lib
	@echo Dung CUDA_HOME=$(CUDA_HOME)
	$(NIM) c $(NIMFLAGS) --cc:vcc -d:withCuda -d:backend=cuda \
		--passL:"$(CUDA_LIB) $(CUDA_LIBDIR)\cudart.lib $(CUDA_LIBDIR)\cublas.lib" \
		$(TEST_SRC)
	$(RUN_PREFIX)$(TEST_BIN)
else
run-cuda: cuda-lib
	@echo ">> Dung CUDA_HOME=$(CUDA_HOME) (arch=$(ARCH))"
	$(NIM) c $(NIMFLAGS) -d:withCuda -d:backend=cuda \
		--passL:"-L. $(CUDA_LIBDIRS) -lcudakernels -lcudart -lcublas -lstdc++" \
		$(TEST_SRC)
	$(RUN_PREFIX)$(TEST_BIN)
endif

clean:
ifeq ($(IS_WINDOWS),yes)
	-if exist test_nimformer.exe $(RM_F) test_nimformer.exe
	-if exist metal_ai.exe $(RM_F) metal_ai.exe
	-if exist nimcache $(RM_RF) nimcache
	-del /Q /F model_*.nimq 2>nul
	-if exist $(CUDA_OBJ) $(RM_F) $(CUDA_OBJ)
	-if exist $(CUDA_LIB) $(RM_F) $(CUDA_LIB)
else
	$(RM_F) test_nimformer metal_ai
	$(RM_F) test_nimformer.exe metal_ai.exe
	$(RM_RF) nimcache
	$(RM_F) model_*.nimq
	$(RM_F) $(CUDA_OBJ) $(CUDA_LIB)
endif
