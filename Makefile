# Makefile — build project Nimformer

.DEFAULT_GOAL := all

NIM      := nim
NIMFLAGS := -d:release --hints:off --mm:arc --opt:speed --path:. --path:test/codestral --path:test/deepseek-inference --path:test/spectron --path:kernelfortest
INFER_LORA_NIMCACHE := nimcache/infer-lora
INFER_LORA_NIMFLAGS := $(NIMFLAGS) -d:codestralInference
NIM_INFER_NIMCACHE := nimcache/nim-inference
BYBYLANG_TEST_BINS := test/bybylang_test/native_gpu_test_bin test/bybylang_test/cpu_test_bin test/bybylang_test/capi_test_bin
C_API_SRC := lib_c_api.nim
C_API_NIMCACHE := nimcache/c-api
C_API_CHECK_NIMCACHE := nimcache/c-api-check

TEST_SRC  := test/spectron/test_nimformer.nim

# ── OS detection ──
ifeq ($(OS),Windows_NT)
    IS_WINDOWS := yes
    IS_MACOS := no
else
    IS_WINDOWS := no
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Darwin)
        IS_MACOS := yes
    else
        IS_MACOS := no
    endif
endif

# Shared-library name for the C ABI (Python ctypes / C / Rust consumers).
ifeq ($(IS_WINDOWS),yes)
    C_API_LIB := nimformers.dll
else
    ifeq ($(IS_MACOS),yes)
        C_API_LIB := libnimformers.dylib
    else
        C_API_LIB := libnimformers.so
    endif
endif

PYTHON ?= python3
PYTEST ?= pytest
PYTEST_FLAGS ?= -v

ifeq ($(IS_WINDOWS),yes)
    EXE_EXT    := 
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

# ── Mamba binaries: graft / finetune (LoRA) / inference ──
GRAFT_SRC       := test/codestral/graft_attention.nim
FINETUNE_SRC    := test/codestral/finetune_lora.nim
INFER_LORA_SRC  := test/codestral/infer_lora.nim
NIM_INFER_SRC   := test/deepseek-inference/nim_inference.nim
SPECTRON_SRC    := test/spectron/test_inferencespectron.nim
COMPACT_LORA_SRC := test/codestral/compact_lora_checkpoint.nim
MIGRATE_LORA_SRC := test/codestral/migrate_lora_json_to_bin.nim
MAMBA3_FINETUNE_SRC := test/mamba3/finetune_lora.nim

GRAFT_BIN       := graft_attention_bin$(EXE_EXT)
FINETUNE_BIN    := finetune_lora_bin$(EXE_EXT)
INFER_LORA_BIN  := infer_lora_bin$(EXE_EXT)
COMPACT_LORA_BIN := compact_lora_checkpoint$(EXE_EXT)
MIGRATE_LORA_BIN := migrate_lora_json_to_bin$(EXE_EXT)
MAMBA3_HF_FINETUNE_BIN := mamba3_hf_finetune_bin$(EXE_EXT)
NIM_INFER_BIN   := nim_inference$(EXE_EXT)
SPECTRON_BIN    := test_inferencespectron$(EXE_EXT)
CLEAN_BINS      := $(TEST_BIN) $(GRAFT_BIN) $(FINETUNE_BIN) $(INFER_LORA_BIN) $(C_API_LIB) \
                   $(COMPACT_LORA_BIN) $(MIGRATE_LORA_BIN) $(MAMBA3_HF_FINETUNE_BIN) $(NIM_INFER_BIN) $(SPECTRON_BIN) \
                   $(BYBYLANG_TEST_BINS) $(addsuffix .nim,$(BYBYLANG_TEST_BINS))

.PHONY: shaders all build run run-cpu run-metal run-vulkan run-opencl run-cuda clean \
        download-models download-datasets download-codestral-tokenizer \
        build-graft build-finetune build-infer-lora build-compact-lora \
        build-migrate-lora build-nim-inference build-spectron build-for-python \
        python-test python-example c-example bybylang-test bench-gflops \
        graft finetune-mamba hf-train build-hf-train infer-lora infer-mamba infer-deepseek run-spectron mamba-pipeline \
        build-formers formers-cli formers flash-attention-test q4-gemm-test formers-test formers-fir-test formers-smoke formers-clean

all: build run

build:
	$(NIM) check $(NIMFLAGS) $(TEST_SRC)
	$(NIM) c $(NIMFLAGS) $(TEST_SRC)

run:
	$(NIM) check $(NIMFLAGS) $(TEST_SRC)
	$(NIM) c $(NIMFLAGS) $(TEST_SRC)
	./test/spectron/$(TEST_BIN)

run-cpu:
	$(NIM) check $(NIMFLAGS) -d:backend=cpu $(TEST_SRC)
	$(NIM) c $(NIMFLAGS) -d:backend=cpu $(TEST_SRC)
	./test/spectron/$(TEST_BIN)

run-metal:
ifeq ($(IS_MACOS),yes)
	$(NIM) check $(NIMFLAGS) -d:backend=metal $(TEST_SRC)
	$(NIM) c $(NIMFLAGS) -d:backend=metal $(TEST_SRC)
	./test/spectron/$(TEST_BIN)
else
	@echo "Metal is only supported on macOS"
endif

run-vulkan:
	$(NIM) check $(NIMFLAGS) -d:backend=vulkan $(TEST_SRC)
	$(NIM) c $(NIMFLAGS) -d:backend=vulkan $(TEST_SRC)
	$(RUN_PREFIX)test/spectron/$(TEST_BIN)

run-opencl:
	$(NIM) check $(NIMFLAGS) -d:backend=opencl $(TEST_SRC)
	$(NIM) c $(NIMFLAGS) -d:backend=opencl $(TEST_SRC)
	./test/spectron/$(TEST_BIN)

clean:
ifeq ($(IS_WINDOWS),yes)
	-$(RM_F) $(CLEAN_BINS) $(addsuffix .exe,$(CLEAN_BINS))
	-if exist nimcache $(RM_RF) nimcache
	-del /Q /F model_*.nimq 2>nul
else
	$(RM_F) $(CLEAN_BINS) $(addsuffix .exe,$(CLEAN_BINS))
	$(RM_RF) nimcache
	$(RM_F) model_*.nimq
	$(RM_RF) other/formers.n/build $(FORMERS_CLI)
endif
run-cuda:
	$(NIM) check $(NIMFLAGS) -d:backend=cuda $(TEST_SRC)
	$(NIM) c $(NIMFLAGS) -d:backend=cuda $(TEST_SRC)
	./test/spectron/$(TEST_BIN)

# Download public Hugging Face assets into the paths used by inference/training.
download-models:
	@set -eu; \
	mkdir -p test/deepseek-inference/models test/codestral/models/hf \
	  test/mamba3/models/mamba3-siso-187m test/mamba3/models/runs/mamba3-siso-187m-gated \
	  test/mamba3/models/mamba3-siso-1.5b test/mamba3/models/runs/mamba3-siso-1.5b-gated; \
	for file in model.nimq config.json tokenizer.json tokenizer_config.json; do \
	  curl -fL --retry 3 -C - "https://huggingface.co/bobbydeveloper2014/deepseek-coder-6.7b-nimq/resolve/main/$$file" -o "test/deepseek-inference/models/$$file"; \
	done; \
	for file in config.json params.json generation_config.json special_tokens_map.json tokenizer.json tokenizer_config.json tokenizer.model tokenizer.model.v3; do \
	  curl -fL --retry 3 -C - "https://huggingface.co/bobbydeveloper2014/codestral-v0.1-7b-nimq/resolve/main/$$file" -o "test/codestral/models/hf/$$file"; \
	done; \
	curl -fL --retry 3 -C - https://huggingface.co/bobbydeveloper2014/codestral-v0.1-7b-nimq/resolve/main/codestral_mamba_only.nimq -o test/codestral/models/codestral_mamba_only.nimq; \
	curl -fL --retry 3 -C - https://huggingface.co/bobbydeveloper2014/codestral-v0.1-nimq-qlora-linearattention/resolve/main/jamba_parallel_lora.bin -o test/codestral/models/jamba_parallel_lora.bin; \
	echo 'test/codestral/models/hf' > test/codestral/models/jamba_parallel_lora.bin.hf_tokenizer_dir; \
	for file in mamba3_siso_187m.nimq; do \
	  curl -fL --retry 3 -C - "https://huggingface.co/bobbydeveloper2014/mamba-siso-175m-nimq/resolve/main/$$file" -o "test/mamba3/models/$$file"; \
	done; \
	for file in config.json tokenizer.json tokenizer_config.json special_tokens_map.json; do \
	  curl -fL --retry 3 -C - "https://huggingface.co/bobbydeveloper2014/mamba-siso-175m-nimq/resolve/main/$$file" -o "test/mamba3/models/mamba3-siso-187m/$$file"; \
	done; \
	for file in lora.bin; do \
	  curl -fL --retry 3 -C - "https://huggingface.co/bobbydeveloper2014/mamba-siso-175m-nimq/resolve/main/$$file" -o "test/mamba3/models/runs/mamba3-siso-187m-gated/$$file"; \
	done; \
	echo 'test/mamba3/models/mamba3-siso-187m' > test/mamba3/models/runs/mamba3-siso-187m-gated/lora.bin.hf_tokenizer_dir; \
	curl -fL --retry 3 -C - https://huggingface.co/bobbydeveloper2014/mamba-siso-1.5b-nimq/resolve/main/mamba3_siso_1.5b.nimq -o test/mamba3/models/mamba3_siso_1.5b.nimq; \
	for file in config.json tokenizer.json tokenizer_config.json special_tokens_map.json; do \
	  curl -fL --retry 3 -C - "https://huggingface.co/bobbydeveloper2014/mamba-siso-1.5b-nimq/resolve/main/$$file" -o "test/mamba3/models/mamba3-siso-1.5b/$$file"; \
	done; \
	for file in lora-g1.bin; do \
	  curl -fL --retry 3 -C - "https://huggingface.co/bobbydeveloper2014/mamba-siso-1.5b-nimq/resolve/main/$$file" -o "test/mamba3/models/runs/mamba3-siso-1.5b-gated/$$file"; \
	done; \
	echo 'test/mamba3/models/mamba3-siso-1.5b' > test/mamba3/models/runs/mamba3-siso-1.5b-gated/lora-g1.bin.hf_tokenizer_dir

# Repair tokenizer assets without downloading multi-GB model weights again.
download-codestral-tokenizer:
	bash test/codestral/download_codestral_tokenizer.sh "$(INFER_LORA_DEFAULT_HFDIR)"

download-datasets:
	@set -eu; \
	mkdir -p test/spectron; \
	for shard in 1 2 3 4 5; do \
	  file="python_code_dataset_$${shard}.jsonl"; \
	  curl -fL --retry 3 -C - "https://huggingface.co/datasets/bobbydeveloper2014/python-code/resolve/main/$$file" -o "test/spectron/$$file"; \
	done; \
	curl -fL --retry 3 -C - https://huggingface.co/datasets/bobbydeveloper2014/mega-nim-python-code/resolve/main/nim_python_code_dataset_mega.jsonl -o test/spectron/nim_python_code_dataset_mega.jsonl

# ══════════════════════════════════════════════════════════════════
# Mamba: graft (Attention grafted into a Mamba-only checkpoint) / LoRA finetune /
# inference. Every target BUILDS AND THEN RUNS with sensible defaults when
# no ARGS are given - it always logs what it runs (which values,
# defaults or ARGS) and prints argument help when a required file is missing.
# ══════════════════════════════════════════════════════════════════

# Per-target defaults (override with environment variables or make graft ARGS="...")
# GRAFT_DEFAULT_OLD = GRAFT_DEFAULT_OUT = test/codestral/models/codestral_mamba_only.nimq:
# DELIBERATE - this .nimq is the output of quantize_codestral.nim (see the usage
# in that file) AND ALSO the CFG_NIMQ_PATH_DEFAULT that finetune_lora.nim
# reads back (finetune-mamba defaults to no argv). Graft OVERWRITES that very file
# (it reads everything into RAM first, then writes - safe with the current jamba_hybrid.
# loadCheckpoint/saveCheckpoint) so finetune-mamba always sees the
# "grafted" file without renaming. To keep the original, pass ARGS
# with another output name.
GRAFT_DEFAULT_OLD     := test/codestral/models/codestral_mamba_only.nimq
GRAFT_DEFAULT_PATTERN := $(shell python3 -c "print('MMMA'*16)" 2>/dev/null || echo MMMA)
GRAFT_DEFAULT_OUT     := test/codestral/models/codestral_mamba_only.nimq

INFER_LORA_DEFAULT_CKPT  := test/codestral/models/jamba_parallel_lora.bin
# The "[INST] ...[/INST] " template is for Codestral (Mistral tokenizer V3),
# not ChatML "<|user|>/<|assistant|>" (see
# finetune_lora.nim/lineToText and infer_lora.nim). The space after
# "[/INST]" is kept to match the training distribution (see the comment in infer_lora.nim).
INFER_LORA_DEFAULT_PROMPT:= [INST] Xin chao[/INST] 
INFER_LORA_DEFAULT_TEMP  := 0.7
# ADDED: infer_lora.nim (the "no LoRA checkpoint" fallback branch) still loads
# the REAL tokenizer through newHfTokenizer(hfDir) AND reads the base weights through nimqPath -
# the Makefile did NOT pass these 2 values BEFORE, so they fell back to the
# CFG_HF_DIR_DEFAULT/CFG_NIMQ_PATH_DEFAULT hard-coded SEPARATELY in infer_lora.nim
# (each binary used to have its own default path); set here to match
# the tokenizer and base checkpoint in test/codestral/models/.
INFER_LORA_DEFAULT_HFDIR := test/codestral/models/hf
INFER_LORA_DEFAULT_NIMQ  := $(GRAFT_DEFAULT_OUT)

# infer-deepseek (nim_inference.nim) is NOT part of the Mamba/LoRA pipeline -
# it is separate inference for another, GPTQ model (deepseek-coder). The default
# backend is "auto", like the parameterless newBackend() used across the repo.
INFER_DEEPSEEK_DEFAULT_MODEL   := test/deepseek-inference/models
INFER_DEEPSEEK_DEFAULT_NIMQ    := test/deepseek-inference/models/model.nimq
INFER_DEEPSEEK_DEFAULT_BACKEND := auto

build-graft:
	$(NIM) check $(NIMFLAGS) $(GRAFT_SRC)
	$(NIM) c $(NIMFLAGS) -o:$(GRAFT_BIN) $(GRAFT_SRC)

build-finetune:
	$(NIM) check $(NIMFLAGS) $(FINETUNE_SRC)
	$(NIM) c $(NIMFLAGS) -o:$(FINETUNE_BIN) $(FINETUNE_SRC)

# Direct Hugging Face source loading is opt-in and requires nimpy + Python torch.
build-hf-train:
	$(NIM) c $(NIMFLAGS) -d:mamba3HfSource -o:$(MAMBA3_HF_FINETUNE_BIN) $(MAMBA3_FINETUNE_SRC)

hf-train: build-hf-train
	$(RUN_PREFIX)$(MAMBA3_HF_FINETUNE_BIN) $(ARGS)

# Infer defaults: independent decode kernels in one session + resident GPU
# prefill (R/P/G). One-dispatch is opt-in after the Iris 640 end-to-end A/B.
# Compare paths with: make infer-lora BYBY_ONE_DISPATCH=1 /
# BYBY_PREFILL_RESIDENT=0 /
# BYBY_GEMV_VARIANT=0 / BYBY_PREFILL_PARALLEL=0 / BYBY_MAMBA_SCAN_PARALLEL=0 /
# BYBY_LP_SCAN_SERIAL=1 / BYBY_FP4_PAD_M=0.
# Optional bounded in-place FP4 merge validates delta before overwriting.
export BYBY_MODEL_MMAP ?= 1
export BYBY_MERGE_SCRATCH_MIB ?= 64
export BYBY_GEMV_TG ?= auto
export BYBY_MERGE_LORA ?= 0
export BYBY_ONE_DISPATCH ?= 0
# Auto selects the available backend; request Vulkan explicitly when needed.
export BYBY_INFER_BACKEND ?= auto
export BYBY_VULKAN_DEVICE_LOCAL ?= auto
export BYBY_PREFILL_RESIDENT ?= 1
export BYBY_GEMV_VARIANT ?= auto
export BYBY_PREFILL_FP4_KERNEL ?= auto
export BYBY_GPU_AUTOTUNE ?= 1
export BYBY_Q4_TUNED ?= 1
export BYBY_VULKAN_TENSOR_CORES ?= 1
export BYBY_MERGE_ALL_LAYERS_SESSION ?= 1
export BYBY_MERGE_QKV_ATTN_SESSION ?= 1
export BYBY_DISABLE_KV_RESIDENT ?= 0
export BYBY_DECODE_PROFILE ?= 1
export BYBY_PREFILL_PROFILE ?= 1
export BYBY_INFER_VERBOSE ?= 1
export BYBY_DEBUG_SAMPLING ?= 0
export BYBY_G_DIAGNOSTICS ?= 0
export BYBY_TRACE_BLOCKS ?= 0
export BYBY_MACOS_HIGH_PRIORITY ?= 1

build-infer-lora:
	$(NIM) check $(NIMFLAGS) kernelfortest/fp4_gemv_gpu.nim
	$(NIM) check $(INFER_LORA_NIMFLAGS) $(INFER_LORA_SRC)
	$(NIM) c $(INFER_LORA_NIMFLAGS) --nimcache:$(INFER_LORA_NIMCACHE) -o:$(INFER_LORA_BIN) $(INFER_LORA_SRC)

build-compact-lora:
	$(NIM) check $(NIMFLAGS) $(COMPACT_LORA_SRC)
	$(NIM) c $(NIMFLAGS) -o:$(COMPACT_LORA_BIN) $(COMPACT_LORA_SRC)

build-migrate-lora:
	$(NIM) check $(NIMFLAGS) $(MIGRATE_LORA_SRC)
	$(NIM) c $(NIMFLAGS) -o:$(MIGRATE_LORA_BIN) $(MIGRATE_LORA_SRC)

build-nim-inference:
	$(NIM) check $(NIMFLAGS) $(NIM_INFER_SRC)
	$(NIM) c $(NIMFLAGS) --nimcache:$(NIM_INFER_NIMCACHE) -o:$(NIM_INFER_BIN) $(NIM_INFER_SRC)

build-spectron:
	$(NIM) check $(NIMFLAGS) $(SPECTRON_SRC)
	$(NIM) c $(NIMFLAGS) -o:$(SPECTRON_BIN) $(SPECTRON_SRC)

build-for-python:
	$(NIM) check $(NIMFLAGS) --nimcache:$(C_API_CHECK_NIMCACHE) $(C_API_SRC)
	$(NIM) c $(NIMFLAGS) --app:lib --nimcache:$(C_API_NIMCACHE) -o:$(C_API_LIB) $(C_API_SRC)
	@echo "Built $(C_API_LIB) (C ABI v2); header: include/nimformers.h"

python-test: build-for-python
	@echo "[python-test] Suite: test/python_test"
	@echo "[python-test] Library: $(C_API_LIB) (C ABI v2)"
	$(PYTEST) $(PYTEST_FLAGS) test/python_test
	@echo "[python-test] Suite finished."

python-example: build-for-python
	$(PYTHON) examples/python/basic.py

c-example: build-for-python
	$(CC) -Iinclude examples/c/smoke.c -L. -lnimformers -Wl,-rpath,$(CURDIR) -o examples/c/smoke
	DYLD_LIBRARY_PATH=. LD_LIBRARY_PATH=. ./examples/c/smoke

bybylang-test:
	bash ./test/bybylang_test/run.sh

bench-gflops: build-for-python
	$(PYTHON) benchmarks/bench_gflops.py

# ── graft: build + RUN AUTOMATICALLY. No ARGS -> the GRAFT_DEFAULT_*
# defaults. If the default old.bin/.nimq does not exist, PRINTS HELP
# and stops (does not run blindly). ──
# custom example: make graft ARGS="checkpoint.bin MMAMMSM out.bin"
graft: build-graft
ifeq ($(strip $(ARGS)),)
	@echo "[graft] No ARGS -> using the defaults:"
	@echo "[graft]   old     = $(GRAFT_DEFAULT_OLD)"
	@echo "[graft]   pattern = $(GRAFT_DEFAULT_PATTERN)"
	@echo "[graft]   out     = $(GRAFT_DEFAULT_OUT)"
	@if [ "$(GRAFT_DEFAULT_OLD)" = "$(GRAFT_DEFAULT_OUT)" ]; then \
		echo "[graft] NOTE: out has the same name as old (deliberate, see the GRAFT_DEFAULT_* comment in the Makefile) - $(GRAFT_DEFAULT_OLD) will be OVERWRITTEN after grafting."; \
	fi
	@if [ ! -f "$(GRAFT_DEFAULT_OLD)" ]; then \
		echo "[graft] ERROR: $(GRAFT_DEFAULT_OLD) not found."; \
		echo "[graft] This file must exist first (output of test/codestral/quantize_codestral_bin <hf_dir> $(GRAFT_DEFAULT_OLD))."; \
		echo "[graft] Usage: make graft ARGS=\"<old.bin> <pattern vd MMAMMSM> <out.bin>\""; \
		echo "[graft]         or: make graft ARGS=\"--from-hf <hf.bin> <vocab> <embedDim> <nHeads> <ffMult> <nLayers> <pattern> <out.bin>\""; \
		exit 1; \
	fi
	@echo "[graft] Running: $(RUN_PREFIX)$(GRAFT_BIN) $(GRAFT_DEFAULT_OLD) $(GRAFT_DEFAULT_PATTERN) $(GRAFT_DEFAULT_OUT)"
	$(RUN_PREFIX)$(GRAFT_BIN) $(GRAFT_DEFAULT_OLD) $(GRAFT_DEFAULT_PATTERN) $(GRAFT_DEFAULT_OUT)
else
	@echo "[graft] Running with the given ARGS: $(RUN_PREFIX)$(GRAFT_BIN) $(ARGS)"
	$(RUN_PREFIX)$(GRAFT_BIN) $(ARGS)
endif

# ── finetune-mamba: finetune_lora.nim NOW READS ARGV (optional) and falls back
# to its fixed config (CFG_*_DEFAULT in finetune_lora.nim) when nothing is given.
# argv order: initMode oldCheckpoint hfDir hfSource nimqPath outCheckpointPath ──
# custom example: make finetune-mamba ARGS="codestral - test/codestral/models/hf nimq test/codestral/models/codestral_mamba_only.nimq test/codestral/models/my_lora.bin"
# (use "-" for a parameter that keeps its default when a later one must be set)
finetune-mamba: build-finetune
ifeq ($(strip $(ARGS)),)
	@echo "[finetune-mamba] No ARGS -> using the fixed config in finetune_lora.nim (CFG_*_DEFAULT)."
	@echo "[finetune-mamba] Running: $(RUN_PREFIX)$(FINETUNE_BIN)"
	$(RUN_PREFIX)$(FINETUNE_BIN)
else
	@echo "[finetune-mamba] Running with the given ARGS: $(RUN_PREFIX)$(FINETUNE_BIN) $(ARGS)"
	$(RUN_PREFIX)$(FINETUNE_BIN) $(ARGS)
endif

# ── infer-lora: no ARGS -> the INFER_LORA_DEFAULT_* defaults.
# If the default checkpoint does not exist, prints help and stops. ──
# full argv: ckptPath prompt temperature hfDir nimqPath (the last 2 are NEW -
# infer_lora.nim needs both hfDir (the real tokenizer) AND nimqPath (the base
# weights, used when falling back to pure Mamba with no LoRA checkpoint yet)).
# custom example: make infer-lora ARGS="test/codestral/models/jamba_lora.bin '[INST] Xin chao[/INST] ' 0.7 test/codestral/models/hf test/codestral/models/codestral_mamba_only.nimq"
infer-lora: build-infer-lora
ifeq ($(strip $(ARGS)),)
	@echo "[infer-lora] No ARGS -> using the defaults:"
	@echo "[infer-lora]   checkpoint prefix (name)     = $(INFER_LORA_DEFAULT_CKPT)"
	@echo "[infer-lora]   LoRA adapter (if any)        = $(INFER_LORA_DEFAULT_CKPT)"
	@echo "[infer-lora]   prompt      = $(INFER_LORA_DEFAULT_PROMPT)"
	@echo "[infer-lora]   temperature = $(INFER_LORA_DEFAULT_TEMP)"
	@echo "[infer-lora]   hfDir       = $(INFER_LORA_DEFAULT_HFDIR)"
	@echo "[infer-lora]   nimqPath    = $(INFER_LORA_DEFAULT_NIMQ)"
	@if [ -f "$(INFER_LORA_DEFAULT_CKPT)" ]; then \
		echo "[infer-lora] Found $(INFER_LORA_DEFAULT_CKPT) -> using the trained LoRA."; \
	else \
		echo "[infer-lora] $(INFER_LORA_DEFAULT_CKPT) NOT found - infer_lora.nim will AUTOMATICALLY"; \
		echo "[infer-lora] fall back to PURE Mamba from the pretrained base (no error, see loadPureMambaForInfer)."; \
		echo "[infer-lora] Run 'make finetune-mamba' first to use a fine-tuned version."; \
	fi
	@if [ ! -f "$(INFER_LORA_DEFAULT_NIMQ)" ]; then \
		echo "[infer-lora] WARNING: $(INFER_LORA_DEFAULT_NIMQ) not found - run 'make graft' (or quantize_codestral_bin) first."; \
	fi
	@echo "[infer-lora] Running: $(RUN_PREFIX)$(INFER_LORA_BIN) $(INFER_LORA_DEFAULT_CKPT) \"$(INFER_LORA_DEFAULT_PROMPT)\" $(INFER_LORA_DEFAULT_TEMP) \"$(INFER_LORA_DEFAULT_HFDIR)\" $(INFER_LORA_DEFAULT_NIMQ)"
	$(RUN_PREFIX)$(INFER_LORA_BIN) "$(INFER_LORA_DEFAULT_CKPT)" "$(INFER_LORA_DEFAULT_PROMPT)" "$(INFER_LORA_DEFAULT_TEMP)" "$(INFER_LORA_DEFAULT_HFDIR)" "$(INFER_LORA_DEFAULT_NIMQ)"
else
	@echo "[infer-lora] Running with the given ARGS: $(RUN_PREFIX)$(INFER_LORA_BIN) $(ARGS)"
	$(RUN_PREFIX)$(INFER_LORA_BIN) $(ARGS)
endif

# ── infer-mamba: ALIAS of infer-lora - the RIGHT inference for the
# Mamba pipeline (graft -> finetune-mamba -> infer-mamba), using the
# FINE-TUNED checkpoint (base .bin + .lora.json adapter) through infer_lora_bin, NOT
# a separate GPTQ model. For the deepseek GPTQ model,
# unrelated to Mamba, use 'make infer-deepseek'. ──
infer-mamba: infer-lora

# ── infer-deepseek: inference through nim_inference.nim - a COMPLETELY SEPARATE GPTQ model
# (deepseek-coder), NOT sharing the Mamba graft/finetune
# pipeline. Default backend "auto" (like the parameterless newBackend() used
# across the repo - picks an available GPU, raises if there is none instead of
# silently falling back to the CPU). ──
# custom example: make infer-deepseek ARGS="test/deepseek-inference/models test/deepseek-inference/models/model.nimq auto"
infer-deepseek: build-nim-inference
ifeq ($(strip $(ARGS)),)
	@echo "[infer-deepseek] No ARGS -> using the defaults:"
	@echo "[infer-deepseek]   modelPath = $(INFER_DEEPSEEK_DEFAULT_MODEL)"
	@echo "[infer-deepseek]   nimqPath  = $(INFER_DEEPSEEK_DEFAULT_NIMQ)"
	@echo "[infer-deepseek]   backend   = $(INFER_DEEPSEEK_DEFAULT_BACKEND)"
	@if [ ! -f "$(INFER_DEEPSEEK_DEFAULT_NIMQ)" ]; then \
		echo "[infer-deepseek] WARNING: $(INFER_DEEPSEEK_DEFAULT_NIMQ) not found in the current directory - loading the weights will fail."; \
		echo "[infer-deepseek] Usage: make infer-deepseek ARGS=\"<modelPath> <nimqPath> <backend> <prompt> <maxNewTokens> <temperature> <topP> <systemText>\""; \
	fi
	@echo "[infer-deepseek] Running: $(RUN_PREFIX)$(NIM_INFER_BIN) $(INFER_DEEPSEEK_DEFAULT_MODEL) $(INFER_DEEPSEEK_DEFAULT_NIMQ) $(INFER_DEEPSEEK_DEFAULT_BACKEND)"
	$(RUN_PREFIX)$(NIM_INFER_BIN) "$(INFER_DEEPSEEK_DEFAULT_MODEL)" "$(INFER_DEEPSEEK_DEFAULT_NIMQ)" "$(INFER_DEEPSEEK_DEFAULT_BACKEND)"
else
	@echo "[infer-deepseek] Running with the given ARGS: $(RUN_PREFIX)$(NIM_INFER_BIN) $(ARGS)"
	$(RUN_PREFIX)$(NIM_INFER_BIN) $(ARGS)
endif

# ── formers.n (other/formers.n): llama.cpp-style framework written in BybyLang;
# detects Llama / Mamba-2 / Mamba-2 + LoRA/Linear++ -> formers-cli ──
#   make formers-cli                      build other/formers.n/formers-cli
#   make formers                          build, then chat with FORMERS_MODELS
#   make formers FORMERS_MODELS=test/deepseek-inference/models
#   make formers ARGS='--models test/codestral/models -p "..." -n 128'
FORMERS_CLI    := other/formers.n/formers-cli
FORMERS_MODELS ?= test/codestral/models

build-formers:
	$(MAKE) -C other/formers.n build

formers-cli: build-formers

formers: build-formers
ifeq ($(strip $(ARGS)),)
	@if [ ! -d "$(FORMERS_MODELS)" ]; then \
		echo "[formers] $(FORMERS_MODELS) not found - run make download-models, or pass FORMERS_MODELS=<dir> / ARGS='--models <dir> ...'"; \
		exit 1; \
	fi
	$(FORMERS_CLI) --models "$(FORMERS_MODELS)" -cnv
else
	$(FORMERS_CLI) $(ARGS)
endif

flash-attention-test:
	nim c -d:release --hints:off --path:vendor/bybylang --nimcache:nimcache/test-flash \
	  -o:nimcache/test-flash/test_flash_attention test/deepseek-inference/test_flash_attention.nim
	nimcache/test-flash/test_flash_attention $(ARGS)

q4-gemm-test:
	nim c -d:release --hints:off --path:. --path:vendor/bybylang --path:kernelfortest --nimcache:nimcache/test-q4gemm \
	  -o:nimcache/test-q4gemm/test_q4_gemm test/deepseek-inference/test_q4_gemm.nim
	nimcache/test-q4gemm/test_q4_gemm $(ARGS)

formers-test:
	$(MAKE) -C other/formers.n test

formers-fir-test:
	$(MAKE) -C other/formers.n fir-test

formers-smoke:
	$(MAKE) -C other/formers.n smoke

formers-clean:
	$(MAKE) -C other/formers.n clean

run-spectron: build-spectron
	@echo "[run-spectron] Running: $(RUN_PREFIX)$(SPECTRON_BIN) $(ARGS)"
	$(RUN_PREFIX)$(SPECTRON_BIN) $(ARGS)

# ── mamba-pipeline: runs the whole finetunemamba.sh (existing env variables
# set SKIP_TO_FINETUNE automatically when an adaptive checkpoint exists - see the script) ──
mamba-pipeline:
	@echo "[mamba-pipeline] Running ./test/codestral/finetunemamba.sh (set env variables before make to customise, e.g. SKIP_TO_FINETUNE=1 make mamba-pipeline)"
	./test/codestral/finetunemamba.sh
# ── Compiles every Vulkan shader (.comp -> .spv). Needs glslc (Vulkan SDK / brew install shaderc).
# Usage: make shaders   (add -B to force a rebuild)
VK_KERNEL_DIR := vendor/bybylang/backends/vulkan/kernels
shaders:
	@command -v glslc >/dev/null || { echo "[shaders] glslc not found - brew install shaderc (or install the Vulkan SDK)"; exit 1; }
	@for f in $(VK_KERNEL_DIR)/*.comp; do \
		out=$${f%.comp}.spv; \
		if [ ! -f "$$out" ] || [ "$$f" -nt "$$out" ]; then \
			echo "[shaders] glslc $$f"; glslc "$$f" -o "$$out" || exit 1; \
		fi; \
	done
	@echo "[shaders] done."
