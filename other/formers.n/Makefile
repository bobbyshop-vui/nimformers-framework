# formers.n — build the BybyLang sources into formers-cli and run the tests.
#
#   make            build ./formers-cli
#   make test       architecture-detection test (no model load, no GPU)
#   make smoke      tiny random Llama (2 layers, GQA) end to end on the GPU
#
# Paths are relative to the repository root; this Makefile changes there so
# the generated Nim code and the tests see the same layout as the root
# Makefile (make build-formers / make formers-test).

ROOT        := $(abspath $(dir $(lastword $(MAKEFILE_LIST)))/../..)
HERE        := other/formers.n
BYBYLANG    := vendor/bybylang/bybylang
CLI_SRC     := $(HERE)/src/formers_cli.bybylang
CLI_BIN     := $(HERE)/formers-cli
TEST_SRC    := $(HERE)/tests/test_detect.bybylang
FIR_TEST_SRC := $(HERE)/tests/test_fir.bybylang
SOURCES     := $(shell find $(ROOT)/$(HERE)/src -name '*.bybylang')

export BYBYLANG_NIMCACHE ?= $(ROOT)/nimcache/formers-n
export FIR_LIFT_TMP ?= /tmp/fir-lift-tests

.PHONY: all build build-arm64 test fir-test fir-graph-test fir-probe fir-lift-test smoke clean

all: build

$(ROOT)/$(BYBYLANG): $(ROOT)/vendor/bybylang/bybylang.nim
	$(MAKE) -C $(ROOT)/vendor/bybylang build

# BybyLang writes <out>.nim next to <out> and compiles it; Nim picks up
# $(HERE)/nim.cfg because build/ is below it. A Nim module name cannot
# contain '-', so the binary is built as formers_cli and renamed.
build: $(ROOT)/$(BYBYLANG)
	cd $(ROOT) && mkdir -p $(HERE)/build && \
	  $(BYBYLANG) $(CLI_SRC) --aot=$(HERE)/build/formers_cli --quiet && \
	  cp $(HERE)/build/formers_cli $(CLI_BIN)
	@echo "[formers.n] built $(CLI_BIN)"
ifeq ($(shell uname -s),Darwin)
ifeq ($(FORMERS_NO_ARM64),)
	$(MAKE) -C $(ROOT)/$(HERE) build-arm64
endif
endif

# Apple Silicon binary built on an Intel Mac (formers-cli is x86_64 and runs under Rosetta there).
ARM64_BIN   := $(HERE)/formers-cli_arm64
build-arm64: $(ROOT)/$(BYBYLANG)
	cd $(ROOT) && mkdir -p $(HERE)/build-arm64 && \
	  BYBYLANG_NIMCACHE=$(ROOT)/nimcache/formers-n-arm64 \
	  BYBYLANG_NIM_FLAGS='--cpu:arm64 --passC:"-arch arm64" --passL:"-arch arm64"' \
	  $(BYBYLANG) $(CLI_SRC) --aot=$(HERE)/build-arm64/formers_cli --quiet && \
	  cp $(HERE)/build-arm64/formers_cli $(ARM64_BIN)
	@echo "[formers.n] built $(ARM64_BIN)"

test: $(ROOT)/$(BYBYLANG)
	cd $(ROOT) && mkdir -p $(HERE)/build && \
	  $(BYBYLANG) $(TEST_SRC) --aot=$(HERE)/build/test_detect --quiet && \
	  $(HERE)/build/test_detect

fir-test: fir-graph-test $(ROOT)/$(BYBYLANG)
	cd $(ROOT) && mkdir -p $(HERE)/build && \
	  $(BYBYLANG) $(FIR_TEST_SRC) --aot=$(HERE)/build/test_fir --quiet && \
	  $(HERE)/build/test_fir && \
	  $(BYBYLANG) $(HERE)/tests/test_fir_script.bybylang --aot=$(HERE)/build/test_fir_script --quiet && \
	  $(HERE)/build/test_fir_script && \
	  $(MAKE) -C $(HERE) build && \
	  $(CLI_BIN) --fir-check models/mamba3-1.5b/model.fir && \
	  for f in models/*/model.fir; do $(CLI_BIN) --fir-check $$f || exit 1; done

SMOKE_DIR := $(HERE)/build/tiny-llama

smoke: build
	cd $(ROOT) && \
	  $(BYBYLANG) $(HERE)/tests/make_tiny_llama.bybylang --aot=$(HERE)/build/make_tiny_llama --quiet && \
	  $(HERE)/build/make_tiny_llama $(SMOKE_DIR) && \
	  $(CLI_BIN) --models $(SMOKE_DIR) -p "def fib(n):" -n 8 --temp 0 --no-progress

clean:
	rm -rf $(ROOT)/$(HERE)/build $(ROOT)/$(CLI_BIN) $(BYBYLANG_NIMCACHE)

fir-graph-test: $(ROOT)/$(BYBYLANG)
	cd $(ROOT) && mkdir -p $(HERE)/build && \
	  $(BYBYLANG) models/gpt2-125m/test_model.bybylang --aot=$(HERE)/build/test_fir_graph --quiet && \
	  $(HERE)/build/test_fir_graph

# Trace recorder, lifter, library numeric test and builder round trip.
fir-lift-test: build
	cd $(ROOT) && nim c -r -d:release --hints:off --path:vendor/bybylang \
	  --nimcache:nimcache/fir-rope-upload-test -o:$(HERE)/build/test_rope_uploads $(HERE)/firlift/test_rope_uploads.nim
	mkdir -p $(FIR_LIFT_TMP)
	cd $(ROOT) && nim c -d:release --hints:off --path:vendor/bybylang \
	  --nimcache:nimcache/firlift -o:$(HERE)/firlift/firlift $(HERE)/firlift/firlift_main.nim
	cd $(ROOT) && nim c -d:release --hints:off --path:vendor/bybylang \
	  --nimcache:nimcache/fir-library-test -o:$(HERE)/firlift/test_library $(HERE)/firlift/test_library.nim
	cd $(ROOT) && $(HERE)/firlift/test_library
	cd $(ROOT) && nim c -d:release --hints:off --path:vendor/bybylang \
	  --nimcache:nimcache/fir-fuse-test -o:$(HERE)/firlift/test_fuse $(HERE)/firlift/test_fuse.nim
	cd $(ROOT) && $(HERE)/firlift/test_fuse
	cd $(ROOT) && BYBYLANG_NIMCACHE=$(ROOT)/nimcache/fir-raw $(BYBYLANG) $(HERE)/tests/test_fir_raw.bybylang --aot=$(HERE)/build/test_fir_raw --quiet && \
	  $(HERE)/build/test_fir_raw
	cd $(ROOT) && nim c -d:release --hints:off --mm:arc --path:vendor/bybylang --path:. --path:kernelfortest \
	  --nimcache:nimcache/session-gpu-test -o:$(HERE)/firlift/test_session_gpu $(HERE)/firlift/test_session_gpu.nim
	cd $(ROOT) && nim c -d:release --hints:off --path:vendor/bybylang \
	  --nimcache:nimcache/fir-backend-available -o:$(HERE)/build/backend_available $(HERE)/firlift/backend_available.nim
	# Every backend present (Metal, Vulkan/MoltenVK) records an op trace; each must lower completely.
	for b in metal vulkan; do \
	  cd $(ROOT) && $(HERE)/build/backend_available $$b || continue; \
	  cd $(ROOT) && rm -f $(FIR_LIFT_TMP)/session-ops-$$b.fprobe && \
	  BACKEND=$$b BYBY_FIR_PROBE=$(FIR_LIFT_TMP)/session-ops-$$b.fprobe BYBY_FIR_PROBE_STEPS=decode:1 $(HERE)/firlift/test_session_gpu || exit 1; \
	  if test -f $(FIR_LIFT_TMP)/session-ops-$$b.fprobe && grep -q '^op ' $(FIR_LIFT_TMP)/session-ops-$$b.fprobe; then \
	    $(HERE)/firlift/firlift --trace $(FIR_LIFT_TMP)/session-ops-$$b.fprobe --config $(HERE)/firlift/coverage_config.json \
	      --out $(FIR_LIFT_TMP)/session-ops-$$b --coverage || exit 1; fi; \
	done
	cd $(ROOT) && nim c -d:release --hints:off --path:vendor/bybylang \
	  --nimcache:nimcache/builder-test -o:$(HERE)/firlift/builder_test $(HERE)/firlift/builder_test.nim
	cd $(ROOT) && $(HERE)/firlift/builder_test
	cd $(ROOT) && $(CLI_BIN) --fir-check $(FIR_LIFT_TMP)/builder/model.fir
	rm -rf $(FIR_LIFT_TMP)/builder $(FIR_LIFT_TMP)/session-ops-metal $(FIR_LIFT_TMP)/session-ops-vulkan
	rm -f $(FIR_LIFT_TMP)/session-ops-metal.fprobe $(FIR_LIFT_TMP)/session-ops-vulkan.fprobe

# JSON-lines diagnostic runner; the Python quality check keeps all output in memory.
fir-probe: $(ROOT)/$(BYBYLANG)
	cd $(ROOT) && mkdir -p $(HERE)/build && \
	  $(BYBYLANG) models/gpt2-125m/probe.bybylang --aot=$(HERE)/build/fir_probe --quiet
