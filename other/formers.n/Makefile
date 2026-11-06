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

.PHONY: all build test fir-test smoke clean

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

test: $(ROOT)/$(BYBYLANG)
	cd $(ROOT) && mkdir -p $(HERE)/build && \
	  $(BYBYLANG) $(TEST_SRC) --aot=$(HERE)/build/test_detect --quiet && \
	  $(HERE)/build/test_detect

fir-test: $(ROOT)/$(BYBYLANG)
	cd $(ROOT) && mkdir -p $(HERE)/build && \
	  $(BYBYLANG) $(FIR_TEST_SRC) --aot=$(HERE)/build/test_fir --quiet && \
	  $(HERE)/build/test_fir && \
	  $(MAKE) -C $(HERE) build && \
	  $(CLI_BIN) --fir-check models/mamba3-1.5b/model.fir

SMOKE_DIR := $(HERE)/build/tiny-llama

smoke: build
	cd $(ROOT) && \
	  $(BYBYLANG) $(HERE)/tests/make_tiny_llama.bybylang --aot=$(HERE)/build/make_tiny_llama --quiet && \
	  $(HERE)/build/make_tiny_llama $(SMOKE_DIR) && \
	  $(CLI_BIN) --models $(SMOKE_DIR) -p "def fib(n):" -n 8 --temp 0 --no-progress

clean:
	rm -rf $(ROOT)/$(HERE)/build $(ROOT)/$(CLI_BIN) $(BYBYLANG_NIMCACHE)
