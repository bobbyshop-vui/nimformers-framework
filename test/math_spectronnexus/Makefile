NIM ?= nim
BUILD_DIR := nimcache
ifeq ($(OS),Windows_NT)
    EXE := .exe
endif
BIN := $(BUILD_DIR)/math_spectronnexus$(EXE)

OEIS_DIR := data/oeis

.PHONY: build train train-quality train-research train-explorer train-planner train-program eval-program eval data oeis-data bench-oeis bench-research

build:
	$(NIM) c --hints:off -d:release --mm:arc --opt:speed --nimcache:$(BUILD_DIR)/c --out:$(BIN) math_spectronnexus.nim

train: build
	./$(BIN) train $(ARGS)

train-quality: build
	./$(BIN) train --profile=quality $(ARGS)

train-research: build
	./$(BIN) train --profile=research $(ARGS)

train-explorer: build
	./$(BIN) train-explorer $(ARGS)

train-planner: build
	./$(BIN) train-planner $(ARGS)

train-program: build
	./$(BIN) train-program $(ARGS)

eval-program: build
	./$(BIN) eval-program $(ARGS)

eval: build
	./$(BIN) eval $(ARGS)

data: build
	./$(BIN) data $(ARGS)

# The offline OEIS index (CC BY-SA 4.0 data, kept out of git). Refresh by running it again.
oeis-data:
	mkdir -p $(OEIS_DIR)
	curl -fL --retry 3 -o $(OEIS_DIR)/stripped.gz https://oeis.org/stripped.gz
	curl -fL --retry 3 -o $(OEIS_DIR)/names.gz https://oeis.org/names.gz
	gunzip -f $(OEIS_DIR)/stripped.gz $(OEIS_DIR)/names.gz

# Far-term benchmark for research_sequence. Compiled, with AVX2 on x86_64.
BENCH_BIN := $(BUILD_DIR)/research_benchmark$(EXE)
ifeq ($(shell uname -m),x86_64)
    BENCH_SIMD := --passC:-mavx2 --passC:-mfma --passL:-mavx2
endif

bench-research: build
	$(NIM) c --hints:off -d:release --mm:arc --opt:speed $(BENCH_SIMD) --nimcache:$(BUILD_DIR)/c_bench --out:$(BENCH_BIN) research_benchmark.nim
	./$(BENCH_BIN) "$(CURDIR)" $(ARGS)

# Explorer accuracy on 70 classic OEIS entries (needs oeis-data).
bench-oeis: build
	python3 oeis_benchmark.py "$(CURDIR)"
