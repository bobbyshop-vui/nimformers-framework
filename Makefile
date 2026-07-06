# Makefile — build project Nimformer
#
#   make            -> build HẾT (test_nimformer + metal_ai) rồi chạy test_nimformer
#   make build      -> build HẾT, không chạy
#   make run        -> build + chạy test_nimformer
#   make metal      -> build + chạy riêng metal_ai
#   make clean      -> xoá file build + checkpoint .nimq

NIM      := nim
NIMFLAGS := -d:release --hints:off

TEST_SRC  := test_nimformer.nim
METAL_SRC := metal_ai.nim

.PHONY: all build run metal clean

all: build run

build:
	$(NIM) c $(NIMFLAGS) $(TEST_SRC)
	$(NIM) c $(NIMFLAGS) $(METAL_SRC)

run:
	$(NIM) c $(NIMFLAGS) $(TEST_SRC)
	./test_nimformer

metal:
	$(NIM) c $(NIMFLAGS) $(METAL_SRC)
	./metal_ai

clean:
	rm -f test_nimformer metal_ai
	rm -f test_nimformer.exe metal_ai.exe
	rm -rf nimcache
	rm -f model_*.nimq