# Makefile — build project Nimformer

NIM      := nim
NIMFLAGS := -d:release --hints:off

TEST_SRC  := test_nimformer.nim

# ── Nhận diện OS ──
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

.PHONY: all build run run-cpu run-metal run-opencl run-tsic clean

all: build run

build:
	$(NIM) c $(NIMFLAGS) $(TEST_SRC)

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

run-opencl:
	$(NIM) c $(NIMFLAGS) -d:backend=opencl $(TEST_SRC)
	$(RUN_PREFIX)$(TEST_BIN)

run-tsic:
	$(NIM) c $(NIMFLAGS) -d:backend=tsic $(TEST_SRC)
	$(RUN_PREFIX)$(TEST_BIN)

clean:
ifeq ($(IS_WINDOWS),yes)
	-if exist test_nimformer.exe $(RM_F) test_nimformer.exe
	-if exist nimcache $(RM_RF) nimcache
	-del /Q /F model_*.nimq 2>nul
else
	$(RM_F) test_nimformer
	$(RM_F) test_nimformer.exe
	$(RM_RF) nimcache
	$(RM_F) model_*.nimq
endif