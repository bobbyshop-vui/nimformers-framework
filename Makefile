# Makefile — build project Nimformer

NIM      := nim
NIMFLAGS := -d:release --hints:off --mm:arc --opt:speed
INFER_LORA_NIMCACHE := nimcache/infer-lora
NIM_INFER_NIMCACHE := nimcache/nim-inference

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

# ── Cac binary lien quan Mamba: graft / finetune (LoRA) / inference ──
GRAFT_SRC       := graft_attention.nim
FINETUNE_SRC    := finetune_lora.nim
INFER_LORA_SRC  := infer_lora.nim
NIM_INFER_SRC   := nim_inference.nim
SPECTRON_SRC    := test_inferencespectron.nim

GRAFT_BIN       := graft_attention_bin$(EXE_EXT)
FINETUNE_BIN    := finetune_lora_bin$(EXE_EXT)
INFER_LORA_BIN  := infer_lora_bin$(EXE_EXT)
COMPACT_LORA_BIN := compact_lora_checkpoint$(EXE_EXT)
MIGRATE_LORA_BIN := migrate_lora_json_to_bin$(EXE_EXT)
NIM_INFER_BIN   := nim_inference$(EXE_EXT)
SPECTRON_BIN    := test_inferencespectron$(EXE_EXT)

.PHONY: shaders all build run run-cpu run-metal run-opencl run-tsic clean \
        build-graft build-finetune build-infer-lora build-nim-inference build-spectron \
        graft finetune-mamba infer-lora infer-mamba infer-deepseek run-spectron mamba-pipeline

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
run-cuda:
	nim c -d:release --hints:off -d:backend=cuda test_nimformer.nim
	./test_nimformer$(EXE_EXT)

# ══════════════════════════════════════════════════════════════════
# Mamba: graft (cay Attention vao checkpoint Mamba-only) / finetune LoRA /
# inference. Moi target BUILD RoI TU DONG CHAY LUON voi mac dinh hop ly neu
# khong truyen ARGS - luon in log truoc khi chay (dang dung gia tri gi,
# mac dinh hay ARGS) va in huong dan argument neu file can thiet chua co.
# ══════════════════════════════════════════════════════════════════

# Mac dinh cho tung target (ghi de qua bien moi truong hoac make graft ARGS="...")
# GRAFT_DEFAULT_OLD = GRAFT_DEFAULT_OUT = codestral_mamba_7b_fp4.nimq: day la
# CHU DINH - file .nimq nay la output cua quantize_codestral.nim (xem usage
# doc trong file do) VA DONG THOI la CFG_NIMQ_PATH_DEFAULT ma finetune_lora.nim
# doc lai (finetune-mamba mac dinh khong argv). Graft ghi DE LEN CHINH file
# do (doc toan bo vao RAM truoc, moi ghi lai sau - an toan voi jamba_hybrid.
# loadCheckpoint/saveCheckpoint hien tai) de finetune-mamba luon thay dung
# file "da cay" ma khong can doi ten. Neu ban muon giu ban goc, truyen ARGS
# voi ten out khac.
GRAFT_DEFAULT_OLD     := codestral_mamba_only.nimq
GRAFT_DEFAULT_PATTERN := $(shell python3 -c "print('MMMA'*16)" 2>/dev/null || echo MMMA)
GRAFT_DEFAULT_OUT     := codestral_mamba_only.nimq

INFER_LORA_DEFAULT_CKPT  := jamba_parallel_lora.bin
# SUA: template "[INST] ...[/INST] " dung cho Codestral (Mistral tokenizer V3,
# sentencepiece) - KHONG phai ChatML "<|user|>/<|assistant|>" (xem
# finetune_lora.nim/lineToText va infer_lora.nim). Giu khoang trang sau
# "[/INST]" cho khop phan phoi luc train (xem comment trong infer_lora.nim).
INFER_LORA_DEFAULT_PROMPT:= [INST] Xin chao[/INST] 
INFER_LORA_DEFAULT_TEMP  := 0.7
# THEM: infer_lora.nim (nhanh fallback "khong co checkpoint LoRA") van load
# tokenizer THAT qua newHfTokenizer(hfDir) VA doc trong so goc qua nimqPath -
# 2 gia tri nay TRUOC DAY khong duoc Makefile truyen, nen se roi vao
# CFG_HF_DIR_DEFAULT/CFG_NIMQ_PATH_DEFAULT hard-code RIENG trong infer_lora.nim
# ("/Volumes/Data 1/..." - KHAC voi "Data 2" cua finetune_lora.nim/GRAFT_DEFAULT_*
# o tren) - dat lai o day cho KHOP voi phan con lai cua pipeline nay.
INFER_LORA_DEFAULT_HFDIR := /Volumes/Data 2/models/Mamba-Codestral-7B-v0.1
INFER_LORA_DEFAULT_NIMQ  := $(GRAFT_DEFAULT_OUT)

# infer-deepseek (nim_inference.nim) KHONG lien quan pipeline Mamba/LoRA -
# la inference rieng cho 1 model GPTQ khac (deepseek-coder). Backend mac
# dinh la "auto" giong newBackend() khong tham so dung khap noi trong repo.
INFER_DEEPSEEK_DEFAULT_MODEL   := TheBloke/deepseek-coder-6.7B-instruct-GPTQ
INFER_DEEPSEEK_DEFAULT_NIMQ    := model.nimq
INFER_DEEPSEEK_DEFAULT_BACKEND := auto

build-graft:
	$(NIM) c $(NIMFLAGS) -o:$(GRAFT_BIN) $(GRAFT_SRC)

build-finetune:
	$(NIM) c $(NIMFLAGS) -o:$(FINETUNE_BIN) $(FINETUNE_SRC)

# Infer defaults: independent decode kernels in one session + resident GPU
# prefill (R/P/G). One-dispatch is opt-in after the Iris 640 end-to-end A/B.
# Compare paths with: make infer-lora BYBY_ONE_DISPATCH=1 /
# BYBY_PREFILL_RESIDENT=0 /
# BYBY_GEMV_VARIANT=0 / BYBY_PREFILL_PARALLEL=0 / BYBY_MAMBA_SCAN_PARALLEL=0 /
# BYBY_LP_SCAN_SERIAL=1 / BYBY_FP4_PAD_M=0.
# Gop LoRA vao fp4 lam token rac -> TAT. LoRA chay rieng.
export BYBY_MERGE_LORA ?= 0
export BYBY_ONE_DISPATCH ?= 0
export BYBY_PREFILL_RESIDENT ?= 1
export BYBY_GEMV_VARIANT ?= auto
export BYBY_PREFILL_FP4_KERNEL ?= auto
export BYBY_GPU_AUTOTUNE ?= 1
export BYBY_VULKAN_TENSOR_CORES ?= 1
export BYBY_MERGE_ALL_LAYERS_SESSION ?= 1
export BYBY_MERGE_QKV_ATTN_SESSION ?= 1
export BYBY_DISABLE_KV_RESIDENT ?= 0
export BYBY_DECODE_PROFILE ?= 1
export BYBY_PREFILL_PROFILE ?= 1
export BYBY_INFER_VERBOSE ?= 1
export BYBY_DEBUG_SAMPLING ?= 1
export BYBY_G_DIAGNOSTICS ?= 0
export BYBY_TRACE_BLOCKS ?= 0
export BYBY_MACOS_HIGH_PRIORITY ?= 1

# One-dispatch duoc build mac dinh; -d:noOneDispatch de bo han khoi binary.
# Keep the Metal extension reproducible even when bybylang is a fresh submodule.
.PHONY: prepare-resident-metal
prepare-resident-metal:
	@if ! grep -q 'proc metalSessionMatmulMpsNTEnc' vendor/bybylang/backends/metal/metal_backend.nim; then \
		patch --dry-run -d vendor/bybylang -p1 < patches/bybylang-resident-mps.patch >/dev/null && \
		patch -d vendor/bybylang -p1 < patches/bybylang-resident-mps.patch; \
	fi

.PHONY: prepare-vulkan-inference
prepare-vulkan-inference: prepare-resident-metal
	@if ! grep -q 'proc vulkanTensorCoreAvailable' vendor/bybylang/backends/vulkan/vulkan_driver.nim; then \
		patch --dry-run -d vendor/bybylang -p1 < patches/bybylang-inference-gpu.patch >/dev/null && \
		patch -d vendor/bybylang -p1 < patches/bybylang-inference-gpu.patch; \
	fi

build-infer-lora: prepare-vulkan-inference
	$(NIM) c $(NIMFLAGS) --nimcache:$(INFER_LORA_NIMCACHE) -o:$(INFER_LORA_BIN) $(INFER_LORA_SRC)

build-compact-lora:
	$(NIM) c $(NIMFLAGS) -o:$(COMPACT_LORA_BIN) compact_lora_checkpoint.nim

build-migrate-lora:
	$(NIM) c $(NIMFLAGS) -o:$(MIGRATE_LORA_BIN) migrate_lora_json_to_bin.nim

build-nim-inference: prepare-vulkan-inference
	$(NIM) c $(NIMFLAGS) --nimcache:$(NIM_INFER_NIMCACHE) -o:$(NIM_INFER_BIN) $(NIM_INFER_SRC)

build-spectron:
	$(NIM) c $(NIMFLAGS) -o:$(SPECTRON_BIN) $(SPECTRON_SRC)

# ── graft: build + TU DONG CHAY. Khong truyen ARGS -> dung mac dinh
# GRAFT_DEFAULT_*. Neu file old.bin/.nimq mac dinh chua ton tai, IN HUONG
# DAN roi dung (khong chay bua). ──
# vd tuy chinh: make graft ARGS="checkpoint.bin MMAMMSM out.bin"
graft: build-graft
ifeq ($(strip $(ARGS)),)
	@echo "[graft] Khong truyen ARGS -> dung mac dinh:"
	@echo "[graft]   old     = $(GRAFT_DEFAULT_OLD)"
	@echo "[graft]   pattern = $(GRAFT_DEFAULT_PATTERN)"
	@echo "[graft]   out     = $(GRAFT_DEFAULT_OUT)"
	@if [ "$(GRAFT_DEFAULT_OLD)" = "$(GRAFT_DEFAULT_OUT)" ]; then \
		echo "[graft] LUU Y: out trung ten voi old (chu dinh, xem comment GRAFT_DEFAULT_* trong Makefile) - $(GRAFT_DEFAULT_OLD) se bi GHI DE sau khi cay xong."; \
	fi
	@if [ ! -f "$(GRAFT_DEFAULT_OLD)" ]; then \
		echo "[graft] LOI: khong thay $(GRAFT_DEFAULT_OLD)."; \
		echo "[graft] File nay phai co truoc (output cua quantize_codestral_bin <hf_dir> $(GRAFT_DEFAULT_OLD))."; \
		echo "[graft] Huong dan: make graft ARGS=\"<old.bin> <pattern vd MMAMMSM> <out.bin>\""; \
		echo "[graft]         hoac: make graft ARGS=\"--from-hf <hf.bin> <vocab> <embedDim> <nHeads> <ffMult> <nLayers> <pattern> <out.bin>\""; \
		exit 1; \
	fi
	@echo "[graft] Dang chay: $(RUN_PREFIX)$(GRAFT_BIN) $(GRAFT_DEFAULT_OLD) $(GRAFT_DEFAULT_PATTERN) $(GRAFT_DEFAULT_OUT)"
	$(RUN_PREFIX)$(GRAFT_BIN) $(GRAFT_DEFAULT_OLD) $(GRAFT_DEFAULT_PATTERN) $(GRAFT_DEFAULT_OUT)
else
	@echo "[graft] Dang chay voi ARGS truyen vao: $(RUN_PREFIX)$(GRAFT_BIN) $(ARGS)"
	$(RUN_PREFIX)$(GRAFT_BIN) $(ARGS)
endif

# ── finetune-mamba: finetune_lora.nim GIO DOC DUOC ARGV (tuy chon), fallback
# ve config cung (CFG_*_DEFAULT trong finetune_lora.nim) neu khong truyen gi.
# Thu tu argv: initMode oldCheckpoint hfDir hfSource nimqPath outCheckpointPath ──
# vd tuy chinh: make finetune-mamba ARGS="codestral - - nimq codestral_mamba_7b_fp4.nimq my_lora.bin"
# (dung "-" cho tham so muon giu mac dinh nhung van can set tham so sau no)
finetune-mamba: build-finetune
ifeq ($(strip $(ARGS)),)
	@echo "[finetune-mamba] Khong truyen ARGS -> dung DUNG config cung trong finetune_lora.nim (CFG_*_DEFAULT)."
	@echo "[finetune-mamba] Dang chay: $(RUN_PREFIX)$(FINETUNE_BIN)"
	$(RUN_PREFIX)$(FINETUNE_BIN)
else
	@echo "[finetune-mamba] Dang chay voi ARGS truyen vao: $(RUN_PREFIX)$(FINETUNE_BIN) $(ARGS)"
	$(RUN_PREFIX)$(FINETUNE_BIN) $(ARGS)
endif

# ── infer-lora: khong truyen ARGS -> dung mac dinh INFER_LORA_DEFAULT_*.
# Neu checkpoint mac dinh chua ton tai, in huong dan roi dung. ──
# argv day du: ckptPath prompt temperature hfDir nimqPath (2 tham so cuoi MOI
# them - infer_lora.nim can ca hfDir (tokenizer that) LAN nimqPath (trong so
# goc, dung khi fallback ve Mamba thuan luc chua co checkpoint LoRA nao)).
# vd tuy chinh: make infer-lora ARGS="jamba_lora.bin '[INST] Xin chao[/INST] ' 0.7 '/path/hf' model.nimq"
infer-lora: build-infer-lora
ifeq ($(strip $(ARGS)),)
	@echo "[infer-lora] Khong truyen ARGS -> dung mac dinh:"
	@echo "[infer-lora]   checkpoint prefix (ten dat)  = $(INFER_LORA_DEFAULT_CKPT)"
	@echo "[infer-lora]   adapter LoRA (neu co)        = $(INFER_LORA_DEFAULT_CKPT)"
	@echo "[infer-lora]   prompt      = $(INFER_LORA_DEFAULT_PROMPT)"
	@echo "[infer-lora]   temperature = $(INFER_LORA_DEFAULT_TEMP)"
	@echo "[infer-lora]   hfDir       = $(INFER_LORA_DEFAULT_HFDIR)"
	@echo "[infer-lora]   nimqPath    = $(INFER_LORA_DEFAULT_NIMQ)"
	@if [ -f "$(INFER_LORA_DEFAULT_CKPT)" ]; then \
		echo "[infer-lora] Tim thay $(INFER_LORA_DEFAULT_CKPT) -> se dung LoRA da train."; \
	else \
		echo "[infer-lora] KHONG tim thay $(INFER_LORA_DEFAULT_CKPT) - infer_lora.nim se TU DONG"; \
		echo "[infer-lora] fallback ve Mamba THUAN tu base pretrain (khong loi, xem loadPureMambaForInfer)."; \
		echo "[infer-lora] Chay 'make finetune-mamba' truoc neu muon dung ban da fine-tune."; \
	fi
	@if [ ! -f "$(INFER_LORA_DEFAULT_NIMQ)" ]; then \
		echo "[infer-lora] CANH BAO: khong thay $(INFER_LORA_DEFAULT_NIMQ) - can 'make graft' (hoac quantize_codestral_bin) truoc."; \
	fi
	@echo "[infer-lora] Dang chay: $(RUN_PREFIX)$(INFER_LORA_BIN) $(INFER_LORA_DEFAULT_CKPT) \"$(INFER_LORA_DEFAULT_PROMPT)\" $(INFER_LORA_DEFAULT_TEMP) \"$(INFER_LORA_DEFAULT_HFDIR)\" $(INFER_LORA_DEFAULT_NIMQ)"
	$(RUN_PREFIX)$(INFER_LORA_BIN) "$(INFER_LORA_DEFAULT_CKPT)" "$(INFER_LORA_DEFAULT_PROMPT)" "$(INFER_LORA_DEFAULT_TEMP)" "$(INFER_LORA_DEFAULT_HFDIR)" "$(INFER_LORA_DEFAULT_NIMQ)"
else
	@echo "[infer-lora] Dang chay voi ARGS truyen vao: $(RUN_PREFIX)$(INFER_LORA_BIN) $(ARGS)"
	$(RUN_PREFIX)$(INFER_LORA_BIN) $(ARGS)
endif

# ── infer-mamba: ALIAS cua infer-lora - day moi la inference DUNG cho
# pipeline Mamba (graft -> finetune-mamba -> infer-mamba), dung checkpoint
# DA FINE-TUNE (base .bin + adapter .lora.json) qua infer_lora_bin, KHONG
# phai model GPTQ rieng biet. Neu ban muon inference model deepseek GPTQ
# khong lien quan Mamba, dung 'make infer-deepseek'. ──
infer-mamba: infer-lora

# ── infer-deepseek: inference qua nim_inference.nim - day la 1 model GPTQ
# (deepseek-coder) HOAN TOAN RIENG, KHONG dung chung pipeline graft/finetune
# Mamba. Backend mac dinh "auto" (giong newBackend() khong tham so dung
# khap repo - tu chon GPU co san, raise loi neu khong co GPU thay vi am
# tham roi ve CPU). ──
# vd tuy chinh: make infer-deepseek ARGS="TheBloke/deepseek-coder-6.7B-instruct-GPTQ model.nimq auto"
infer-deepseek: build-nim-inference
ifeq ($(strip $(ARGS)),)
	@echo "[infer-deepseek] Khong truyen ARGS -> dung mac dinh:"
	@echo "[infer-deepseek]   modelPath = $(INFER_DEEPSEEK_DEFAULT_MODEL)"
	@echo "[infer-deepseek]   nimqPath  = $(INFER_DEEPSEEK_DEFAULT_NIMQ)"
	@echo "[infer-deepseek]   backend   = $(INFER_DEEPSEEK_DEFAULT_BACKEND)"
	@if [ ! -f "$(INFER_DEEPSEEK_DEFAULT_NIMQ)" ]; then \
		echo "[infer-deepseek] CANH BAO: khong thay $(INFER_DEEPSEEK_DEFAULT_NIMQ) trong thu muc hien tai - se loi khi load weights."; \
		echo "[infer-deepseek] Huong dan: make infer-deepseek ARGS=\"<modelPath> <nimqPath> <backend> <prompt> <maxNewTokens> <temperature> <topP> <systemText>\""; \
	fi
	@echo "[infer-deepseek] Dang chay: $(RUN_PREFIX)$(NIM_INFER_BIN) $(INFER_DEEPSEEK_DEFAULT_MODEL) $(INFER_DEEPSEEK_DEFAULT_NIMQ) $(INFER_DEEPSEEK_DEFAULT_BACKEND)"
	$(RUN_PREFIX)$(NIM_INFER_BIN) "$(INFER_DEEPSEEK_DEFAULT_MODEL)" "$(INFER_DEEPSEEK_DEFAULT_NIMQ)" "$(INFER_DEEPSEEK_DEFAULT_BACKEND)"
else
	@echo "[infer-deepseek] Dang chay voi ARGS truyen vao: $(RUN_PREFIX)$(NIM_INFER_BIN) $(ARGS)"
	$(RUN_PREFIX)$(NIM_INFER_BIN) $(ARGS)
endif

run-spectron: build-spectron
	@echo "[run-spectron] Dang chay: $(RUN_PREFIX)$(SPECTRON_BIN) $(ARGS)"
	$(RUN_PREFIX)$(SPECTRON_BIN) $(ARGS)

# ── mamba-pipeline: chay toan bo finetunemamba.sh (bien env co san se
# tu dong SKIP_TO_FINETUNE neu da co checkpoint adaptive - xem script) ──
mamba-pipeline:
	@echo "[mamba-pipeline] Dang chay ./finetunemamba.sh (dat bien env truoc make de tuy chinh, vd: SKIP_TO_FINETUNE=1 make mamba-pipeline)"
	./finetunemamba.sh
# ── Bien dich toan bo shader Vulkan (.comp -> .spv). Can glslc (Vulkan SDK / brew install shaderc).
# Dung: make shaders   (them -B de ep build lai)
VK_KERNEL_DIR := vendor/bybylang/backends/vulkan/kernels
shaders:
	@command -v glslc >/dev/null || { echo "[shaders] Khong thay glslc - brew install shaderc (hoac cai Vulkan SDK)"; exit 1; }
	@for f in $(VK_KERNEL_DIR)/*.comp; do \
		out=$${f%.comp}.spv; \
		if [ ! -f "$$out" ] || [ "$$f" -nt "$$out" ]; then \
			echo "[shaders] glslc $$f"; glslc "$$f" -o "$$out" || exit 1; \
		fi; \
	done
	@echo "[shaders] xong."
