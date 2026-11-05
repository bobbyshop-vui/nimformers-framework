# Run from repository root: make -f test/mamba3/Makefile <target>
NIM ?= nim
FLAGS := -d:release --hints:off
OUT := test/mamba3
.PHONY: all train infer quantize smoke
all: train infer quantize
train:
	$(NIM) c $(FLAGS) --nimcache:nimcache/mamba3-train -o:$(OUT)/finetune_lora_bin $(OUT)/finetune_lora.nim
infer:
	$(NIM) c $(FLAGS) -d:codestralInference --nimcache:nimcache/mamba3-infer -o:$(OUT)/infer_lora_bin $(OUT)/infer_lora.nim
quantize:
	$(NIM) c $(FLAGS) --nimcache:nimcache/mamba3-quant -o:$(OUT)/quantize_mamba3_bin $(OUT)/quantize_mamba3.nim
smoke:
	$(NIM) c -r $(FLAGS) -d:mamba3SmokeLora --nimcache:nimcache/mamba3-smoke -o:$(OUT)/smoke_bin $(OUT)/smoke.nim
