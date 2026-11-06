# formers.n

A llama.cpp-style inference framework written in **BybyLang** (`.bybylang`, compiled ahead
of time to Nim) on top of nimformers. One command, `formers-cli`, detects the model in
`--models` and runs DeepSeek/Llama, Codestral (Mamba-2), Mamba-3 SISO, or any self-contained FIR graph model (`model.fir` with `runtime graph`), with optional
Mamba LoRA/Linear++ checkpoints.

The Llama path runs **exactly what `nim_inference.nim` runs**:
`newLlamaLConfig` (every size read from `config.json`) → `newLlamaModelL` →
`loadLlamaWeightsL` (mmap, Q4 kept packed) → `uploadAllResidentWeights` (zero-copy resident
Q4 + autotune) → `forwardStep` (decode with every layer in one GPU session, automatic
fallback, thermal yield). A single prompt calls `nim_inference.generate` itself; with
`--temp 0`, `formers-cli` produces the same tokens as `./nim_inference`. Chat steps the same
`forwardStep` over one KV cache kept across turns.

The Mamba-2 path imports the Mamba-2/Linear++ blocks and GPU kernels (`jamba_hybrid`,
`lora`, `infer_lora.forwardPrefillBatch`), the `.nimq`/`.bin` loaders and the tokenizer.
Everything else is formers.n's own, with sizes taken from `config.json`: model assembly,
LoRA loading, sampler, generation loop.

The Mamba-3 SISO path reuses `test/mamba3`'s quantized base/LoRA loader, recurrent GPU
runtime and tokenizer/generation implementation. It supports the official 187M and
1.5B SISO dimensions, one-shot generation and `-cnv` chat (chat template from the model
directory, same loop as the FIR graph models).

## FIR (model-side IR)

`--fir <file.fir>` makes the Mamba-3 resident path take its micro-kernel programs from a
FIR file (`models/mamba3-1.5b/model.fir`) through a registry, instead of the built-in
builders; output is identical. `--fir-check <file.fir>` resolves every kernel name and
builds every micro program without loading weights. Details: [`docs/fir.md`](../../docs/fir.md)
and [`models/FIR.md`](../../models/FIR.md). Tests: `make fir-test` (or `make formers-fir-test` at
the repository root).

## Layout

```
other/formers.n/
├── Makefile, nim.cfg            # AOT build; nim.cfg adds --path for the generated Nim
├── src/
│   ├── formers_cli.bybylang     # entry: options -> detect -> router
│   └── formers/
│       ├── cli/options.bybylang       # llama.cpp-style flags
│       ├── io/headers.bybylang        # .nimq / LoRA .bin header readers (payloads skipped)
│       ├── io/model_files.bybylang    # finds .nimq, config.json, tokenizer, LoRA in --models
│       ├── arch/detect.bybylang       # architecture detection + cross-checked sizes
│       ├── sampling/sampler.bybylang  # temp / top-p / min-p / repeat penalty
│       ├── runtime/signals.bybylang   # Ctrl+C, tok/s in the window title
│       ├── runtime/console.bybylang   # "> " chat input
│       ├── models/llama.bybylang      # Llama runner (nim_inference's own functions)
│       ├── models/mamba2.bybylang     # Mamba-2 base / LoRA / Linear++ runner
│       ├── models/mamba3.bybylang     # Mamba-3 SISO base / LoRA / Linear++ runner
│       ├── ir/                        # FIR: parser, types, kernel registry (fir_*.bybylang)
│       └── router.bybylang            # detection -> runner (false when unsupported)
└── tests/
    ├── test_detect.bybylang     # fake configs/tensors + real headers, no weights loaded
    ├── test_fir.bybylang        # FIR parse/registry + Mamba-3 kernels on the CPU interpreter
    └── make_tiny_llama.bybylang # random 2-layer Llama for a GPU smoke test
```

## Python API

The pure standard-library Python 3.9+ wrapper is in [`python/`](python/). It invokes
`formers-cli` for generation and keeps one `-cnv` process alive for multi-turn chat.
Build the CLI with `make -C other/formers.n build`, then use:

```python
from formers import Model

model = Model("path/to/models", backend="auto")
text = model.generate("Explain a hash table", max_tokens=128)
with model.chat() as chat:
    reply = chat.send("Hello")
```

Pass `cli_path=` or set `FORMERS_CLI` to select the executable. Python tests use a fake
CLI and run without loading model weights: `cd other/formers.n/python && python3 -m unittest`.

### OpenAI-compatible server

Run the standard-library HTTP server with `python3 -m formers.server --models path/to/models`.
It exposes `/v1/chat/completions` and `/v1/models`:

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"path/to/models","messages":[{"role":"user","content":"Hello"}]}'
```

## Detection and routing

| Tensors in the `.nimq` + `config.json` | arch | Route |
|---|---|---|
| `model.layers.N.self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`, `model_type` llama/mistral | `llama` | `nim_inference.nim` |
| `backbone.layers.N.mixer.{A_log,in_proj,conv1d,dt_bias,...}`, `model_type` mamba2 | `mamba2` | Mamba-2 base (the base-Mamba path of `infer_lora.nim`) |
| Mamba-2 + LoRA `.bin` with adapters only | `mamba2+lora` | LoRA checkpoint (like `infer_lora.nim`) |
| Mamba-2 + LoRA `.bin` with P/G/A blocks (`newBlock.blockN.attn.*`) | `mamba2+linear++` | same, keeping the checkpoint's `normPreset` |
| Mamba-3 SISO tensors + official 187M or 1.5B config | `mamba3` | `test/mamba3` resident inference |
| Mamba-3 SISO + LoRA `.bin` (R/P/G blocks) | `mamba3` | `test/mamba3` inference with the checkpoint |
| F32 safetensors + `model.fir` declaring `runtime graph` | `fir` | generic FIR graph executor |
| anything else (gemma, qwen2, Mamba-1, ...) | `unknown` | not run, `fmRun` returns `false`, exit 3 |

Detection checks the layer count, hidden size, head and KV-head counts, head_dim, FFN size,
vocab, the Mamba-2 `in_proj`/`conv1d` sizes, Mamba-3 SISO projection/MLP tensor shapes and the LoRA pattern/sizes against the real
shape of every tensor; any mismatch is rejected with its reason. For Llama it also checks
that `tokenizer.json` is a Hugging Face tokenizer (older `export_to_nimq.py` versions
overwrote it).

## Benchmarks

Intel Iris Plus 640 and Tesla T4 results per model: [DeepSeek Coder](../../test/deepseek-inference/benchmark.md) ·
[Codestral Mamba](../../test/codestral/benchmark.md) · [Mamba-3](../../test/mamba3/benchmark.md) ·
[GPT-2 125M via FIR](../../models/gpt2-125m/benchmark.md) · [TinyLlama](../../models/tinyllama/benchmark.md)

## Usage

```bash
make build-formers          # -> other/formers.n/formers-cli
make formers-test           # detection test (no GPU, no weights loaded)
make formers-fir-test       # FIR test (no GPU, no weights loaded)
make formers-smoke          # tiny random Llama end to end on the GPU
```

```bash
other/formers.n/formers-cli --models test/deepseek-inference/models -p "Write quicksort in Python" -n 128
other/formers.n/formers-cli --models test/codestral/models -p "Hello" -n 60 --temp 0.7
other/formers.n/formers-cli --models test/codestral/models --no-lora -p "Hello"
# Mamba-3 187M + gated LoRA (one-shot; add -cnv for chat)
other/formers.n/formers-cli --models test/mamba3/models -m test/mamba3/models/mamba3_siso_187m.nimq --config test/mamba3/models/mamba3-siso-187m/config.json --tokenizer test/mamba3/models/mamba3-siso-187m --lora test/mamba3/models/runs/mamba3-siso-187m-gated/lora.bin -p 'Write a Python function to check if a string is a palindrome' -n 64 --temp 0.7 --backend metal

# Mamba-3 1.5B + gated LoRA
other/formers.n/formers-cli --models test/mamba3/models -m test/mamba3/models/mamba3_siso_1.5b.nimq --config test/mamba3/models/mamba3-siso-1.5b/config.json --tokenizer test/mamba3/models/mamba3-siso-1.5b --lora test/mamba3/models/runs/mamba3-siso-1.5b-gated/lora-g1.bin -p 'Write a Python function to check if a string is a palindrome' -n 64 --temp 0.7 --backend metal

# Inspect detection without loading weights or generating
other/formers.n/formers-cli --models test/mamba3/models -m test/mamba3/models/mamba3_siso_187m.nimq --config test/mamba3/models/mamba3-siso-187m/config.json --tokenizer test/mamba3/models/mamba3-siso-187m --lora test/mamba3/models/runs/mamba3-siso-187m-gated/lora.bin --detect-only

# Run a base model without its LoRA
other/formers.n/formers-cli --models test/mamba3/models -m test/mamba3/models/mamba3_siso_187m.nimq --config test/mamba3/models/mamba3-siso-187m/config.json --tokenizer test/mamba3/models/mamba3-siso-187m --no-lora -p 'Write a Python function to check if a string is a palindrome' -n 64 --temp 0.7 --backend metal
other/formers.n/formers-cli --models <dir> --detect-only
```

### Chat (like `llama-cli -cnv`)

```bash
other/formers.n/formers-cli --models test/deepseek-inference/models -cnv
other/formers.n/formers-cli --models test/codestral/models -cnv -p "You are a coding assistant"
```

```
> write a fibonacci function
(reply)
> add memoization
(reply, the previous turn is still in context)
```

The model is loaded once and the context is kept between turns. In chat `-p` is the system
prompt and `-n` limits each reply (default 512). End a line with `\` to continue on the next
line; `/exit` or end of input quits. Ctrl+C during a reply stops it, Ctrl+C at the prompt
quits.

- Llama: the KV cache holds `-c` tokens (default 1024) and is resident on the GPU for every
  layer, **allocated once before prefill** like `nim_inference.generate`. It takes about
  2 × layers × kv_heads × ctx × head_dim × 8 bytes (GPU + RAM): about 2 GiB per 1024 tokens
  for DeepSeek 6.7B. On a 16 GB machine whose iGPU shares the RAM, a large `-c` pushes the
  mmapped weights out and tok/s drops sharply. Each turn tokenizes and feeds only the new
  chat-template text (end of the previous reply + the user message + assistant header),
  like llama.cpp's `common_chat_format_single`. Every turn is a prefill: the flash prefill
  attends to the cached turns too (with `BYBY_FLASH_PREFILL=0`, or if it fails once, later
  turns go through `forwardStep` decode instead). A full context is reported and the chat
  stops.
- Mamba-2: the recurrent state lives on the GPU, there is no context limit. Each turn feeds
  `</s>` (when the previous reply was cut short) + `[INST] ...[/INST]`.
- Mamba-3 and FIR graph models (`runtime graph` in `model.fir`): a template-driven loop in
  `models/mamba2` (`fmTemplateChatLoop`). The prompt is the model's chat template (`chat_template`
  in `tokenizer_config.json` or a `chat_template.jinja` next to it); a missing template is an
  error before the weights load. Mamba-3 keeps its recurrent state on the GPU; a graph model
  keeps its KV cache, so the context is `positions` (limited by `-c`). Templates ship in the
  GPT-2 and Mistral-style model folders (`chat_template.jinja`).

Chat templates that do not emit `bos_token` (DeepSeek-coder, TinyLlama) get BOS prepended
when the tokenizer adds one, like llama.cpp.

While generating, tok/s is shown in the **terminal window title**, never in the scrollback
(a line pinned with cursor moves used to overwrite scrolled log lines). `--no-progress`
turns it off.

### Startup

Llama chat: load the weights, resident upload + Q4 autotune, allocate the KV cache, then a
**short warmup** (a 4-token prefill and 2 decode steps create the prefill/decode pipelines,
then the KV cache is reset). `ready: warmup N s (decode ~M ms/tok)` is printed before the
`> ` prompt; `--no-warmup` skips it. A single prompt has no warmup, like `nim_inference`.

Mamba-2 warms up with decode steps until the step time settles. The resident parallel
prefill JIT-compiles its scan kernels for each chunk length, so only whole chunks of
`BYBY_PREFILL_BATCH_SIZE` tokens (default 128) go through it; the rest of a prompt, and every
later chat turn, runs on the decode kernels the warmup already compiled. Prompts shorter
than one chunk therefore compile nothing after `ready`; the first prompt with a full chunk
compiles the chunk kernels once and later ones reuse them.

On a Vulkan dGPU with tensor cores (`prefill=tensor-cores-only` in the device line;
`infer_lora`'s cooperative routes then take every GEMM from 2 rows up), prefill beats
decode even for a few tokens, so the remainder of a prompt also goes through 16-token
chunks and later turns prefill as well, with `forwardPrefillBatch(continued = true)`
carrying the chat state in. The warmup runs one 16-token prefill, so these kernels
compile before `ready` too: only the 128- and 16-token lengths ever compile. With
`BYBY_ONE_DISPATCH=1` the decode state lives apart from the prefill's, so later turns
decode instead.

### Logs

Quiet by default: the loader, LoRA, JIT, Metal and kernel-tuning logs of the libraries
(which write to fd 1/2 directly) go to `/dev/null`, and only formers.n's own lines show:
one `loading` line, `ready`, the generated text, the statistics and errors. `-v/--verbose`
shows everything, as before. An error in quiet mode prints its message and suggests
`--verbose` for the full log. Use `--verbose` to see Mamba-3 runtime statistics; set
`BYBY_METAL_JIT_LOG=0` to suppress Metal JIT messages in verbose output.

A Mamba-2 LoRA is merged into the resident FP4-KM weights at startup
(`infer_lora.mergeResidentLoraIfEnabled`, ~35 s on Codestral 7B): the decode kernels then
drop the A/B fusion and the answers match a merged checkpoint. `--no-merge-lora`
(or `BYBY_MERGE_LORA=0`) keeps the adapters separate and fused in the kernels; the merge
also steps aside by itself when `BYBY_ONE_DISPATCH=1`, `BYBY_MODEL_MMAP=0` or the process
has no memory headroom for it.

Flags: `--models`, `-m/--model`, `--lora`, `--no-lora`, `--no-merge-lora`, `--config`, `--tokenizer`,
`-p/--prompt`, `-f/--file`, `--system`, `--raw-prompt`, `-n/--n-predict`, `--temp`,
`--top-p`, `--min-p`, `--repeat-penalty`, `--repeat-last-n`, `-s/--seed`, `--backend`,
`-ngl/--ngl/--n-gpu-layers`, `-cnv/--conversation`, `-i/--interactive`, `-c/--ctx-size`,
`--no-progress`, `--no-warmup`, `-v/--verbose`, `--detect-only`, `--fir`, `--fir-f32`, `--fir-check`, `-h`. Exit codes: 0 ok, 1 runtime error,
2 bad options, 3 unsupported model.

## Module API

Every module is BybyLang (`src/formers/**.bybylang`); `formers_cli.bybylang` only parses the
options and calls `fmRun`.

| Module | Functions |
| --- | --- |
| `cli/options` | `fmDefaultOptions`, `fmParseOptions` (argv → options, exit code 2 on a bad flag), `fmUsage`, `fmOptValue`, `fmToInt`, `fmToFloat` |
| `io/model_files` | `fmFindModelFiles` (weights, LoRA, config and tokenizer from `--models`), `fmSearchDirs`, `fmPickNimq`, `fmFilesWithExt`, `fmHasTokenizer` |
| `io/headers` | `fmReadNimqHeader`, `fmReadLoraHeader`, `fmIsLoraBin`, `fmLayerCount`, `fmShapeIndex`, `fmReadI32`, `fmReadStr`, `fmSkip` (headers only, no tensor data) |
| `arch/detect` | `fmParseConfig`, `fmModelType`, `fmLooksLlama`, `fmLooksMamba2`, `fmExpectShape`, `fmCheckHfTokenizer`, `fmJInt`, `fmJFloat`, `fmJRequired` |
| `router` | `fmRun`: detection → `fmRunLlama` or `fmRunMamba2`, exit code 3 for an unsupported model |
| `models/llama` | `fmRunLlama`, `fmLlamaLoad`, `fmLlamaNewChat`, `fmLlamaReset`, `fmLlamaFeed` (flash prefill of a turn, decode if it fails), `fmLlamaWarmup`, `fmLlamaReplyTo`, `fmLlamaChatLoop`, `fmLlamaPromptIds`, `fmLlamaWithBos`, `fmLlamaKvMiB`, `fmLlamaStats`, `fmCommonPrefix` |
| `models/mamba2` | `fmRunMamba2`, `fmMambaLoadBase`, `fmMambaLoadLora`, `fmMambaPrepareGpu`, `fmMambaNewChat`, `fmMambaFeed`, `fmMambaPrefillChunk`, `fmMambaPrefillPieces`, `fmMambaTensorCores`, `fmMambaWarmup`, `fmMambaReplyTo`, `fmMambaGenerate`, `fmMambaChatLoop`, `fmMambaTurnText`, `fmMambaSetNorm`, `fmMambaStats` |
| `sampling/sampler` | `fmNewSampler` (temp, top-p, min-p, repeat penalty, seed), `fmSample`, `fmSamplerSummary` |
| `runtime/console` | `fmConsoleInit` (quiet mode: library fd 1/2 → `/dev/null`), `fmSay`, `fmOut`, `fmErr`, `fmQuiet`, `fmReadTurn` (multi-line `\` input), `fmLibStdout`, `fmOutIsTty`, `fmErrIsTty` |
| `runtime/signals` | `fmOnCtrlC`, `fmStopRequested`, `fmClearStop`, `fmSetGenerating`, `fmShowProgress` / `fmEndProgress` (tok/s in the window title), `fmLiveProgress` |

Environment variables the Llama route reads through `nim_inference`:
`BYBY_FLASH_PREFILL` (default 1; 0 keeps the old prefill), `BYBY_PREFILL_CB_MS` (GPU time per
prefill command buffer, default 300), `BYBY_DECODE_CB_MS` (a decode token is split only when it
takes longer, default 1500), `BYBY_PREFILL_MAX_ROWS` (default 512), `BYBY_PREFILL_SESSION_MB`
(default 1024). Command buffers are only split where `sessionPacesCommandBuffers` is true, never
on Apple Silicon.

## Not done yet

- `--ngl` accepts only values ≥ the layer count (default: all). `nim_inference`'s decode
  refuses non-resident layers and Mamba-2 encodes a whole token in one command buffer, so a
  partial offload is reported as an error instead of being ignored.
- Llama single prompt: `--min-p`, `--repeat-penalty` and `--repeat-last-n` have no effect
  because `nim_inference.generate` takes only temp/top-p; they apply in `-cnv`. Mistral's
  sliding window is not emulated. `head_dim` ≠ hidden/heads, attention bias and
  rope_scaling other than linear/dynamic are rejected.
- Llama chat: no `/reset` or context shifting when full.
- Mamba-2: sizes come from the config, but the GPU kernels have only been run at the
  Codestral 7B size. Kernel settings come from `infer_lora.configureInferenceDefaults`.
  No norm-preset bisect yet.
- The Codestral path (base, LoRA, Linear++) builds but has not been run end to end, since it
  needs the 7B model.

## Graph models through FIR

A directory whose `model.fir` declares `runtime graph` runs directly from its
`model.safetensors` with `tokenizer.json` (native HF BPE); no `.nimq` conversion and
no model-specific code in formers.n. `models/gpt2-125m` is a complete example.

```sh
make formers-cli
other/formers.n/formers-cli --fir-check models/gpt2-125m/model.fir
other/formers.n/formers-cli --models models/gpt2-125m \
  --fir models/gpt2-125m/model.fir --raw -p "Hello, my name is" -n 32 --temp 0 --backend cpu
# GPU check outside the sandbox (or use --backend vulkan)
other/formers.n/formers-cli --models models/gpt2-125m \
  --fir models/gpt2-125m/model.fir --raw -p "Hello, my name is" -n 32 --temp 0 --backend metal
make formers-fir-test
python3 models/gpt2-125m/check_hf.py --backend cpu
```

`src/formers/ir/fir_graph.bybylang` schedules the whole FIR decode graph in one GPU
session per token; `fir_kernel.bybylang` turns the `kernel ... endkernel` blocks of
`model.fir` into `VmProgram`s (one `session_vm.emit*` call per line), so the model's own
operations (layernorm, `gelu_new`, bias add, cached attention) live in the model
file, not in Nim. `models/fir_model.bybylang` handles tokenization, the shared sampler, EOS
and generation (chat lives in `models/mamba2.bybylang`); `io/safetensors.bybylang` loads only graph weights.
Linear layers are Q4 resident by default (`--fir-f32` keeps them fp32 as the
correctness reference). Tests run the file's kernels via `runGrid` and full tiny graphs
through the production CPU backend (tolerance `1e-4`), and reject malformed kernel blocks.

Without `--fir`, the model directory's `model.fir` is used. This is one-shot
completion (`-cnv` chat uses the model's chat template); partial offload, unsupported config variants and context overflow
are rejected. On Intel Iris Plus 640, chat decode runs at 8.57 tok/s (Metal) and 11.40 tok/s (Vulkan) with Q4,
10.78 / 8.54 tok/s with `--fir-f32`; see
[`models/gpt2-125m/benchmark.md`](../../models/gpt2-125m/benchmark.md). The fp32 path matches offline Hugging Face on all 32 greedy
tokens.
