# formers.n

A llama.cpp-style inference framework written in **BybyLang** (`.bybylang`, compiled ahead
of time to Nim) on top of nimformers. One command, `formers-cli`, detects the model in
`--models` and runs DeepSeek/Llama or Codestral (Mamba-2, with or without LoRA/Linear++).

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
│       └── router.bybylang            # detection -> runner (false when unsupported)
└── tests/
    ├── test_detect.bybylang     # fake configs/tensors + real headers, no weights loaded
    └── make_tiny_llama.bybylang # random 2-layer Llama for a GPU smoke test
```

## Detection and routing

| Tensors in the `.nimq` + `config.json` | arch | Route |
|---|---|---|
| `model.layers.N.self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`, `model_type` llama/mistral | `llama` | `nim_inference.nim` |
| `backbone.layers.N.mixer.{A_log,in_proj,conv1d,dt_bias,...}`, `model_type` mamba2 | `mamba2` | Mamba-2 base (the base-Mamba path of `infer_lora.nim`) |
| Mamba-2 + LoRA `.bin` with adapters only | `mamba2+lora` | LoRA checkpoint (like `infer_lora.nim`) |
| Mamba-2 + LoRA `.bin` with P/G/A blocks (`newBlock.blockN.attn.*`) | `mamba2+linear++` | same, keeping the checkpoint's `normPreset` |
| anything else (gemma, qwen2, gpt2, Mamba-1, ...) | `unknown` | not run, `fmRun` returns `false`, exit 3 |

Detection checks the layer count, hidden size, head and KV-head counts, head_dim, FFN size,
vocab, the Mamba-2 `in_proj`/`conv1d` sizes and the LoRA pattern/sizes against the real
shape of every tensor; any mismatch is rejected with its reason. For Llama it also checks
that `tokenizer.json` is a Hugging Face tokenizer (older `export_to_nimq.py` versions
overwrote it).

## Usage

```bash
make build-formers          # -> other/formers.n/formers-cli
make formers-test           # detection test (no GPU, no weights loaded)
make formers-smoke          # tiny random Llama end to end on the GPU
```

```bash
other/formers.n/formers-cli --models test/deepseek-inference/models -p "Write quicksort in Python" -n 128
other/formers.n/formers-cli --models test/codestral/models -p "Hello" -n 60 --temp 0.7
other/formers.n/formers-cli --models test/codestral/models --no-lora -p "Hello"
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
`--verbose` for the full log.

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
`--no-progress`, `--no-warmup`, `-v/--verbose`, `--detect-only`, `-h`. Exit codes: 0 ok, 1 runtime error,
2 bad options, 3 unsupported model.

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
