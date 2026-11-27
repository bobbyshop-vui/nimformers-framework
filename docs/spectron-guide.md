
# Train Spectron with `test_nimformer`

[English](spectron-guide.md) | [Tiếng Việt](spectron-guide.vi.md)

`test/spectron/test_nimformer.nim` is both an importable model/training
module and a training program when compiled as the main module. It uses
embeddings, Spectron blocks combining Linear++ attention with Mamba2SSD,
a final LayerNorm, then a vocabulary projection. It has its own model and tokenizer.

## Train the MoE on 5 billion tokens

`test/spectron/data/scale5b/` contains 4.5 billion English tokens and 500 million
tokens for code, distillation, and syntax exercises. The English sources are
[English Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) and
[TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories). Most of
the code allocation is raw Python from permissively licensed repositories in
[CodeParrot Clean](https://huggingface.co/datasets/codeparrot/codeparrot-clean).
One million tokens cover exercises such as filling in missing parentheses.

The builder serializes every conversation in
`test/spectron/claude_distill_dataset.jsonl` as compact JSONL. The next-token
objective covers every message, including assistant `tool_calls`, function
names and argument strings, call IDs, tool results, and error flags. The
manifest reports how many conversations, calls, and results were included.

The existing 756-ID BPE vocabulary uses 10 bits per ID in a little-endian
`uint8` stream. Training shards occupy 6.25 GB, with an extra 1.25 MB for
500,000 validation tokens per domain. The trainer maps the shards and samples
them by the token counts in the manifest.

The packed corpus is published in
[bobbydeveloper2014/bybyai-spectron](https://huggingface.co/bobbydeveloper2014/bybyai-spectron/tree/main/data/scale5b)
under `data/scale5b/`. Install it into `test/spectron/data/scale5b/` from the
repository root with Python 3.11 or later and `curl`:

```sh
make download
```

`make download` runs `download-spectron-scale-data` and the other dataset
downloads. To fetch only the packed corpus, run
`make download-spectron-scale-data`. The downloader pins the Hub commit,
resumes partial files, and checks each shard's byte count and SHA-256 hash
from the manifest. Shards that are already installed and match their hash
are not downloaded again.

In the published distill shard, email addresses, user and host names, local
path identifiers, and credential patterns are masked. Every conversation,
tool call, and tool result is retained.

To rebuild from the sources, install PyArrow as well. Use `--distill=PATH` to
select a different conversation file, `--sources=DIR` to choose where
downloads are cached, and `--redact-distill` to apply the same masking as
the published shard.

```sh
python3 -m pip install pyarrow
make prepare-spectron-scale-data
```

The default MoE shares four Spectron blocks with a hidden size of 512 and
eight heads. A router selects one expert per token. Each expert contains four
feed-forward layers and **29,985,344 parameters**, in addition to shared
weights. An auxiliary loss balances routing and `expert_tokens` reports the
number of tokens routed to each expert.

```sh
make run-metal ARGS='--dataset=test/spectron/data/scale5b --moe-experts=2 --batch=16 --seq=128 --steps=6000'
```

MoE runs save to `test/spectron/models/moe/finetune.nimq` unless
`--checkpoint=PATH` is given. This keeps the dense chat checkpoint
`test/spectron/models/finetune.nimq` unchanged. Choose the largest batch that
fits in memory; each step samples `batch × seq` prediction positions.

Run the same command to resume for another 6,000 steps. The log reports
`step`, `epoch`, and cumulative tokens. An epoch counts five billion
successful prediction positions. The checkpoint stores the router, each
expert, and Adam moments with per-expert update counts. Run with
`--steps=0 --moe-experts=3` on the same checkpoint to grow it to three
experts and save immediately.

`--steps=0 --merge-experts=PATH` appends experts from a packed MoE checkpoint
with the same BPE, hidden size, expert width, and expert depth. It imports
the expert weights and initializes the new router rows. Use a donor trained
from the same backbone and continue training to adapt routing.

### Tensor Cores and checkpoint uploads

On NVIDIA GPUs, build with Vulkan and pass `--tensor-cores=require`. Linear
layer GEMMs in the forward and backward passes then run through
`VK_KHR_cooperative_matrix` with FP16 inputs and FP32 accumulation. Weights
and Adam state remain FP32. Before training, the trainer checks a 64×64 GEMM
against the exact result. It stops with an error when the driver offers no
cooperative matrices, rather than falling back to FP32. Each `[tensor-core]`
log line counts the GEMMs that ran on Tensor Cores and the small GEMMs that
used FP32.

```sh
make shaders
make build-spectron-trainer SPECTRON_TRAIN_BACKEND=vulkan
./test/spectron/test_nimformer --dataset=test/spectron/data/scale5b --moe-experts=2 \
  --batch=64 --seq=128 --steps=6000 --tensor-cores=require
```

`--checkpoint-hook=PATH` runs an executable after every save with the
checkpoint path and step. `test/spectron/hf_checkpoint_sync.py` uploads to
the Hugging Face Hub in the background, so training continues during the
upload. When a newer save arrives during an upload, the hook keeps only the
newest snapshot for the next upload. It reads `HF_TOKEN` and uploads to
`SPECTRON_HF_REPO` (default `bobbydeveloper2014/bybyai-spectron`) at
`SPECTRON_HF_CHECKPOINT` (default `moe/finetune.nimq`).

```sh
export HF_TOKEN=...   # a write token
python3 test/spectron/hf_checkpoint_sync.py --restore test/spectron/models/moe/finetune.nimq
./test/spectron/test_nimformer --dataset=test/spectron/data/scale5b --moe-experts=2 \
  --batch=64 --steps=6000 --tensor-cores=require --checkpoint-every=20 \
  --checkpoint-hook=test/spectron/hf_checkpoint_sync.py
python3 test/spectron/hf_checkpoint_sync.py --wait test/spectron/models/moe/finetune.nimq
```

`--restore` downloads the Hub checkpoint only when the local file is missing.
`--wait` returns after the last queued upload has finished.

### Train on Colab T4

[`notebooks/spectron_moe_colab_t4.ipynb`](../notebooks/spectron_moe_colab_t4.ipynb)
installs Nim and the NVIDIA Vulkan driver files, downloads the corpus and the
latest `moe/finetune.nimq`, and builds the Vulkan trainer. It then runs three
training steps at each batch size from 16 upward. It measures tok/s after the
first step and keeps the fastest batch that fits on the T4 and in Colab's
memory. Training uses `--tensor-cores=require` and the HRM core
(`HRM_CYCLES = 2`, `HRM_L_STEPS = 2`, `HRM_SEGMENTS = 2` in the settings
cell), and saves and uploads every 20 steps. The batch probe runs with the
same HRM settings. After a disconnect, run the notebook again to continue from the
last uploaded checkpoint and Adam state. Add a Hugging Face write token named
`HF_TOKEN` to Colab Secrets before running it.

### HRM core

The trainer can run the Spectron blocks as a Hierarchical Reasoning Model
([HRM, arXiv:2506.21734](https://arxiv.org/abs/2506.21734)). The first
`--hrm-l-layers` blocks form the fast low-level module L, and the remaining
blocks form the slow high-level module H. In each high-level cycle, L updates
its state `--hrm-l-steps` times from `zL + zH + x`, where `x` is the
normalized token embedding. H then updates once from `zH + zL`. A LayerNorm
follows each module, and the H state feeds the MoE and the LM head.

```sh
./test/spectron/test_nimformer --dataset=test/spectron/data/scale5b --moe-experts=2 \
  --batch=64 --seq=128 --steps=6000 --tensor-cores=require \
  --hrm-cycles=2 --hrm-l-steps=2 --hrm-segments=1
```

- `--hrm-cycles` turns HRM on and sets the high-level cycles per segment
  (default 2 when another HRM flag is given).
- `--hrm-l-steps` sets the low-level steps per cycle (default 2).
- `--hrm-l-layers` sets how many blocks belong to L (default half of the
  blocks). The rest belong to H.
- `--hrm-segments` sets the deep-supervision segments per optimizer step
  (default 1). Each segment starts from the previous segment's state and
  computes its own loss. The trainer averages the segment gradients into one
  Adam update.

Training uses the one-step gradient from the paper: only the last L update
and the last H update of each segment record activations for backward.
Each segment runs `cycles × (l_steps × l_layers + h_layers)` block passes,
which the `[hrm]` startup line prints. With 4 blocks and the defaults, that
is 12 block passes instead of 4.

A new HRM run scales the output projections of every block's attention,
Mamba and feed-forward branches by 0.1. Random full-size blocks make the
recurrence chaotic. Without the scaling, a 2-cycle, 2-step, 2-segment core
turned GPU rounding differences into logit differences larger than the
logits, so the same checkpoint gave a different validation loss in each
run. With the scaling, repeated forward passes agree to about 2e-4, and a
resumed run reproduces the saved validation loss.
`--init-checkpoint` weights are not scaled.

The checkpoint stores the HRM layout (`__hrm_layout`) and the three HRM norms
with their Adam state. On resume, the trainer reads the layout from the
checkpoint, so the HRM flags can be left out. `--hrm-cycles`,
`--hrm-l-steps` and `--hrm-segments` may change on resume, because they only
change compute. `--hrm-l-layers` must match the checkpoint. A checkpoint
saved without HRM cannot gain it on resume; start a new `--checkpoint` with
`--init-checkpoint` instead.

## Train on the packed dataset

The packed dataset contains **2 billion training tokens** split evenly between
synthetic Python and English prose. A separate validation set contains 500,000
tokens per domain. The Python generator covers 32 exercise families, from
loops and list operations to search, graphs, and dynamic programming. It
varies the inputs and emits solutions with executable assertions. Validation
uses different examples from the same exercise families. The English stream
combines 750 million tokens from [English Wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia)
with 250 million from [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories).
The manifest records the source revisions and token counts.

Use the existing dataset in `test/spectron/data/packed/`, or build it again
with Python, PyArrow and `curl` installed:

```sh
python3 -m pip install pyarrow
make prepare-spectron-data
```

The builder uses the existing Spectron BPE tokenizer. Each binary stores
little-endian `uint16` token IDs. The six streams occupy 4,002,000,000 bytes
(about 3.73 GiB). The trainer maps the streams read-only and copies the
sampled windows into each batch. Downloaded Parquet sources are cached
separately under `test/spectron/data/sources/` when using the default build
paths.

Start on Metal from the existing Spectron weights:

```sh
make run-metal ARGS='--dataset=test/spectron/data/packed --init-checkpoint=test/spectron/models/finetune.nimq --steps=6000'
```

The new run saves to `test/spectron/models/finetune_packed.nimq`. To continue
from that file, omit `--init-checkpoint`. The packed checkpoint stores the
completed step and Adam moments. The step seeds the batch sampler and drives
the learning-rate schedule across resumes. Supply `--checkpoint=PATH` for a
separate run. The tokenizer must match the one used to encode the dataset.
Use `make run-vulkan` with the same arguments on a Vulkan system.

The model reuses forward activations during backpropagation. The packed
trainer starts with linear learning-rate warmup, uses cosine decay and clips
the global gradient norm. A fresh run uses a peak learning rate of `0.0012`
and Adam with a second-moment decay of `0.98`. It keeps the training weights
in FP32. Set `--requantize-every=N`
to apply APF every N optimizer steps. Older packed checkpoints resume with
the Adam decay they were trained with.

The trainer checks logits, loss, gradients, weights, and Adam moments.
It discards a proposed update if one of these checks fails. For a numerical
failure during an update, it retries up to twice with a different batch and
halves the learning rate on each retry. Checkpoints store the peak rate and
the resulting multiplier. Pass `--lr=VALUE` to override the saved peak.
The training log shows the peak pre-clipping
gradient norm and the fraction of batches clipped.

The trainer mixes code and English equally. Over the first
4,000 steps the code mix moves from 65% basic, 25% intermediate and 10%
advanced to 35%, 35% and 30%. The trainer logs token throughput and validation
loss separately for code and English. At the default `--batch=16 --seq=128`,
6,000 steps consume 12,288,000 target positions.

The log reports the number of completed epochs and `epoch_tokens` toward the
next one. One epoch counts as many successful training target positions as
there are tokens in the training corpus. Randomly sampled windows can overlap
within an epoch. With 2 billion training tokens and the default batch and
sequence length, the first epoch completes at `[epoch] epoch=1 step=976563`.
The checkpoint stores the cumulative token count so it stays accurate when
batch or sequence length changes on resume. For older checkpoints, the trainer
estimates previous tokens from the current batch and sequence length and
prints `epoch_estimated=true`.

Run `./test/spectron/test_nimformer --help` for the packed options.

To chat with the packed checkpoint:

```sh
make run-spectron ARGS='--checkpoint=test/spectron/models/finetune_packed.nimq --tokenizer=test/spectron/data/packed/tokenizer.json'
```


## Build, run, and chat with the demo corpus

From the repository root:

```sh
make download-datasets
make build
./test/spectron/test_nimformer
```

The first command downloads six code JSONLs. If data already exists, start
with `make build`. That target builds the shared runtime and worker, then
checks/compiles the Spectron trainer; running the binary trains. Use
`make build-shared` to build only the shared runtime and worker.
Combine the steps on macOS:

```sh
make run-metal
```

On a system with a Vulkan loader and driver, use `make run-vulkan` to check,
compile with `-d:backend=vulkan`, and run the same Spectron trainer. The target
has no macOS-only gate. The trainer follows the `CFG_*` settings below; this is
not Mamba3 inference.

Or use `make run` for automatic backend detection. Each launch trains an
additional `CFG_N_STEPS` steps, generates from `CFG_GEN_PROMPT`, then exits.

Chat with a trained checkpoint:

```sh
make run-spectron
```

This compiles `test/spectron/test_inferencespectron.nim` into
`./test_inferencespectron`, loads checkpoint/tokenizer, and accepts prompts
at `>>>`. Enter `exit` or `quit` to stop. Chat keeps session history and
produces up to 128 tokens per turn at temperature 0.7, as configured in the
inference source. Use `ARGS` to pass `--checkpoint=PATH` and `--tokenizer=PATH`.


## Trainer startup

1. Read the Dolly corpus if present.
2. Scan each code JSONL and keep up to 4,000 valid examples per file through
   reservoir sampling, rather than concatenate all shards in full.
3. Load the saved BPE or train from about 300,000 characters drawn from
   the sources, then save it.
4. Load an existing checkpoint or initialize a model.
5. Encode the corpus, sample random windows, and train next-token prediction with Adam.
6. Save periodically, save once at the end, and generate a sample.

During training, `RawTokenMemory` keeps the encoded corpus token IDs and records
each sampled input window with its step, batch row, corpus offset, and length.
The exact input IDs from each window feed both Linear++ attention and the
Mamba2 parallel scan. `rawTokenAt` and `rawTokensForWindow` can recover the
corresponding BPE IDs and raw byte pieces for inspection or replay; token text
is reconstructed on demand to avoid duplicating strings across the corpus.
This is an in-process training memory and is not added to the model checkpoint.


## Input data

Default `CFG_CORPUS_PATH`/`CFG_CODE_CORPUS_PATHS` paths:

- `test/spectron/databricks-dolly-15k.jsonl`.
- `test/spectron/nim_python_code_dataset_mega.jsonl`.
- `test/spectron/python_code_dataset_1.jsonl` through `python_code_dataset_5.jsonl`.

`make download-datasets` downloads the six code files; provide Dolly separately
at its configured path. The Dolly branch reads **raw file text**, including
JSONL syntax, rather than converting rows into instructions/responses like
the Codestral trainer. Point `CFG_CORPUS_PATH` at your own UTF-8 text file
if you want a plain text corpus.

Each code row is a JSON object. A valid example:

```json
{"topic":"add two numbers","language":"Python","code":"def add(a, b):\n    return a + b","note":"Return the sum."}
```

`context` can replace `topic`; `explanation` can replace `note`.
Topic/context, language, and code must be nonempty. The trainer constructs
`Write <language> code for: <topic>`, then the note and code. Invalid JSON
or rows missing required content are skipped.

Missing files produce `[corpus] Khong tim thay ...`; remaining sources are
still used. If every source is empty/missing, the trainer repeats
`Nimformer Spectron demo corpus.` as demo text. Check corpus logs to know
what data was used before evaluating model quality.


## Change configuration

Edit `CFG_*` near the top of `test/spectron/test_nimformer.nim`, then rebuild:

- `CFG_EMBED_DIM=64`, `CFG_N_HEADS=4`, `CFG_N_LAYERS=2`, `CFG_FF_MULT=4`:
  new-model dimensions; embedding dimension must be divisible by head count.
- `CFG_BATCH_SIZE=32`, `CFG_SEQ_LEN=128`: windows per batch and tokens per window.
- `CFG_N_STEPS=6000`: additional steps **per launch**.
- `CFG_LR=3e-4`: Adam learning rate.
- `CFG_REQUANTIZE_EVERY=50`: cast parameters to APF-selected CustomFloat
  every 50 optimizer steps; keep float32 between casts.
- `CFG_LOG_EVERY=100`: logging interval.
- `CFG_CHECKPOINT_EVERY=5`, `CFG_CHECKPOINT_PATH`: save interval and destination.
- `CFG_NUM_MERGES=500`, `CFG_BPE_TRAIN_CHARS=300000`: settings for a new BPE.
- `CFG_CODE_EXAMPLES_PER_FILE=4000`: retained examples per code file.
- `CFG_GEN_PROMPT`, `CFG_GEN_MAX_NEW_TOKENS=40`, `CFG_GEN_TEMPERATURE=0`:
  generation at the end of training; temperature 0 uses greedy decoding.

For a shorter demo trial, set steps to `100`, batch to `4`, sequence length
to `64`, then run `make run-metal`. For packed training, use the
`--steps`, `--batch`, and `--lr` options in `ARGS`.

A loaded checkpoint supplies its own architecture; changing model-size
constants does not resize an existing model.


## Checkpoints, tokenizers, and resume

Default files:

```text
test/spectron/models/finetune.nimq
test/spectron/models/spectron_bpe_tokenizer.json
```

The demo checkpoint stores raw fp32 parameters, architecture, and completed step
count. A checkpoint at 25,080 resumes at 25,081 and ends at 31,080 with the
default additional step count. Adam `m`, `v`, and optimizer step are not
saved: resume continues the weights with a fresh optimizer, rather than
exactly reproducing uninterrupted training. Batch RNG is also reset.

Old checkpoints without final LayerNorm retain their old inference behavior
when loaded. The training entry point enables LayerNorm on resume and logs
that change; subsequent saves include `finalNorm.gamma` and `finalNorm.beta`.

Keep a backup before continuing:

```sh
cp test/spectron/models/finetune.nimq test/spectron/models/finetune.backup.nimq
cp test/spectron/models/spectron_bpe_tokenizer.json test/spectron/models/tokenizer.backup.json
```

For a new model, set `CFG_CHECKPOINT_PATH` to a file that does not exist.
For a new tokenizer too, use a separate experiment directory: also update
`BPE_TOKENIZER_PATH` and the corresponding `createDir` in the module. Adjust
the chat program's `--checkpoint` and `--tokenizer` options to load that model.

Keep checkpoint and tokenizer paired. Changing merge count does not retrain
an existing tokenizer; `getOrTrainBPE` loads it. Another tokenizer can have
the same vocabulary size but different token IDs, invalidating the match
between checkpoint and data.


## Read loss

Illustrative log:

```text
step   100 | loss = 5.1200 | mean_loss(99) = 5.4800
```

`loss` is cross-entropy averaged over `batch * seq` tokens in the current
batch, measured before Adam updates. `mean_loss(n)` averages the `n` batches
since the previous log. The first log contains one batch, so the next window
may contain fewer than 100. Batches use different random windows, so individual
losses can rise and fall. Packed training additionally reports fixed-window
validation losses in lines beginning with `[val]`.

Uniform logits with vocabulary 756 give about `ln(756) = 6.628`; this is a
reference, not an upper bound. Confidently predicting the wrong token can
produce much larger loss. Final LayerNorm stabilizes hidden magnitudes
before the lm head, and APF must select an exponent range that contains
the tensor's actual values.


## Use from a Nim program

Save as `spectron_demo.nim` at the repository root:

```nim
import std/random
import backend, test_nimformer

proc main() =
  randomize(42)
  let ctx = newBackend("auto")
  defer: closeBackend(ctx)
  let tok = trainBPE("hello nim hello world ", 8)
  let tokens = encodeBPE(tok, "hello nim hello world ")
  let ids = @[tokens[0 ..< tokens.len - 1]]
  let targets = @[tokens[1 .. ^1]]
  var model = newNimformerModelSpectron(tok.vocabSize, 16, 2, 1, 2)
  var adam = newModelAdamState(model)
  for step in 1 .. 10:
    echo step, ": ", trainStep(model, adam, ids, targets, ctx)
  let prompt = encodeBPE(tok, "hello ")
  let generated = model.generate(prompt, 8, ctx, temperature = 0'f32)
  echo decodeBPEStr(tok, generated)

main()
```

Compile and run:

```sh
nim c -r -d:release --mm:arc --opt:speed \
  --path:. --path:test/codestral --path:test/deepseek-inference \
  --path:test/spectron --path:kernelfortest spectron_demo.nim
```

Importing does not start demo training because its entry point is guarded
by `when isMainModule`. Ten steps on a tiny corpus only demonstrate API
usage; generated text does not represent a model trained on enough data.
See the [Training API](training-api.md) for signatures.


## Check the loss fixes

The APF test does not need a GPU:

```sh
nim c -r -d:release --path:. -o:/tmp/test_customfloat_range test/test_customfloat_range.nim
```

The Metal Spectron test uses a small model and temporary checkpoints. It
checks loss, finite-difference gradients, old/new checkpoints, and 110 steps
crossing APF boundaries 50/100. For HRM it checks the one-step gradients
against finite differences, repeated forward passes of a 2-cycle, 2-step,
2-segment core, an HRM checkpoint round trip, and 110 training steps:

```sh
nim c -r -d:release --mm:arc --opt:speed \
  --path:. --path:test/codestral --path:test/deepseek-inference \
  --path:test/spectron --path:kernelfortest \
  -o:/tmp/test_training_stability test/spectron/test_training_stability.nim
```
