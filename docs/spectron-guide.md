
# Train Spectron with `test_nimformer`

[English](spectron-guide.md) | [Tiếng Việt](spectron-guide.vi.md)

`test/spectron/test_nimformer.nim` is both an importable model/training
module and a training program when compiled as the main module. It uses
embeddings, Spectron blocks combining Linear++ attention with Mamba2SSD,
a final LayerNorm, then a vocabulary projection. It has its own model and tokenizer.


## Build, run, and chat

From the repository root:

```sh
make download-datasets
make build
./test/spectron/test_nimformer
```

The first command downloads six code JSONLs. If data already exists, start
with `make build`. Building only checks/compiles; running the binary trains.
Combine the steps on macOS:

```sh
make run-metal
```

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
inference source. It currently ignores `ARGS`; edit constants and rebuild
to change these settings.


## Trainer startup

1. Read the Dolly corpus if present.
2. Scan each code JSONL and keep up to 4,000 valid examples per file through
   reservoir sampling, rather than concatenate all shards in full.
3. Load the saved BPE or train from about 300,000 characters drawn from
   the sources, then save it.
4. Load an existing checkpoint or initialize a model.
5. Encode the corpus, sample random windows, and train next-token prediction with Adam.
6. Save periodically, save once at the end, and generate a sample.


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

For a shorter trial, set steps to `100`, batch to `4`, sequence length to
`64`, then run `make run-metal`. There are no `--steps`, `--batch`, or `--lr`
options; Codestral variables such as `BYBY_SEQ_LEN` do not apply here.

A loaded checkpoint supplies its own architecture; changing model-size
constants does not resize an existing model.


## Checkpoints, tokenizers, and resume

Default files:

```text
test/spectron/models/finetune.nimq
test/spectron/models/spectron_bpe_tokenizer.json
```

The checkpoint stores raw fp32 parameters, architecture, and completed step
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
the chat program's `CHECKPOINT_PATH`/`BPE_TOKENIZER_PATH` to chat with that model.

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
losses can rise and fall. This entry point computes training loss only,
without a validation split/loss.

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
crossing APF boundaries 50/100:

```sh
nim c -r -d:release --mm:arc --opt:speed \
  --path:. --path:test/codestral --path:test/deepseek-inference \
  --path:test/spectron --path:kernelfortest \
  -o:/tmp/test_training_stability test/spectron/test_training_stability.nim
```
