#!/usr/bin/env python3
"""Offline GPT-2 fp32 regression and resident Q4 quality/throughput report.

Default: require the original fp32 32/32 HF fixture. --quality: report Q4
matching prefixes and Q4/fp32 logit error/top-1 at identical HF prefixes.
No files or downloads are created.

Run from the repo root after make formers-cli. Requires installed torch,
transformers, safetensors and tokenizers, and the existing local GPT-2 assets.
"""
import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

PROMPT = "Hello, my name is"
QUALITY_PROMPTS = [PROMPT, "The capital of France is", "def fibonacci(n):"]
# Locally obtained from HF GPT2LMHeadModel in eval mode, greedy FP32 on CPU.
KNOWN_IDS = [1757, 13, 314, 1101, 257, 6260, 11, 290, 314, 1101, 257, 6260,
             13, 314, 1101, 257, 6260, 13, 314, 1101, 257, 6260, 13, 314,
             1101, 257, 6260, 13, 314, 1101, 257, 6260]


def prefix_length(actual, expected):
    return next((i for i, (a, b) in enumerate(zip(actual, expected)) if a != b),
                min(len(actual), len(expected)))


def quality_check(args, root, model_dir, tokenizer, model, torch):
    import numpy as np

    requests, hf_logits, expected = [], [], []
    for text in args.prompt or QUALITY_PROMPTS:
        ids = tokenizer(text, return_tensors="pt").input_ids
        with torch.no_grad():
            sequence = model.generate(ids, max_new_tokens=args.tokens, do_sample=False,
                                      pad_token_id=tokenizer.eos_token_id)[0]
            continuation = sequence[ids.shape[1]:].tolist()
            steps = min(args.logit_steps, len(continuation))
            # Teacher-forced logits at exactly the same prefixes in every run.
            logits = model(sequence[:ids.shape[1]+steps-1].unsqueeze(0)).logits[0]
        expected.append(continuation)
        hf_logits.append(logits[ids.shape[1]-1:ids.shape[1]+steps-1].numpy())
        requests.append(dict(prompt=ids[0].tolist(), teacher=continuation[:steps],
                             count=len(continuation)))
    del model
    gc.collect()
    probe = root / "other/formers.n/build/fir_probe"
    if not probe.is_file():
        raise SystemExit("Build the offline probe: make -C other/formers.n fir-probe")
    runs = {}
    for precision in ("fp32", "q4"):
        print(f"Running resident {precision} on {args.backend} ({len(requests)} prompts)...", flush=True)
        completed = subprocess.run([str(probe), str(model_dir), args.backend, precision],
                                   input="".join(json.dumps(r)+"\n" for r in requests),
                                   cwd=root, text=True, capture_output=True)
        if completed.returncode:
            raise SystemExit(f"FAIL {precision} probe: {completed.stderr} {completed.stdout[:2000]}")
        runs[precision] = [json.loads(line) for line in completed.stdout.splitlines()]
        assert len(runs[precision]) == len(requests)
    texts = args.prompt or QUALITY_PROMPTS
    for i, (text, reference) in enumerate(zip(texts, expected)):
        fp, q4 = runs["fp32"][i], runs["q4"][i]
        f = np.asarray(fp["logits"], dtype=np.float32)
        q = np.asarray(q4["logits"], dtype=np.float32)
        h = hf_logits[i]
        assert f.shape == q.shape == h.shape and np.isfinite(f).all() and np.isfinite(q).all()
        assert fp["generated"] == reference, f"fp32 vs HF greedy mismatch for {text!r}"
        print(f"{text!r}: Q4/HF matching prefix {prefix_length(q4['generated'], reference)}/{len(reference)}; "
              f"Q4/fp32 max logit error={np.max(np.abs(q-f)):.6f}, "
              f"top-1 agreement={np.mean(q.argmax(-1)==f.argmax(-1)):.1%} ({len(f)} HF prefixes); "
              f"fp32/HF max error={np.max(np.abs(f-h)):.6f}")
        for precision, run in (("fp32", fp), ("q4", q4)):
            steps = len(requests[i]["prompt"])+len(reference)-1
            assert run["sessions"] == run["logits_reads"] == steps
            prefill = run["prefill_tokens"]/run["prefill_seconds"]
            decode = run["decode_steps"]/run["decode_seconds"] if run["decode_steps"] else 0
            print(f"  {precision}: sequential prefill {run['prefill_tokens']} tokens / "
                  f"{run['prefill_seconds']:.3f}s = {prefill:.3f} tok/s; "
                  f"decode {run['decode_steps']} steps / {run['decode_seconds']:.3f}s = {decode:.3f} tok/s; "
                  f"{steps} sessions, {steps} logits reads")
    print("PASS fp32 greedy vs offline HF; Q4 quality reported without a greedy-match threshold. "
          "Timings exclude load/JIT preparation; first-use backend costs may remain.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["cpu", "metal", "vulkan"], default="cpu")
    parser.add_argument("--quality", action="store_true", help="compare resident Q4/fp32 logits and greedy prefixes on several prompts")
    parser.add_argument("--tokens", type=int, default=32, help="quality-mode greedy continuation length")
    parser.add_argument("--logit-steps", type=int, default=8, help="quality-mode HF prefixes for logit comparisons")
    parser.add_argument("--prompt", action="append", help="override quality prompts; may be repeated")
    args = parser.parse_args()
    if args.tokens < 1 or args.logit_steps < 1:
        parser.error("--tokens and --logit-steps must be positive")
    import torch
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast

    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[2]
    model_dir = root / "models/gpt2-125m"
    tokenizer = GPT2TokenizerFast.from_pretrained(model_dir, local_files_only=True)
    model = GPT2LMHeadModel.from_pretrained(model_dir, local_files_only=True).cpu().eval()
    if args.quality:
        quality_check(args, root, model_dir, tokenizer, model, torch)
        return
    prompt = tokenizer(PROMPT, return_tensors="pt")
    assert prompt.input_ids[0].tolist() == [15496, 11, 616, 1438, 318]
    with torch.no_grad():
        output = model.generate(**prompt, max_new_tokens=32, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
    expected_ids = output[0, prompt.input_ids.shape[1]:].tolist()
    assert expected_ids == KNOWN_IDS, f"HF baseline changed: {expected_ids}"
    expected = tokenizer.decode(expected_ids)
    del model, output
    gc.collect()
    command = [str(root / "other/formers.n/formers-cli"), "--models", str(model_dir),
               "--fir", str(model_dir / "model.fir"), "--fir-f32", "--raw", "-p", PROMPT,
               "-n", "32", "--temp", "0", "--backend", args.backend]
    started = time.monotonic()
    completed = subprocess.run(command, cwd=root, text=True, capture_output=True)
    print(completed.stderr, end="")
    if completed.returncode:
        raise SystemExit(f"FAIL CLI exited {completed.returncode}: {completed.stdout}")
    actual = completed.stdout.removesuffix("\n")
    assert actual == expected, f"greedy mismatch\nactual: {actual!r}\nHF: {expected!r}"
    assert tokenizer.encode(actual, add_special_tokens=False) == expected_ids
    print(f"PASS GPT-2 {args.backend}: 32/32 greedy token IDs match offline HF "
          f"and known fixture ({time.monotonic()-started:.2f} s)")
    print(repr(actual))


if __name__ == "__main__":
    main()
