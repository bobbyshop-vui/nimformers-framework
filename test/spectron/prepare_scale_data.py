"""Pack English, raw source code and local distillation examples for Spectron."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import suppress
import gzip
import hashlib
import json
from pathlib import Path
from random import Random
import re
import subprocess

from prepare_packed_data import (
    HEADER, HERE, SEED, STORIES, STORIES_REVISION, STORY_SHARDS, STORY_VAL,
    WIKI, WIKI_REVISION, build_lock, documents, english_documents, fetch,
)


WIKI_GROUPS = tuple(
    tuple((f"wiki-{i:02d}.parquet",
           f"{WIKI}/resolve/{WIKI_REVISION}/20231101.en/train-{i:05d}-of-00041.parquet")
          for i in group)
    for group in (range(9), range(9, 17), range(17, 36), range(36, 41))
)
STORY_TRAIN = (*STORY_SHARDS, *(
    (f"stories-{i:02d}.parquet", f"{STORIES}/resolve/{STORIES_REVISION}/data/{file}")
    for i, file in (
        (2, "train-00002-of-00004-a26307300439e943.parquet"),
        (3, "train-00003-of-00004-d243063613e5a057.parquet"),
    )
))
CODEPARROT = "https://huggingface.co/datasets/codeparrot/codeparrot-clean"
CODE_REVISION = "35a59fb025bc0a102f7d96eac09d145b896d487b"
CODE_SHARDS = tuple(
    (f"file-{i:012d}.json.gz", f"{CODEPARROT}/resolve/{CODE_REVISION}/file-{i:012d}.json.gz")
    for i in range(1, 13)
)
PERMISSIVE = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "0bsd",
              "unlicense", "cc0-1.0", "psf-2.0", "python-2.0"}

DISTILL_REDACTIONS = (
    (r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63})+\b",
     "redacted@example.invalid"),
    (r"(?<=/Users/)[A-Za-z0-9_.-]+(?=/|\b)|(?<=/home/)[A-Za-z0-9_.-]+(?=/|\b)", "local-user"),
    (r"(?<=-Users-)[A-Za-z0-9_.]+\b|(?<=-home-)[A-Za-z0-9_.]+\b", "local-user"),
    (r"(?<=var-folders-)[A-Za-z0-9_]{2}-[A-Za-z0-9_-]+?(?=-[TCX]-|-[TCX]\b)", "local"),
    (r"(?<=pytest-of-)[A-Za-z0-9_.-]+", "local-user"),
    (r"(?<![A-Za-z0-9._%+-])[A-Za-z0-9._-]{1,64}@(?:MacBook|iMac|Mac-mini|Mac-Pro|Mac-Studio)[A-Za-z0-9._-]*",
     "local-user@local-host"),
    (r"\b(?:MacBook|iMac|Mac-mini|Mac-Pro|Mac-Studio)(?:-[A-Za-z0-9]+)*-cua-[A-Za-z0-9-]+(?:\.local)?\b",
     "local-host"),
    (r"(?i)\bho[aà]ng\b", "local-user"),
    (r"/Volumes/[^/\s\"'<>]{1,60}/Laptrinh/App/", "/workspace/"),
    (r"(?:/private)?/var/folders/[A-Za-z0-9_]{2}/[A-Za-z0-9_-]{4,}", "/tmp/local"),
    (r"(?<=://)[^\s:@/\"'<>]{1,100}:[^\s:@/\"'<>]{1,100}@", "user:redacted@"),
    (r"\b(?:hf_[A-Za-z0-9]{30,}|gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{60,}|"
     r"sk-(?:proj-|ant-[a-z0-9]+-|svcacct-)?[A-Za-z0-9_\-]{32,}|(?:AKIA|ASIA)[A-Z0-9]{16}|"
     r"AIza[A-Za-z0-9_\-]{35}|xox[baprs]-[A-Za-z0-9-]{20,})\b", "redacted-key"),
    (r"(?i)(\bbearer\s+)[A-Za-z0-9._~+/-]{20,}=*", r"\1redacted"),
)
DISTILL_REDACTIONS = tuple((re.compile(pattern), replacement) for pattern, replacement in DISTILL_REDACTIONS)

SYNTAX = (
    ("Complete the function call.", "{name} = sorted({item}___", ")"),
    ("Close the function signature.", "def count_{name}({item}: list[int]___ -> int:", ")"),
    ("Finish the argument list.", "def update_{name}___value, step=1):", "("),
    ("Complete the list comprehension.", "{name} = [x for x in {item} if x > {number}___", "]"),
    ("Close the slice.", "{name} = {item}[:{number}___", "]"),
    ("Finish the dictionary literal.", "{name} = {{'limit': {number}___", "}"),
    ("End the loop header.", "for row in {item}___\n    print(row)", ":"),
    ("Complete the type annotation.", "{name}: list___int] = []", "["),
    ("Close the tuple.", "{name} = (None, {number}___", ")"),
    ("Close the context-manager call.", "with open({item}, 'r'___ as stream:", ")"),
    ("End the class header.", "class {class_name}(object)___\n    pass", ":"),
    ("Complete the attribute access.", "{name} = {item}___strip()", "."),
)


def syntax_documents():
    rng = Random(SEED)
    names = ("value", "paths", "results", "records", "window", "bucket", "scores", "sample")
    while True:
        task, snippet, answer = rng.choice(SYNTAX)
        name, item = rng.sample(names, 2)
        example = snippet.format(name=name, item=item, number=rng.randrange(2, 4096), class_name=name.title())
        yield f"{task}\n{example}\nMissing syntax: {answer}\n\n".encode()


def distill_records(path):
    with path.open(encoding="utf-8-sig") as lines:
        for line_no, line in enumerate(lines, 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                messages = record["messages"]
                if not isinstance(messages, list) or any(
                    not isinstance(message, dict) or not isinstance(message.get("role"), str)
                    for message in messages
                ):
                    raise ValueError("Expected messages with roles")
            except (ValueError, KeyError, TypeError) as error:
                raise ValueError(f"Invalid distill record at line {line_no}") from error
            yield record


def redact_distill(value):
    if isinstance(value, str):
        for pattern, replacement in DISTILL_REDACTIONS:
            value = pattern.sub(replacement, value)
        return value
    if isinstance(value, list):
        return [redact_distill(item) for item in value]
    if isinstance(value, dict):
        return {key: redact_distill(item) for key, item in value.items()}
    return value


def distill_documents(path, redact=False):
    for record in distill_records(path):
        if redact:
            record = redact_distill(record)
        yield (json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")


def distill_counts(path):
    counts = dict(records=0, messages=0, tool_calls=0, tool_results=0)
    for record in distill_records(path):
        counts["records"] += 1
        for message in record["messages"]:
            counts["messages"] += 1
            counts["tool_calls"] += len(message.get("tool_calls") or [])
            counts["tool_results"] += message["role"] == "tool"
    return counts


def raw_code_documents(cache, validation):
    seen = set()
    for source in CODE_SHARDS:
        path = fetch(cache, source)
        with gzip.open(path, "rt", encoding="utf-8") as lines:
            for line in lines:
                example = json.loads(line)
                code = example.get("content")
                if str(example.get("license", "")).lower() not in PERMISSIVE or not isinstance(code, str):
                    continue
                if not code.strip() or len(code) > 300_000:
                    continue
                data = (code.rstrip("\r\n") + "\n\n").encode("utf-8")
                key = hashlib.blake2b(data, digest_size=16).digest()
                if (key[0] == 0) == validation and key not in seen:
                    seen.add(key)
                    yield data


def wiki_documents(cache, group, validation=False):
    seen = set()
    for source in group:
        for doc in documents(fetch(cache, source)):
            key = hashlib.blake2b(doc, digest_size=16).digest()
            if (key[0] == 0) == validation and key not in seen:
                seen.add(key)
                yield doc


def stories_documents(cache):
    paths = [fetch(cache, source) for source in STORY_TRAIN]
    yield from english_documents(paths, "stories", False, fetch(cache, STORY_VAL))


def pack_bytes(packer, tokenizer, output, bits, quota, docs):
    receipt = output.with_suffix(".tokens")
    if output.exists() and (quota is not None or receipt.exists()):
        count = quota if quota is not None else int(receipt.read_text())
        if output.stat().st_size == (count * bits + 7) // 8:
            return count
        raise ValueError(f"Invalid packed byte length for {output}")
    temp = output.with_suffix(".part")
    try:
        with temp.open("wb") as out:
            with subprocess.Popen(
                [str(packer), str(tokenizer), "all" if quota is None else str(quota), str(bits)],
                stdin=subprocess.PIPE, stdout=out, stderr=subprocess.PIPE, bufsize=1 << 16,
            ) as process:
                try:
                    for doc in docs:
                        process.stdin.write(HEADER.pack(len(doc)))
                        process.stdin.write(doc)
                except BrokenPipeError:
                    pass
                finally:
                    with suppress(BrokenPipeError):
                        process.stdin.close()
                report = process.stderr.read().decode("utf-8", errors="replace").strip()
                if process.wait() != 0:
                    raise RuntimeError(f"{output.name}: {report}")
        count = int(report)
        if (quota is not None and count != quota) or temp.stat().st_size != (count * bits + 7) // 8:
            raise RuntimeError(f"Invalid token count for {output}")
        temp.replace(output)
        if quota is None:
            receipt.write_text(str(count))
        print(f"[data] {output.name} tokens={count:,} bytes={output.stat().st_size:,}", flush=True)
        return count
    finally:
        temp.unlink(missing_ok=True)


def build(args):
    root = args.output
    args.sources.mkdir(parents=True, exist_ok=True)
    tokenizer_bytes = args.tokenizer.read_bytes()
    vocab = json.loads(tokenizer_bytes)["vocab_size"]
    bits = max(8, (vocab - 1).bit_length())
    if vocab not in range(256, 65537):
        raise ValueError("Expected a byte BPE vocabulary of 256 to 65,536 IDs")
    tokenizer = root / "tokenizer.json"
    wiki_total = args.english_tokens - args.english_tokens // 9
    wiki_quotas = [wiki_total // 4 + (i < wiki_total % 4) for i in range(4)]
    with args.distill.open("rb") as file:
        distill_sha256 = hashlib.file_digest(file, "sha256").hexdigest()
    plan = {
        "format": "spectron-u8-v2", "bits_per_token": bits, "vocab_size": vocab,
        "tokenizer_md5": hashlib.md5(tokenizer_bytes, usedforsecurity=False).hexdigest(),
        "english_tokens": args.english_tokens, "code_tokens": args.code_tokens,
        "syntax_tokens": args.syntax_tokens, "validation_tokens_per_domain": args.validation_tokens,
        "sources": {
            "wikipedia": {"url": WIKI, "revision": WIKI_REVISION,
                          "files": [[name for name, _ in group] for group in WIKI_GROUPS]},
            "tinystories": {"url": STORIES, "revision": STORIES_REVISION},
            "raw_code": {"url": CODEPARROT, "revision": CODE_REVISION, "licenses": sorted(PERMISSIVE)},
            "distill": {"file": args.distill.name, "sha256": distill_sha256,
                        "serialization": "full-conversation-jsonl-v1", **distill_counts(args.distill)},
            "syntax": {"generator": "prepare_scale_data.py", "seed": SEED},
        },
    }
    if args.redact_distill:
        plan["sources"]["distill"]["redactions"] = ["email", "user_and_host_names", "local_paths", "credentials"]
    plan_path = root / ".plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan or tokenizer.read_bytes() != tokenizer_bytes:
            raise ValueError(f"Dataset plan differs in {root}")
    else:
        tokenizer.write_bytes(tokenizer_bytes)
        plan_path.write_text(json.dumps(plan, sort_keys=True))

    def write(name, count, docs, domain, split="train"):
        packed = pack_bytes(args.packer.resolve(), tokenizer, root / name, bits, count, docs)
        return {"file": name, "tokens": packed, "domain": domain, "split": split}

    distill = write("distill-tool-calls.bin", None, distill_documents(args.distill, args.redact_distill), "code")
    if not 0 < distill["tokens"] < args.code_tokens - args.syntax_tokens:
        raise ValueError("The distill file produced no text or exceeds the code token budget")
    raw_quota = args.code_tokens - args.syntax_tokens - distill["tokens"]
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        jobs = [
            pool.submit(write, f"english-wiki-{i}.bin", quota, wiki_documents(args.sources, group), "english")
            for i, (quota, group) in enumerate(zip(wiki_quotas, WIKI_GROUPS))
        ]
        jobs.extend((
            pool.submit(write, "english-stories.bin", args.english_tokens - wiki_total,
                        stories_documents(args.sources), "english"),
            pool.submit(write, "raw-code.bin", raw_quota, raw_code_documents(args.sources, False), "code"),
            pool.submit(write, "syntax.bin", args.syntax_tokens, syntax_documents(), "code"),
            pool.submit(write, "val-english.bin", args.validation_tokens,
                        wiki_documents(args.sources, WIKI_GROUPS[0], True), "english", "validation"),
            pool.submit(write, "val-code.bin", args.validation_tokens,
                        raw_code_documents(args.sources, True), "code", "validation"),
        ))
        streams = [job.result() for job in jobs]
    temp = root / ".manifest.tmp"
    temp.write_text(json.dumps({**plan, "streams": [*streams, distill]}, indent=2) + "\n")
    temp.replace(root / "manifest.json")
    print(f"[data] Completed {root}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packer", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=HERE / "data" / "scale5b")
    parser.add_argument("--sources", type=Path, default=HERE / "data" / "scale-sources")
    parser.add_argument("--tokenizer", type=Path, default=HERE / "models" / "spectron_bpe_tokenizer.json")
    parser.add_argument("--distill", type=Path, default=HERE / "claude_distill_dataset.jsonl")
    parser.add_argument("--redact-distill", action="store_true", help="Mask email, local paths and credential patterns")
    parser.add_argument("--english-tokens", type=int, default=4_500_000_000)
    parser.add_argument("--code-tokens", type=int, default=500_000_000)
    parser.add_argument("--syntax-tokens", type=int, default=1_000_000)
    parser.add_argument("--validation-tokens", type=int, default=500_000)
    parser.add_argument("--jobs", type=int, default=4)
    args = parser.parse_args()
    if args.english_tokens < 36 or not 0 < args.syntax_tokens < args.code_tokens or min(args.validation_tokens, args.jobs) < 1:
        parser.error("Invalid corpus quotas or worker count")
    args.output.mkdir(parents=True, exist_ok=True)
    with build_lock(args.output):
        build(args)


if __name__ == "__main__":
    main()
