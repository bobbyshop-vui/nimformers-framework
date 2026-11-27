"""Build mmap-ready Spectron corpora using exact token quotas, without padding."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, suppress
import hashlib
from itertools import zip_longest
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import tempfile


WIKI = "https://huggingface.co/datasets/wikimedia/wikipedia"
WIKI_REVISION = "ad5752b5e625abfcdeefe5ae0ad2c3721c4b2619"
STORIES = "https://huggingface.co/datasets/roneneldan/TinyStories"
STORIES_REVISION = "6d02f92396f7e3e60c2251070b6e0b168568074b"
WIKI_SHARDS = tuple(
    (f"wiki-{i:02d}.parquet", f"{WIKI}/resolve/{WIKI_REVISION}/20231101.en/train-{i:05d}-of-00041.parquet")
    for i in (0, 3, 16, 37)
)
STORY_SHARDS = (
    ("stories-00.parquet", f"{STORIES}/resolve/{STORIES_REVISION}/data/train-00000-of-00004-2d5a1467fff1081b.parquet"),
    ("stories-01.parquet", f"{STORIES}/resolve/{STORIES_REVISION}/data/train-00001-of-00004-5852b56a2bd28fd9.parquet"),
)
STORY_VAL = (
    "stories-val.parquet",
    f"{STORIES}/resolve/{STORIES_REVISION}/data/validation-00000-of-00001-869c898b519ad725.parquet",
)
LEVELS = ("basic", "intermediate", "advanced")
SEED = 20261127
HEADER = struct.Struct("<I")
HERE = Path(__file__).resolve().parent


def fetch(cache, shard):
    name, url = shard
    path = cache / name
    if not path.exists():
        with tempfile.NamedTemporaryFile(dir=cache, prefix=name, delete=False) as partial:
            temp = Path(partial.name)
        try:
            subprocess.run(
                ["curl", "-fsSL", "--retry", "3", "--connect-timeout", "25", url, "-o", str(temp)],
                check=True,
            )
            temp.replace(path)
        finally:
            temp.unlink(missing_ok=True)
    return path


def documents(path):
    from pyarrow.parquet import ParquetFile

    for batch in ParquetFile(path).iter_batches(batch_size=128, columns=["text"]):
        for text in batch.column(0).to_pylist():
            if text:
                text = text.removesuffix("<|endoftext|>").strip()
                if text:
                    yield (text + "\n\n").encode("utf-8")


def interleave(paths):
    for row in zip_longest(*(documents(path) for path in paths)):
        for doc in row:
            if doc is not None:
                yield doc


def english_documents(paths, source, validation, story_validation):
    seen = set()
    if source == "stories" and not validation:
        seen.update(hashlib.blake2b(doc, digest_size=16).digest() for doc in documents(story_validation))
    for doc in interleave(paths):
        key = hashlib.blake2b(doc, digest_size=16).digest()
        if source == "wiki" and ((key[0] == 0) != validation):
            continue
        if key not in seen:
            seen.add(key)
            yield doc


def code_documents(level, validation):
    from generate_code import examples

    # A deterministic content split excludes exact validation documents from
    # all training shards, even if the random generators hit the same example.
    seen = set()
    for doc in examples(level, SEED + 17 * LEVELS.index(level) + int(validation)):
        doc = doc.encode("utf-8")
        key = hashlib.blake2b(doc, digest_size=16).digest()
        if ((key[0] == 0) == validation) and key not in seen:
            seen.add(key)
            yield doc


def pack(packer, tokenizer, output, quota, docs):
    if output.exists():
        if output.stat().st_size == quota * 2:
            return
        raise ValueError(f"Wrong size for existing shard {output}")
    temp = output.with_suffix(".part")
    try:
        with temp.open("wb", buffering=1 << 20) as packed:
            with subprocess.Popen(
                [str(packer), str(tokenizer), str(quota)],
                stdin=subprocess.PIPE,
                stdout=packed,
                bufsize=1 << 15,
            ) as worker:
                try:
                    for doc in docs:
                        worker.stdin.write(HEADER.pack(len(doc)))
                        worker.stdin.write(doc)
                except BrokenPipeError:
                    pass
                finally:
                    with suppress(BrokenPipeError):
                        worker.stdin.close()
                if worker.wait() != 0:
                    raise RuntimeError(f"Tokenizer failed while building {output.name}")
        if temp.stat().st_size != quota * 2:
            raise RuntimeError(f"{output.name} did not reach its quota of {quota} tokens")
        temp.replace(output)
        print(f"[data] {output.name}: {quota:,} tokens", flush=True)
    finally:
        temp.unlink(missing_ok=True)


def join_shards(output, inputs):
    if output.exists():
        return
    temp = output.with_suffix(".part")
    try:
        with temp.open("wb") as dest:
            for part in inputs:
                with part.open("rb") as src:
                    shutil.copyfileobj(src, dest, length=4 << 20)
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def build_lock(output):
    """Keep partial shard files private to a single builder per output path."""
    with (output / ".build.lock").open("ab") as lock:
        try:
            if sys.platform == "win32":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Another process is building {output}") from error
        yield


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=HERE / "data" / "packed")
    parser.add_argument("--sources", type=Path, default=HERE / "data" / "sources")
    parser.add_argument("--packer", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=HERE / "models" / "spectron_bpe_tokenizer.json")
    parser.add_argument("--code-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--english-tokens", type=int, default=1_000_000_000)
    parser.add_argument("--validation-tokens", type=int, default=500_000,
                        help="Holdout tokens per domain, additional to the training quotas")
    parser.add_argument("--jobs", type=int, default=3)
    parser.add_argument("--part", choices=("all", "code", "english"), default="all")
    args = parser.parse_args()
    if min(args.code_tokens, args.english_tokens, args.validation_tokens) < 1_000 or args.jobs <= 0:
        parser.error("token quotas must be at least 1,000 and jobs must be positive")
    args.output.mkdir(parents=True, exist_ok=True)
    with build_lock(args.output):
        build(args)


def build(args):
    output = args.output
    args.sources.mkdir(parents=True, exist_ok=True)
    tokenizer_bytes = args.tokenizer.read_bytes()
    vocab_size = json.loads(tokenizer_bytes)["vocab_size"]
    tokenizer = output / "tokenizer.json"
    quotas = dict(zip((f"code-{level}.bin" for level in LEVELS),
                      (args.code_tokens * 35 // 100, args.code_tokens * 35 // 100,
                       args.code_tokens - 2 * (args.code_tokens * 35 // 100))))
    quotas.update({"english.bin": args.english_tokens,
                   "val-code.bin": args.validation_tokens,
                   "val-english.bin": args.validation_tokens})
    manifest_path = output / "manifest.json"
    plan = {
        "format": "spectron-u16-v1",
        "vocab_size": vocab_size,
        "tokenizer_md5": hashlib.md5(tokenizer_bytes, usedforsecurity=False).hexdigest(),
        "tokens": quotas,
        "seed": SEED,
        "code": "Python exercises generated by test/spectron/generate_code.py",
        "english": {
            "wikimedia/wikipedia": {"revision": WIKI_REVISION, "train_tokens": args.english_tokens - args.english_tokens // 4,
                                    "files": [url for _, url in WIKI_SHARDS]},
            "roneneldan/TinyStories": {"revision": STORIES_REVISION, "train_tokens": args.english_tokens // 4,
                                     "files": [url for _, url in (*STORY_SHARDS, STORY_VAL)]},
        },
    }
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != plan or tokenizer.read_bytes() != tokenizer_bytes:
            raise ValueError(f"Output directory uses a different dataset plan: {output}")
        for name, count in quotas.items():
            if (output / name).stat().st_size != count * 2:
                raise ValueError(f"Wrong size for {name}")
        print(f"[data] Dataset already built: {output}")
        return
    plan_path = output / ".plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text()) != plan or tokenizer.read_bytes() != tokenizer_bytes:
            raise ValueError(f"Output directory uses a different dataset plan: {output}")
    else:
        tokenizer.write_bytes(tokenizer_bytes)
        plan_path.write_text(json.dumps(plan, sort_keys=True))
    packer = args.packer.resolve()
    work = output / ".parts"
    work.mkdir(exist_ok=True)

    def build_code(level, validation=False):
        name = f"code-{level}.bin"
        count = quotas[name]
        path = output / name
        if validation:
            count, extra = divmod(args.validation_tokens, len(LEVELS))
            count += LEVELS.index(level) < extra
            path = work / f"val-{name}"
        pack(packer, tokenizer, path, count, code_documents(level, validation))

    def build_english(source, validation, paths, story_val):
        total = args.validation_tokens if validation else args.english_tokens
        quota = total // 4 if source == "stories" else total - total // 4
        name = f"val-{source}.bin" if validation else f"{source}.bin"
        pack(packer, tokenizer, work / name, quota, english_documents(paths, source, validation, story_val))

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        jobs = []
        if args.part in ("all", "code") and not (output / "val-code.bin").exists():
            jobs.extend(pool.submit(build_code, level, True) for level in LEVELS)
        if args.part in ("all", "code"):
            jobs.extend(pool.submit(build_code, level) for level in LEVELS)
        if args.part in ("all", "english") and not (output / "english.bin").exists():
            paths = list(pool.map(lambda shard: fetch(args.sources, shard), (*WIKI_SHARDS, *STORY_SHARDS, STORY_VAL)))
            wiki = paths[:len(WIKI_SHARDS)]
            stories = paths[len(WIKI_SHARDS):-1]
            val = paths[-1]
            for validation in (True, False):
                jobs.append(pool.submit(build_english, "wiki", validation, wiki, val))
                jobs.append(pool.submit(build_english, "stories", validation, [val] if validation else stories, val))
        for job in jobs:
            job.result()

    if args.part in ("all", "code"):
        join_shards(output / "val-code.bin", [work / f"val-code-{level}.bin" for level in LEVELS])
    if args.part in ("all", "english"):
        join_shards(output / "english.bin", [work / "wiki.bin", work / "stories.bin"])
        join_shards(output / "val-english.bin", [work / "val-wiki.bin", work / "val-stories.bin"])
    for path in work.glob("*.bin"):
        if ((path.name.startswith("val-code-") and (output / "val-code.bin").exists()) or
                (path.name in {"wiki.bin", "stories.bin", "val-wiki.bin", "val-stories.bin"} and
                 (output / "english.bin").exists() and (output / "val-english.bin").exists())):
            path.unlink()
    if not all((output / name).exists() for name in quotas):
        return
    for name, count in quotas.items():
        if (output / name).stat().st_size != count * 2:
            raise RuntimeError(f"Wrong size for {name}")
    temp = output / ".manifest.tmp"
    temp.write_text(json.dumps(plan, indent=2) + "\n")
    temp.replace(manifest_path)
    print(f"[data] Completed dataset: {output}", flush=True)


if __name__ == "__main__":
    main()
