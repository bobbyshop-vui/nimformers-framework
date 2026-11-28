"""Pack complete English answers and source files for end-of-sequence training."""

import argparse
import ast
import gzip
import hashlib
from itertools import islice
import json
from pathlib import Path
import subprocess
import tempfile
import warnings

from prepare_packed_data import HEADER, HERE, build_lock, fetch
from prepare_scale_data import CODE_SHARDS, PERMISSIVE


def documents(path, field, limit=None):
    lines = (gzip.open(path, "rt", encoding="utf-8-sig") if path.suffix == ".gz"
             else path.open(encoding="utf-8-sig"))
    with lines:
        for line in islice(lines, limit):
            try:
                record = json.loads(line)
                text = record.get(field)
            except (ValueError, AttributeError):
                continue
            if field == "content" and str(record.get("license", "")).lower() not in PERMISSIVE:
                continue
            if isinstance(text, str):
                text = text.rstrip().lstrip("\r\n")
                if 60 <= len(text) <= 15000 and "\0" not in text:
                    yield text


def select_docs(english, code, codeparrot):
    groups = {"english": [[], []], "code": [[], []]}
    seen = set()

    def add(domain, text):
        doc = text.encode("utf-8")
        digest = hashlib.blake2b(doc, digest_size=16).digest()
        if digest not in seen:
            seen.add(digest)
            groups[domain][int(digest[0] < 13)].append(doc)
            return True
        return False

    for path in english:
        for text in documents(path, "response"):
            add("english", text)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", SyntaxWarning)
        for field, paths, cap in (("code", code, 4000), ("content", codeparrot, 12000)):
            for path in paths:
                count = 0
                for text in documents(path, field, 120_000):
                    try:
                        if not ast.parse(text).body:
                            continue
                    except (SyntaxError, ValueError):
                        continue
                    count += add("code", text + "\n")
                    if count >= cap:
                        break
    return groups


def build(args):
    tokenizer = args.tokenizer.read_bytes()
    vocab = json.loads(tokenizer)["vocab_size"]
    bits = max(8, (vocab - 1).bit_length())
    manifest = {"format": "spectron-eos-v1", "bits_per_token": bits,
                "tokenizer_md5": hashlib.md5(tokenizer, usedforsecurity=False).hexdigest(),
                "documents": {}}
    groups = select_docs(args.english, args.code, args.codeparrot)
    with tempfile.TemporaryDirectory(prefix=".build-", dir=args.output) as directory:
        stage = Path(directory)
        for domain, splits in groups.items():
            for split, docs in zip(("train", "validation"), splits):
                name = f"{domain}-{split}.bin"
                if len(docs) < 8:
                    raise ValueError(f"Too few complete documents for {domain} {split}")
                with (stage / name).open("wb") as output:
                    with subprocess.Popen([str(args.packer.resolve()), str(args.tokenizer), "all", str(bits),
                                           "--records"], stdin=subprocess.PIPE, stdout=output, stderr=subprocess.PIPE) as process:
                        with process.stdin as inputs:
                            for doc in docs:
                                inputs.write(HEADER.pack(len(doc)))
                                inputs.write(doc)
                        report = process.stderr.read().decode().strip()
                        if process.wait():
                            raise RuntimeError(report)
                manifest["documents"][name] = len(docs)
                print(f"[eos-data] {name} documents={len(docs)} text_tokens={int(report)}", flush=True)
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        for name in (*manifest["documents"], "manifest.json"):
            (stage / name).replace(args.output / name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packer", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, default=HERE / "data/scale5b/tokenizer.json")
    parser.add_argument("--output", type=Path, default=HERE / "data/scale5b/eos")
    parser.add_argument("--english", type=Path, nargs="+", default=[HERE / "databricks-dolly-15k.jsonl"])
    parser.add_argument("--code", type=Path, nargs="+", default=sorted(HERE.glob("python_code_dataset_*.jsonl")))
    parser.add_argument("--codeparrot", type=Path, nargs="+", default=[])
    parser.add_argument("--sources", type=Path, default=HERE / "data/eos-sources")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with build_lock(args.output):
        if not args.codeparrot:
            args.sources.mkdir(parents=True, exist_ok=True)
            args.codeparrot = [fetch(args.sources, ("codeparrot-clean-01.json.gz", CODE_SHARDS[0][1]))]
        build(args)


if __name__ == "__main__":
    main()
