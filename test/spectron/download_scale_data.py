"""Install the packed Spectron corpus from one Hugging Face commit."""

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import subprocess
from urllib.parse import quote
from urllib.request import Request, urlopen

from prepare_packed_data import HERE, build_lock

HUB = "https://huggingface.co"


def read_manifest(path):
    data = json.loads(path.read_bytes())
    bits = data["bits_per_token"]
    if data["format"] != "spectron-u8-v2" or bits not in range(8, 17):
        raise ValueError("Unsupported Spectron dataset format")
    streams = {}
    totals = {"english": 0, "code": 0}
    validation = set()
    for stream in data["streams"]:
        name, tokens = stream["file"], stream["tokens"]
        if (not name.endswith(".bin") or Path(name).name != name or "\\" in name or
                name in streams or not isinstance(tokens, int) or tokens <= 0):
            raise ValueError("Invalid packed stream entry")
        domain, split = stream["domain"], stream["split"]
        if domain not in totals or split not in ("train", "validation"):
            raise ValueError("Invalid packed stream split or domain")
        if split == "train":
            totals[domain] += tokens
        else:
            validation.add(domain)
        streams[name] = ((tokens * bits + 7) // 8, stream.get("sha256"))
    if any(total <= 0 or total != data[f"{domain}_tokens"] for domain, total in totals.items()):
        raise ValueError("Packed training quotas disagree with the manifest")
    if validation != set(totals):
        raise ValueError("Packed corpus requires validation data for each domain")
    return data, streams


def tokenizer_matches(path, manifest):
    digest = hashlib.md5(path.read_bytes(), usedforsecurity=False).hexdigest()
    return digest == manifest["tokenizer_md5"]


def installed(path):
    try:
        manifest, streams = read_manifest(path / "manifest.json")
        return (tokenizer_matches(path / "tokenizer.json", manifest) and
                all((path / name).stat().st_size == size for name, (size, _) in streams.items()))
    except (OSError, TypeError, ValueError, KeyError):
        return False


def verify_shard(path, size, digest):
    if path.stat().st_size != size:
        raise IOError(f"Unexpected byte count for {path.name}")
    if digest:
        with path.open("rb") as file:
            if hashlib.file_digest(file, "sha256").hexdigest() != digest:
                raise IOError(f"SHA-256 mismatch for {path.name}")


def auth_headers():
    token = os.environ.get("HF_TOKEN", "").strip()
    return {"Authorization": f"Bearer {token}"} if token else {}


def resolve_commit(repo, repo_type, revision):
    kind = {"model": "models", "dataset": "datasets"}[repo_type]
    url = f"{HUB}/api/{kind}/{repo}/revision/{quote(revision, safe='')}"
    with urlopen(Request(url, headers=auth_headers()), timeout=60) as response:
        sha = json.loads(response.read()).get("sha", "")
    if len(sha) != 40:
        raise RuntimeError("Hugging Face did not return a commit for the dataset")
    return sha


def curl(url, target):
    """Download with resume and retries; the partial file survives interruptions."""
    target.parent.mkdir(parents=True, exist_ok=True)
    command = ["curl", "-fL", "--retry", "5", "--retry-delay", "3", "--connect-timeout", "30",
               "-C", "-", "-o", str(target)]
    for key, value in auth_headers().items():
        command += ["-H", f"{key}: {value}"]
    try:
        subprocess.run([*command, url], check=True)
    except subprocess.CalledProcessError:
        # A finished or stale partial file makes the resume request fail; retry from scratch.
        target.unlink(missing_ok=True)
        subprocess.run([*command, url], check=True)


def download(args):
    if installed(args.output):
        print(f"[spectron-data] Ready at {args.output}", flush=True)
        return
    subdir = PurePosixPath(args.subdir)
    if subdir.is_absolute() or ".." in subdir.parts:
        raise ValueError("Dataset subdirectory must be relative to the Hub repo")
    sha = resolve_commit(args.repo, args.repo_type, args.revision)
    prefix = "" if args.repo_type == "model" else "datasets/"
    stage = args.output / ".hf-download"
    stage.mkdir(parents=True, exist_ok=True)

    def fetch(name):
        target = stage / name
        curl(f"{HUB}/{prefix}{args.repo}/resolve/{sha}/{quote(str(subdir / name))}", target)
        return target

    # Small index files are fetched again so a new commit never mixes with old metadata.
    for small in ("manifest.json", "tokenizer.json"):
        (stage / small).unlink(missing_ok=True)
    index = fetch("manifest.json")
    manifest, streams = read_manifest(index)
    tokenizer = fetch("tokenizer.json")
    if not tokenizer_matches(tokenizer, manifest):
        raise IOError("Dataset tokenizer does not match its manifest")
    for name, (size, digest) in streams.items():
        target = args.output / name
        if target.is_file():
            try:
                verify_shard(target, size, digest)
                print(f"[spectron-data] Verified {name}", flush=True)
                continue
            except OSError:
                pass
        print(f"[spectron-data] Downloading {name} ({size:,} bytes)", flush=True)
        shard = fetch(name)
        verify_shard(shard, size, digest)
        shard.replace(target)
    tokenizer.replace(args.output / "tokenizer.json")
    index.replace(args.output / "manifest.json")
    stage.rmdir()
    print(f"[spectron-data] Installed {args.repo}@{sha} at {args.output}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="bobbydeveloper2014/bybyai-spectron")
    parser.add_argument("--repo-type", choices=("model", "dataset"), default="model")
    parser.add_argument("--revision", default="main")
    parser.add_argument("--subdir", default="data/scale5b")
    parser.add_argument("--output", type=Path, default=HERE / "data" / "scale5b")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    with build_lock(args.output):
        download(args)


if __name__ == "__main__":
    main()
