#!/usr/bin/env python3
"""Mirror a packed Spectron checkpoint to the Hugging Face Hub without pausing training.

Hook mode (called by test_nimformer --checkpoint-hook=THIS_FILE):
    hf_checkpoint_sync.py CHECKPOINT STEP
  Hard-links the saved checkpoint into a queue and returns at once. A background
  worker uploads the newest queued snapshot; older queued snapshots are dropped.

Other modes:
    hf_checkpoint_sync.py --restore CHECKPOINT   download the Hub copy if CHECKPOINT is missing
    hf_checkpoint_sync.py --wait CHECKPOINT      block until the queued upload has finished

Environment:
    HF_TOKEN                  write token (required for uploads)
    SPECTRON_HF_REPO          default bobbydeveloper2014/bybyai-spectron
    SPECTRON_HF_REPO_TYPE     model or dataset (default model)
    SPECTRON_HF_CHECKPOINT    path inside the repo (default moe/finetune.nimq)
    SPECTRON_HF_DRY_RUN=1     queue and log without contacting the Hub
"""

import fcntl
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from urllib.error import HTTPError
from urllib.parse import quote
from urllib.request import Request, urlopen

HUB = "https://huggingface.co"
REPO = os.environ.get("SPECTRON_HF_REPO", "bobbydeveloper2014/bybyai-spectron")
REPO_TYPE = os.environ.get("SPECTRON_HF_REPO_TYPE", "model")
REMOTE = os.environ.get("SPECTRON_HF_CHECKPOINT", "moe/finetune.nimq")
DRY_RUN = os.environ.get("SPECTRON_HF_DRY_RUN", "0") not in ("", "0")
UPLOAD_ATTEMPTS = 3


def log(message):
    print(f"[hf-sync] {message}", flush=True)


def state_dir(checkpoint):
    checkpoint = Path(checkpoint).resolve()
    path = checkpoint.parent / f".hf-sync-{checkpoint.name}"
    path.mkdir(parents=True, exist_ok=True)
    return path


class Lock:
    def __init__(self, path, blocking=True):
        self.path, self.blocking, self.fd = path, blocking, None

    def acquire(self):
        self.fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(self.fd, fcntl.LOCK_EX | (0 if self.blocking else fcntl.LOCK_NB))
            return True
        except BlockingIOError:
            os.close(self.fd)
            self.fd = None
            return False

    def release(self):
        if self.fd is not None:
            fcntl.flock(self.fd, fcntl.LOCK_UN)
            os.close(self.fd)
            self.fd = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *_):
        self.release()


def queued(state, prefix):
    return sorted(state.glob(f"{prefix}-*.nimq"), key=lambda p: int(p.stem.split("-")[1]))


def worker_idle(state):
    probe = Lock(state / "worker.lock", blocking=False)
    if probe.acquire():
        probe.release()
        return True
    return False


def spawn_worker(state):
    subprocess.Popen([sys.executable, os.path.abspath(__file__), "--worker", str(state)],
                     stdin=subprocess.DEVNULL, start_new_session=True)


def enqueue(checkpoint, step):
    checkpoint = Path(checkpoint).resolve()
    state = state_dir(checkpoint)
    with Lock(state / "queue.lock"):
        for old in queued(state, "pending"):
            old.unlink()
        temporary = state / f"pending-{step}.tmp"
        temporary.unlink(missing_ok=True)
        try:
            # The trainer replaces the checkpoint by rename, so this link keeps a stable snapshot.
            os.link(checkpoint, temporary)
        except OSError:
            shutil.copyfile(checkpoint, temporary)
        temporary.replace(state / f"pending-{step}.nimq")
        if worker_idle(state):
            spawn_worker(state)
    log(f"Queued step {step} for {REPO}/{REMOTE}")


def upload(path, step):
    if DRY_RUN:
        # A numeric value simulates a slow upload of that many seconds.
        time.sleep(float(os.environ["SPECTRON_HF_DRY_RUN"]))
        log(f"Dry run: would upload step {step} ({path.stat().st_size:,} bytes) to {REPO}/{REMOTE}")
        return
    from huggingface_hub import HfApi

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    api.upload_file(path_or_fileobj=str(path), path_in_repo=REMOTE, repo_id=REPO,
                    repo_type=REPO_TYPE, commit_message=f"Spectron MoE checkpoint at step {step}")


def run_worker(state):
    state = Path(state)
    worker = Lock(state / "worker.lock")
    worker.acquire()
    with Lock(state / "queue.lock"):
        # Requeue a snapshot left behind by an interrupted worker.
        for stale in queued(state, "uploading"):
            if queued(state, "pending"):
                stale.unlink()
            else:
                stale.replace(state / stale.name.replace("uploading-", "pending-"))
    while True:
        with Lock(state / "queue.lock"):
            pending = queued(state, "pending")
            if not pending:
                worker.release()
                return
            snapshot = pending[-1]
            step = int(snapshot.stem.split("-")[1])
            current = state / f"uploading-{step}.nimq"
            snapshot.replace(current)
        started = time.monotonic()
        for attempt in range(1, UPLOAD_ATTEMPTS + 1):
            try:
                upload(current, step)
                log(f"Uploaded step {step} in {time.monotonic() - started:.0f}s")
                break
            except Exception as error:  # Network and Hub errors must not stop the trainer.
                log(f"Upload of step {step} failed (attempt {attempt}/{UPLOAD_ATTEMPTS}): {error}")
                if attempt < UPLOAD_ATTEMPTS:
                    time.sleep(10 * attempt)
        current.unlink(missing_ok=True)


def wait(checkpoint):
    state = state_dir(checkpoint)
    announced = False
    while True:
        with Lock(state / "queue.lock"):
            pending = bool(queued(state, "pending"))
            idle = worker_idle(state)
            if not pending and idle:
                log("No uploads in progress")
                return
            if pending and idle:
                spawn_worker(state)
        if not announced:
            log("Waiting for the checkpoint upload to finish")
            announced = True
        time.sleep(2)


def restore(checkpoint):
    checkpoint = Path(checkpoint)
    if checkpoint.exists():
        log(f"Using local {checkpoint}")
        return
    prefix = "" if REPO_TYPE == "model" else "datasets/"
    url = f"{HUB}/{prefix}{REPO}/resolve/main/{quote(REMOTE)}"
    token = os.environ.get("HF_TOKEN", "").strip()
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    try:
        urlopen(Request(url, method="HEAD", headers=headers), timeout=60).close()
    except HTTPError as error:
        if error.code == 404:
            log(f"No checkpoint at {REPO}/{REMOTE}; training starts from scratch")
            return
        raise
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    partial = checkpoint.with_name(checkpoint.name + ".download")
    command = ["curl", "-fL", "--retry", "5", "--connect-timeout", "30", "-C", "-", "-o", str(partial)]
    for key, value in headers.items():
        command += ["-H", f"{key}: {value}"]
    subprocess.run([*command, url], check=True)
    partial.replace(checkpoint)
    log(f"Restored {checkpoint} ({checkpoint.stat().st_size:,} bytes) from {REPO}/{REMOTE}")


def main():
    args = sys.argv[1:]
    if len(args) == 2 and args[0] == "--worker":
        run_worker(args[1])
    elif len(args) == 2 and args[0] == "--wait":
        wait(args[1])
    elif len(args) == 2 and args[0] == "--restore":
        restore(args[1])
    elif len(args) == 2 and not args[0].startswith("--"):
        enqueue(args[0], int(args[1]))
    else:
        sys.exit(__doc__)


if __name__ == "__main__":
    main()
