"""Fetch exact upstream source files when internet access is available."""
from pathlib import Path
from urllib.request import urlretrieve

ROOT = Path(__file__).resolve().parent / "exact_upstream"
ROOT.mkdir(exist_ok=True)

FILES = {
    "transformers_modeling_mamba2.py": "https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/mamba2/modeling_mamba2.py",
    "mamba2.py": "https://raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/modules/mamba2.py",
    "mamba2_simple.py": "https://raw.githubusercontent.com/state-spaces/mamba/main/mamba_ssm/modules/mamba2_simple.py",
}

for name, url in FILES.items():
    target = ROOT / name
    print(f"Downloading {url}")
    urlretrieve(url, target)
    print(f"Wrote {target}")
