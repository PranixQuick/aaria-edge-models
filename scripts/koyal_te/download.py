#!/usr/bin/env python3
"""
Download Koyal te-120M model assets from Hugging Face.

Hard rules:
- Hugging Face token comes only from HF_TOKEN.
- If HF_TOKEN is missing: stops with 'HF_TOKEN secret missing'. Never prints or logs the token.
- Gated access: prints clear explanation if access terms have not yet been accepted on Hugging Face.
"""

import argparse
import os
import sys
from pathlib import Path

MODEL_REPO = "adalat-ai/koyal-te-120m-1.0"
FILES_TO_DOWNLOAD = [
    "koyal-te-120m-1.0.nemo",
    "README.md",
]


def token_from_env():
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        val = (os.environ.get(name) or "").strip()
        if val:
            return val
    return None


def format_size(num_bytes: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default=MODEL_REPO, help="Hugging Face model repository")
    parser.add_argument("--out-dir", required=True, type=Path, help="Destination folder")
    args = parser.parse_args()

    token = token_from_env()
    if not token:
        print("::error::HF_TOKEN secret missing", file=sys.stderr)
        sys.exit("HF_TOKEN secret missing")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        sys.exit("huggingface_hub is required: pip install huggingface-hub")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {args.repo} assets to {args.out_dir}...")

    total_bytes = 0
    for filename in FILES_TO_DOWNLOAD:
        print(f"Fetching {filename} from {args.repo}...")
        try:
            downloaded_path = hf_hub_download(
                repo_id=args.repo,
                filename=filename,
                token=token,
                local_dir=str(args.out_dir),
            )
            file_size = Path(downloaded_path).stat().st_size
            total_bytes += file_size
            print(f"  -> {filename}: {format_size(file_size)}")
        except Exception as e:
            msg = str(e)
            if any(term in msg.lower() for term in ("gated", "401", "403", "restricted", "access to model")):
                print(f"::error::Gated model access error: {msg[:300]}", file=sys.stderr)
                print(
                    f"Model {args.repo} is GATED. Please accept its terms once at "
                    f"https://huggingface.co/{args.repo} with the account behind HF_TOKEN, then re-run.",
                    file=sys.stderr,
                )
            else:
                print(f"::error::Failed to download {filename}: {msg[:300]}", file=sys.stderr)
            sys.exit(1)

    print(f"[BANDWIDTH] Model download completed: {format_size(total_bytes)} ({total_bytes} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
