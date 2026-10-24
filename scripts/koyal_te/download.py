#!/usr/bin/env python3
"""
Download Koyal te-120M model assets from Hugging Face using snapshot_download.

Hard rules:
- Hugging Face token comes only from HF_TOKEN.
- If HF_TOKEN is missing: stops with 'HF_TOKEN secret missing'. Never prints or logs the token.
- Gated access: prints clear explanation if access terms have not yet been accepted on Hugging Face.
- Do not guess the file name: snapshot_download with allow_patterns=["*.nemo", "README.md", "*.md", "LICENSE*"],
  finds the single .nemo file (halts if zero or >1), and exports NEMO_MODEL_PATH to $GITHUB_ENV.
"""

import argparse
import os
import sys
from pathlib import Path

MODEL_REPO = "adalat-ai/koyal-te-120m-1.0"
ALLOW_PATTERNS = ["*.nemo", "README.md", "*.md", "LICENSE*"]


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
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is required: pip install huggingface-hub")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading snapshot for {args.repo} into {args.out_dir}...")
    print(f"Filter patterns: {ALLOW_PATTERNS}")

    try:
        snapshot_download(
            repo_id=args.repo,
            allow_patterns=ALLOW_PATTERNS,
            token=token,
            local_dir=str(args.out_dir),
        )
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
            print(f"::error::Failed to download model snapshot: {msg[:300]}", file=sys.stderr)
        sys.exit(1)

    # Find the single .nemo file in the downloaded snapshot
    nemo_files = sorted(list(args.out_dir.rglob("*.nemo")))
    if len(nemo_files) == 0:
        print(f"::error::No .nemo checkpoint file found in snapshot of {args.repo}", file=sys.stderr)
        sys.exit(f"Error: No .nemo file found in repository snapshot at {args.out_dir}")
    if len(nemo_files) > 1:
        nemo_names = [f.name for f in nemo_files]
        print(f"::error::Expected exactly one .nemo file, but found {len(nemo_files)}: {nemo_names}", file=sys.stderr)
        sys.exit(f"Error: Found multiple ({len(nemo_files)}) .nemo files in snapshot: {nemo_names}")

    found_nemo = nemo_files[0]
    file_size = found_nemo.stat().st_size
    print(f"Discovered single .nemo file: {found_nemo.name} ({format_size(file_size)})")

    # Pass found path to subsequent workflow steps via GITHUB_ENV
    github_env = os.environ.get("GITHUB_ENV")
    if github_env:
        with open(github_env, "a", encoding="utf-8") as f:
            f.write(f"NEMO_MODEL_PATH={found_nemo.resolve()}\n")
        print(f"Exported NEMO_MODEL_PATH={found_nemo.resolve()} to GITHUB_ENV")
    else:
        print(f"GITHUB_ENV not set; NEMO_MODEL_PATH is {found_nemo.resolve()}")

    # Compute total downloaded size
    total_bytes = sum(f.stat().st_size for f in args.out_dir.rglob("*") if f.is_file())
    print(f"[BANDWIDTH] Model download completed: {format_size(total_bytes)} ({total_bytes} bytes)")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
