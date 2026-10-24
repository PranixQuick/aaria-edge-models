#!/usr/bin/env python3
"""
Fetch Telugu validation clips from AI4Bharat Kathbath dataset.

Follows pranix-aaria fetch logic:
- Dataset: ai4bharat/Kathbath, config: telugu, split: valid (NEVER train)
- Streaming download with decode=False to avoid unnecessary torch audio dependencies
- Exactly 60 clips
- Writes 16kHz mono WAV clips and manifest.jsonl
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path

DATASET_NAME = "ai4bharat/Kathbath"
CONFIG_NAME = "telugu"
SPLIT_NAME = "valid"
DEFAULT_COUNT = 60

AUDIO_CANDIDATES = ("audio_filepath", "audio")
TEXT_CANDIDATES = ("text", "transcript", "transcription", "sentence")


def token_from_env():
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
        val = (os.environ.get(name) or "").strip()
        if val:
            return val
    return None


def pick_column(available, candidates):
    available_set = set(available or ())
    for c in candidates:
        if c in available_set:
            return c
    return None


def to_16k_mono_file(src_bytes: bytes, dst_path: Path):
    """Write bytes to 16kHz mono WAV file."""
    import soundfile as sf
    import numpy as np

    data, sr = sf.read(io.BytesIO(src_bytes), dtype="float32", always_2d=True)
    mono = np.mean(data, axis=1)

    if sr != 16000:
        # Simple resample or warning; Kathbath is natively 16kHz
        try:
            import scipy.signal
            num_samples = int(len(mono) * 16000 / sr)
            mono = scipy.signal.resample(mono, num_samples)
            sr = 16000
        except ImportError:
            pass

    sf.write(str(dst_path), mono, sr, subtype="PCM_16")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=DEFAULT_COUNT, help="Number of clips to fetch")
    parser.add_argument("--out-dir", required=True, type=Path, help="Audio destination folder")
    parser.add_argument("--manifest", required=True, type=Path, help="Output JSONL manifest path")
    args = parser.parse_args()

    token = token_from_env()
    if not token:
        print("::error::HF_TOKEN secret missing", file=sys.stderr)
        sys.exit("HF_TOKEN secret missing")

    try:
        from datasets import Audio, load_dataset
    except ImportError:
        sys.exit("datasets and soundfile are required: pip install datasets soundfile")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    print(f"Streaming {DATASET_NAME} ({CONFIG_NAME}) [{SPLIT_NAME}] ...")
    try:
        ds = load_dataset(
            DATASET_NAME,
            CONFIG_NAME,
            split=SPLIT_NAME,
            streaming=True,
            token=token,
        )
    except Exception as e:
        msg = str(e)
        if any(term in msg.lower() for term in ("gated", "401", "403", "restricted", "access to model")):
            print(f"::error::Gated dataset access error: {msg[:300]}", file=sys.stderr)
            print(
                f"Dataset {DATASET_NAME} is GATED. Please accept its terms once at "
                f"https://huggingface.co/datasets/{DATASET_NAME} with the account behind HF_TOKEN, then re-run.",
                file=sys.stderr,
            )
        else:
            print(f"::error::Could not load dataset {DATASET_NAME}: {msg[:300]}", file=sys.stderr)
        sys.exit(1)

    columns = getattr(ds, "column_names", None) or ()
    audio_col = pick_column(columns, AUDIO_CANDIDATES)
    text_col = pick_column(columns, TEXT_CANDIDATES)
    if not audio_col or not text_col:
        sys.exit(f"Missing required columns in dataset. Available: {columns}")

    print(f"Columns used: audio={audio_col}, text={text_col}")
    ds = ds.cast_column(audio_col, Audio(decode=False))

    rows = []
    taken = 0
    skipped_count = 0
    total_audio_bytes = 0

    for example in ds:
        if taken >= args.count:
            break
        try:
            text = (example.get(text_col) or "").strip()
            if not text:
                skipped_count += 1
                continue
            raw_audio = (example.get(audio_col) or {}).get("bytes")
            if not raw_audio:
                skipped_count += 1
                continue

            name = f"kathbath_te_{taken:03d}.wav"
            dest = args.out_dir / name
            to_16k_mono_file(raw_audio, dest)
            total_audio_bytes += len(raw_audio)

            row = {
                "audio": name,
                "text": text,
                "lang": "te",
                "condition": "kathbath-telugu-valid",
                "source": f"{DATASET_NAME}:{SPLIT_NAME}",
            }
            rows.append(row)
            taken += 1
        except Exception as e:
            skipped_count += 1
            print(f"Skipping clip due to error: {e}", file=sys.stderr)
            continue

    with open(args.manifest, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Successfully fetched {len(rows)} clips to {args.out_dir}; manifest written to {args.manifest}")
    print(f"Skipped clips: {skipped_count}")
    print(f"[BANDWIDTH] Kathbath audio stream: {total_audio_bytes / (1024 * 1024):.2f} MB")

    if len(rows) != args.count:
        print(
            f"::error::Collected {len(rows)} clips, which does not meet requirement of exactly {args.count} clips (skipped {skipped_count}).",
            file=sys.stderr,
        )
        sys.exit(f"Fetched {len(rows)} clips, but must be exactly {args.count} clips.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
