#!/usr/bin/env python3
"""
ONNX-only model evaluation and comparison for Telugu speech recognisers.

Evaluates ONNX CTC models using sherpa-onnx (OfflineRecognizer.from_nemo_ctc + decode_stream).
Computes corpus WER/CER, per-clip mean WER/CER, empty transcript count, and CPU latency
(median and 95th percentile).

Can compare two models (e.g. IndicConformer vs Koyal te-120M) on the same 60 Kathbath Telugu clips.
No pass/fail threshold - measurement only.
Exits non-zero only on errors (missing files, sha mismatch, fewer than 60 clips, 0 reference words).
"""

import argparse
import hashlib
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

# Add script directory to sys.path for normalise import
sys.path.insert(0, str(Path(__file__).parent))
from normalise import compute_edits_and_lengths, normalise

REQUIRED_MANIFEST_ENTRIES = 60


def sha256_file(path: Path) -> str:
    """Compute hex sha256 of file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_empty_transcript(text: str) -> bool:
    """Check if transcript is empty, whitespace, or normalises to empty string."""
    if not text or not str(text).strip():
        return True
    return not normalise(text).strip()


def compute_p95(values):
    """Compute 95th-percentile value."""
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = max(0, int(math.ceil(0.95 * len(sorted_v))) - 1)
    return sorted_v[idx]


def verify_against_manifest(manifest_path: Path, model_path: Path, tokens_path: Path):
    """Verify SHA-256 of downloaded model and tokens against signed manifest.json."""
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {manifest_path}")

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    te_models = data.get("models", {}).get("te", {})
    files = te_models.get("files", [])
    if not files:
        raise ValueError(f"No files found for models.te in manifest: {manifest_path}")

    manifest_hashes = {f["name"]: f["sha256"] for f in files if "name" in f and "sha256" in f}

    actual_model_sha = sha256_file(model_path)
    expected_model_sha = manifest_hashes.get("model.int8.onnx")
    if expected_model_sha and actual_model_sha != expected_model_sha:
        raise ValueError(
            f"SHA-256 mismatch for {model_path.name}: "
            f"actual={actual_model_sha}, expected in manifest={expected_model_sha}"
        )

    actual_tokens_sha = sha256_file(tokens_path)
    expected_tokens_sha = manifest_hashes.get("tokens.txt")
    if expected_tokens_sha and actual_tokens_sha != expected_tokens_sha:
        raise ValueError(
            f"SHA-256 mismatch for {tokens_path.name}: "
            f"actual={actual_tokens_sha}, expected in manifest={expected_tokens_sha}"
        )

    print(f"Verified SHA-256 against manifest {manifest_path.name}:")
    print(f"  {model_path.name}: {actual_model_sha} (MATCH)")
    print(f"  {tokens_path.name}: {actual_tokens_sha} (MATCH)")


def init_recognizer(model_path: Path, tokens_path: Path):
    """Initialise sherpa-onnx OfflineRecognizer from nemo CTC."""
    import sherpa_onnx

    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model file not found: {model_path}")
    if not tokens_path.is_file():
        raise FileNotFoundError(f"Tokens file not found: {tokens_path}")

    recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=str(model_path),
        tokens=str(tokens_path),
        num_threads=2,
        decoding_method="greedy_search",
        provider="cpu",
    )
    return recognizer


def transcribe_single(recognizer, wav_path: Path):
    """Transcribe a single audio file and measure latency."""
    import soundfile as sf

    if not wav_path.is_file():
        raise FileNotFoundError(f"Audio file not found: {wav_path}")

    audio, sample_rate = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)

    t0 = time.perf_counter()
    recognizer.decode_stream(stream)
    latency_sec = time.perf_counter() - t0

    result_text = stream.result.text
    return result_text, latency_sec


def evaluate_model(recognizer, entries, audio_dir: Path, model_label: str):
    """Evaluate a single recognizer on the provided entries."""
    clip_results = []
    latencies = []

    print(f"\nEvaluating {model_label} on {len(entries)} clips...")
    for idx, item in enumerate(entries, 1):
        audio_file = audio_dir / item["audio"]
        ref_text = item["text"]

        hyp_text, latency_sec = transcribe_single(recognizer, audio_file)
        edits = compute_edits_and_lengths(ref_text, hyp_text)
        latencies.append(latency_sec)

        clip_results.append({
            "audio": item["audio"],
            "reference": ref_text,
            "hypothesis": hyp_text,
            "edits": edits,
            "latency_sec": latency_sec,
        })
        if idx % 15 == 0 or idx == len(entries):
            print(f"  [{idx:02d}/{len(entries)}] wer={edits['clip_wer']:.2f} cer={edits['clip_cer']:.2f} ({latency_sec:.3f}s)")

    total_ref_words = sum(r["edits"]["ref_words"] for r in clip_results)
    total_ref_chars = sum(r["edits"]["ref_chars"] for r in clip_results)
    total_word_edits = sum(r["edits"]["word_edits"] for r in clip_results)
    total_char_edits = sum(r["edits"]["char_edits"] for r in clip_results)

    corpus_wer = (total_word_edits / total_ref_words * 100.0) if total_ref_words > 0 else 0.0
    corpus_cer = (total_char_edits / total_ref_chars * 100.0) if total_ref_chars > 0 else 0.0
    mean_wer = statistics.mean(r["edits"]["clip_wer"] for r in clip_results) * 100.0 if clip_results else 0.0
    mean_cer = statistics.mean(r["edits"]["clip_cer"] for r in clip_results) * 100.0 if clip_results else 0.0

    empty_count = sum(1 for r in clip_results if is_empty_transcript(r["hypothesis"]))
    median_latency = statistics.median(latencies) if latencies else 0.0
    p95_latency = compute_p95(latencies)

    metrics = {
        "model_label": model_label,
        "clips": len(clip_results),
        "total_ref_words": total_ref_words,
        "total_ref_chars": total_ref_chars,
        "total_word_edits": total_word_edits,
        "total_char_edits": total_char_edits,
        "corpus_wer_pct": round(corpus_wer, 2),
        "corpus_cer_pct": round(corpus_cer, 2),
        "mean_wer_pct": round(mean_wer, 2),
        "mean_cer_pct": round(mean_cer, 2),
        "empty_transcripts": empty_count,
        "latency_median_sec": round(median_latency, 3),
        "latency_p95_sec": round(p95_latency, 3),
    }
    return metrics, clip_results


def print_comparison_table(metrics_a, metrics_b=None):
    """Print formatted summary comparison table."""
    print("\n" + "=" * 106)
    print("TELUGU ON-PHONE SPEECH RECOGNISER COMPARISON: Kathbath Telugu Valid (60 clips)")
    print("=" * 106)
    header = f"{'Model':<34} | {'Corpus WER':<11} | {'Corpus CER':<11} | {'Mean WER':<10} | {'Empty':<6} | {'Median Lat':<10} | {'p95 Lat':<10}"
    print(header)
    print("-" * 106)

    def format_row(m):
        return (
            f"{m['model_label']:<34} | "
            f"{m['corpus_wer_pct']:>10.2f}% | "
            f"{m['corpus_cer_pct']:>10.2f}% | "
            f"{m['mean_wer_pct']:>9.2f}% | "
            f"{m['empty_transcripts']:>6d} | "
            f"{m['latency_median_sec']:>9.3f}s | "
            f"{m['latency_p95_sec']:>9.3f}s"
        )

    print(format_row(metrics_a))
    if metrics_b:
        print(format_row(metrics_b))
        print("-" * 106)
        wer_diff = metrics_b["corpus_wer_pct"] - metrics_a["corpus_wer_pct"]
        cer_diff = metrics_b["corpus_cer_pct"] - metrics_a["corpus_cer_pct"]
        print(f"Difference ({metrics_b['model_label']} vs {metrics_a['model_label']}):")
        print(f"  Corpus WER: {wer_diff:+.2f} percentage points")
        print(f"  Corpus CER: {cer_diff:+.2f} percentage points")
    print("=" * 106 + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Kathbath JSONL manifest")
    parser.add_argument("--audio-dir", required=True, type=Path, help="Directory containing audio clips")
    parser.add_argument("--model-a", required=True, type=Path, help="Model A ONNX file")
    parser.add_argument("--tokens-a", required=True, type=Path, help="Model A tokens.txt")
    parser.add_argument("--name-a", default="IndicConformer (current phone)", help="Display label for Model A")
    parser.add_argument("--model-b", type=Path, default=None, help="Model B ONNX file (optional comparison)")
    parser.add_argument("--tokens-b", type=Path, default=None, help="Model B tokens.txt")
    parser.add_argument("--name-b", default="Koyal te-120M", help="Display label for Model B")
    parser.add_argument("--verify-manifest", type=Path, default=None, help="Path to signed manifest.json to verify Model A hashes")
    parser.add_argument("--out-results", type=Path, default=Path("compare_results.json"), help="Output JSON path")
    args = parser.parse_args()

    # 1. Validate manifest existence and entry count
    if not args.manifest.is_file():
        print(f"::error::Manifest file does not exist: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    entries = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(entries) < REQUIRED_MANIFEST_ENTRIES:
        print(
            f"::error::Fewer than {REQUIRED_MANIFEST_ENTRIES} clips in manifest (found {len(entries)})",
            file=sys.stderr,
        )
        sys.exit(1)
    if len(entries) != REQUIRED_MANIFEST_ENTRIES:
        print(f"Warning: Manifest contains {len(entries)} clips (expected {REQUIRED_MANIFEST_ENTRIES})")

    # 2. Validate all audio files exist
    if not args.audio_dir.is_dir():
        print(f"::error::Audio directory does not exist: {args.audio_dir}", file=sys.stderr)
        sys.exit(1)

    missing_audio = []
    for item in entries:
        audio_name = item.get("audio")
        if not audio_name or not (args.audio_dir / audio_name).is_file():
            missing_audio.append(audio_name)
    if missing_audio:
        print(f"::error::Missing {len(missing_audio)} audio files: {missing_audio[:5]}", file=sys.stderr)
        sys.exit(1)

    # 3. Check reference volume > 0 words and > 0 chars
    total_words = sum(len(item.get("text", "").split()) for item in entries)
    total_chars = sum(len(item.get("text", "")) for item in entries)
    if total_words == 0 or total_chars == 0:
        print(f"::error::0 reference words or characters in manifest (words={total_words}, chars={total_chars})", file=sys.stderr)
        sys.exit(1)

    # 4. If verify-manifest provided, verify sha256 for Model A
    if args.verify_manifest:
        try:
            verify_against_manifest(args.verify_manifest, args.model_a, args.tokens_a)
        except Exception as e:
            print(f"::error::Manifest verification failed: {e}", file=sys.stderr)
            sys.exit(1)

    # 5. Evaluate Model A
    try:
        rec_a = init_recognizer(args.model_a, args.tokens_a)
    except Exception as e:
        print(f"::error::Failed to initialize Model A recognizer: {e}", file=sys.stderr)
        sys.exit(1)

    metrics_a, clips_a = evaluate_model(rec_a, entries, args.audio_dir, args.name_a)

    # 6. Evaluate Model B if provided
    metrics_b = None
    clips_b = None
    if args.model_b and args.tokens_b:
        try:
            rec_b = init_recognizer(args.model_b, args.tokens_b)
        except Exception as e:
            print(f"::error::Failed to initialize Model B recognizer: {e}", file=sys.stderr)
            sys.exit(1)
        metrics_b, clips_b = evaluate_model(rec_b, entries, args.audio_dir, args.name_b)

    # 7. Print comparison table
    print_comparison_table(metrics_a, metrics_b)

    # 8. Write compare_results.json
    results_payload = {
        "dataset": "ai4bharat/Kathbath",
        "split": "valid",
        "clip_count": len(entries),
        "total_reference_words": total_words,
        "total_reference_chars": total_chars,
        "model_a": metrics_a,
    }
    if metrics_b:
        results_payload["model_b"] = metrics_b
        results_payload["difference"] = {
            "wer_diff_points": round(metrics_b["corpus_wer_pct"] - metrics_a["corpus_wer_pct"], 2),
            "cer_diff_points": round(metrics_b["corpus_cer_pct"] - metrics_a["corpus_cer_pct"], 2),
        }

    args.out_results.write_text(json.dumps(results_payload, indent=2), encoding="utf-8")
    print(f"Wrote comparison results to {args.out_results}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
