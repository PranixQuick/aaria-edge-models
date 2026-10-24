#!/usr/bin/env python3
"""
Accuracy gate: Kathbath Telugu valid (60 clips) evaluation.

Compares:
(a) Original .nemo checkpoint (koyal-te-120m-1.0)
(b) Quantized ONNX model (model.int8.onnx) run via sherpa_onnx.OfflineRecognizer (v1.13.8)

Both evaluated using identical text normalisation and Levenshtein edit distance scoring.
Corpus-level metrics (total edits / total reference units over all 60 clips) serve as
the definitive evaluation figures, with per-clip mean values preserved as additional columns.

Measures inference latency for (b) on runner CPU (median and 95th-percentile seconds per clip).
Enforces gate: FAIL the job if (b)'s corpus-level WER is > 1.0 percentage point worse than (a)'s.

Captures installed environment versions matching nemo|torch|onnx|sherpa|lightning into
the execution log and accuracy_results.json.
"""

import argparse
import json
import math
import os
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

# Add script directory to sys.path for normalise import
sys.path.insert(0, str(Path(__file__).parent))
from normalise import cer, compute_edits_and_lengths, normalise, wer


def get_installed_versions():
    """Extract installed packages matching nemo|torch|onnx|sherpa|lightning."""
    try:
        freeze_out = subprocess.check_output(
            [sys.executable, "-m", "pip", "freeze"], text=True
        )
        matches = [
            line.strip()
            for line in freeze_out.splitlines()
            if re.search(r"nemo|torch|onnx|sherpa|lightning", line, re.IGNORECASE)
        ]
        return matches
    except Exception as e:
        return [f"error reading pip freeze: {e}"]


def to_16k_mono_file(src_path: Path, dst_path: Path):
    """Ensure audio is 16kHz mono WAV for NeMo transcription."""
    import soundfile as sf
    import numpy as np

    data, sr = sf.read(str(src_path), dtype="float32", always_2d=True)
    mono = np.mean(data, axis=1)
    if sr != 16000:
        try:
            import scipy.signal
            num_samples = int(len(mono) * 16000 / sr)
            mono = scipy.signal.resample(mono, num_samples)
            sr = 16000
        except ImportError:
            pass
    sf.write(str(dst_path), mono, sr, subtype="PCM_16")


def transcribe_nemo(model, wav_path: Path) -> str:
    """Transcribe single clip using original NeMo ASRModel."""
    out = model.transcribe([str(wav_path)], verbose=False)
    if isinstance(out, tuple):
        out = out[0]
    first = out[0]
    text = getattr(first, "text", first)
    return str(text or "")


def transcribe_sherpa_onnx(recognizer, wav_path: Path):
    """Transcribe single clip using sherpa-onnx OfflineRecognizer and measure latency."""
    import soundfile as sf

    audio, sample_rate = sf.read(str(wav_path), dtype="float32")
    if audio.ndim > 1:
        audio = audio[:, 0]

    stream = recognizer.create_stream()
    stream.accept_waveform(sample_rate, audio)

    t0 = time.perf_counter()
    recognizer.decode(stream)
    latency_sec = time.perf_counter() - t0

    result_text = stream.result.text
    return result_text, latency_sec


def compute_p95(values):
    if not values:
        return 0.0
    sorted_v = sorted(values)
    idx = max(0, int(math.ceil(0.95 * len(sorted_v))) - 1)
    return sorted_v[idx]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-path", required=True, type=Path, help="Original .nemo model checkpoint")
    parser.add_argument("--onnx-model", required=True, type=Path, help="model.int8.onnx")
    parser.add_argument("--tokens", required=True, type=Path, help="tokens.txt")
    parser.add_argument("--manifest", required=True, type=Path, help="Kathbath JSONL manifest")
    parser.add_argument("--audio-dir", required=True, type=Path, help="Directory containing audio clips")
    parser.add_argument("--out-results", required=False, type=Path, default=Path("accuracy_results.json"))
    args = parser.parse_args()

    entries = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(entries)} evaluation clips from {args.manifest}")

    # Log installed dependency versions
    installed_versions = get_installed_versions()
    print("\n=== INSTALLED DEPENDENCY VERSIONS (pip freeze | grep -iE 'nemo|torch|onnx|sherpa|lightning') ===")
    for v in installed_versions:
        print(f"  {v}")
    print("=========================================================================================\n")

    # 1. Load original NeMo model (a)
    print("Loading original NeMo model...")
    import nemo.collections.asr as nemo_asr
    nemo_model = nemo_asr.models.ASRModel.restore_from(restore_path=str(args.nemo_path), map_location="cpu")
    nemo_model.eval()

    # 2. Load sherpa-onnx OfflineRecognizer (b)
    print("Initializing sherpa-onnx OfflineRecognizer (sherpa-onnx 1.13.8)...")
    import sherpa_onnx
    print(f"sherpa-onnx version: {getattr(sherpa_onnx, '__version__', 'unknown')}")

    recognizer = sherpa_onnx.OfflineRecognizer.from_nemo_ctc(
        model=str(args.onnx_model),
        tokens=str(args.tokens),
        num_threads=2,
        decoding_method="greedy_search",
        provider="cpu",
    )

    results = []
    latencies_b = []

    print(f"Transcribing and scoring {len(entries)} clips...")
    for idx, item in enumerate(entries, 1):
        audio_file = args.audio_dir / item["audio"]
        ref_text = item["text"]

        # Transcribe with (a) NeMo
        hyp_a = transcribe_nemo(nemo_model, audio_file)
        edits_a = compute_edits_and_lengths(ref_text, hyp_a)

        # Transcribe with (b) sherpa-onnx
        hyp_b, sec_b = transcribe_sherpa_onnx(recognizer, audio_file)
        edits_b = compute_edits_and_lengths(ref_text, hyp_b)
        latencies_b.append(sec_b)

        results.append({
            "audio": item["audio"],
            "reference": ref_text,
            "hyp_a_nemo": hyp_a,
            "wer_a": edits_a["clip_wer"],
            "cer_a": edits_a["clip_cer"],
            "edits_a": edits_a,
            "hyp_b_onnx": hyp_b,
            "wer_b": edits_b["clip_wer"],
            "cer_b": edits_b["clip_cer"],
            "edits_b": edits_b,
            "latency_b_sec": sec_b,
        })
        if idx % 10 == 0 or idx == len(entries):
            print(
                f"[{idx:02d}/{len(entries)}] a: wer={edits_a['clip_wer']:.2f} cer={edits_a['clip_cer']:.2f} | "
                f"b: wer={edits_b['clip_wer']:.2f} cer={edits_b['clip_cer']:.2f} ({sec_b:.2f}s)"
            )

    # 3. Corpus-level calculations (total edits / total reference words or characters over all clips)
    total_ref_words = sum(r["edits_a"]["ref_words"] for r in results)
    total_ref_chars = sum(r["edits_a"]["ref_chars"] for r in results)

    total_word_edits_a = sum(r["edits_a"]["word_edits"] for r in results)
    total_char_edits_a = sum(r["edits_a"]["char_edits"] for r in results)

    total_word_edits_b = sum(r["edits_b"]["word_edits"] for r in results)
    total_char_edits_b = sum(r["edits_b"]["char_edits"] for r in results)

    corpus_wer_a = (total_word_edits_a / total_ref_words * 100.0) if total_ref_words > 0 else 0.0
    corpus_cer_a = (total_char_edits_a / total_ref_chars * 100.0) if total_ref_chars > 0 else 0.0

    corpus_wer_b = (total_word_edits_b / total_ref_words * 100.0) if total_ref_words > 0 else 0.0
    corpus_cer_b = (total_char_edits_b / total_ref_chars * 100.0) if total_ref_chars > 0 else 0.0

    # Per-clip mean error rates
    mean_wer_a = statistics.mean(r["wer_a"] for r in results) * 100.0
    mean_cer_a = statistics.mean(r["cer_a"] for r in results) * 100.0
    mean_wer_b = statistics.mean(r["wer_b"] for r in results) * 100.0
    mean_cer_b = statistics.mean(r["cer_b"] for r in results) * 100.0

    median_sec_b = statistics.median(latencies_b)
    p95_sec_b = compute_p95(latencies_b)

    # The 1.0-point gate uses the corpus-level WER
    wer_diff = corpus_wer_b - corpus_wer_a

    # Print summary table with both corpus and per-clip mean error rates
    print("\n" + "=" * 96)
    print("ACCURACY GATE RESULTS: Kathbath Telugu Valid (60 clips)")
    print("=" * 96)
    print(f"{'Model':<40} | {'Corpus WER':<11} | {'Mean WER':<10} | {'Corpus CER':<11} | {'Mean CER':<10}")
    print("-" * 96)
    print(
        f"{'(a) Original NeMo (koyal-te-120m-1.0)':<40} | "
        f"{corpus_wer_a:>10.2f}% | {mean_wer_a:>9.2f}% | "
        f"{corpus_cer_a:>10.2f}% | {mean_cer_a:>9.2f}%"
    )
    print(
        f"{'(b) model.int8.onnx (sherpa-onnx 1.13.8)':<40} | "
        f"{corpus_wer_b:>10.2f}% | {mean_wer_b:>9.2f}% | "
        f"{corpus_cer_b:>10.2f}% | {mean_cer_b:>9.2f}%"
    )
    print("-" * 96)
    print(f"Corpus-level WER degradation ((b) - (a)): {wer_diff:+.2f} percentage points (threshold: <= 1.00)")
    print(f"Per-clip mean WER degradation: {mean_wer_b - mean_wer_a:+.2f} percentage points")
    print(f"Runner CPU Latency for (b): Median = {median_sec_b:.3f} s/clip, 95th-percentile = {p95_sec_b:.3f} s/clip")
    print(f"Total reference volume: {total_ref_words} words, {total_ref_chars} characters across {len(results)} clips")
    print("=" * 96 + "\n")

    summary = {
        "dataset": "ai4bharat/Kathbath",
        "split": "valid",
        "clips": len(results),
        "gate_metric": "corpus_wer",
        "total_reference_words": total_ref_words,
        "total_reference_chars": total_ref_chars,
        "nemo_original": {
            "corpus_wer_pct": round(corpus_wer_a, 2),
            "mean_wer_pct": round(mean_wer_a, 2),
            "corpus_cer_pct": round(corpus_cer_a, 2),
            "mean_cer_pct": round(mean_cer_a, 2),
            "total_edits_word": total_word_edits_a,
            "total_edits_char": total_char_edits_a,
            "wer_pct": round(corpus_wer_a, 2),
            "cer_pct": round(corpus_cer_a, 2),
        },
        "onnx_int8_sherpa": {
            "corpus_wer_pct": round(corpus_wer_b, 2),
            "mean_wer_pct": round(mean_wer_b, 2),
            "corpus_cer_pct": round(corpus_cer_b, 2),
            "mean_cer_pct": round(mean_cer_b, 2),
            "total_edits_word": total_word_edits_b,
            "total_edits_char": total_char_edits_b,
            "wer_pct": round(corpus_wer_b, 2),
            "cer_pct": round(corpus_cer_b, 2),
            "latency_median_sec": round(median_sec_b, 3),
            "latency_p95_sec": round(p95_sec_b, 3),
        },
        "wer_degradation_points": round(wer_diff, 2),
        "threshold_points": 1.0,
        "gate_passed": bool(wer_diff <= 1.0),
        "installed_versions": installed_versions,
    }

    if args.out_results:
        args.out_results.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Hard gate check: 1.0 point on corpus-level WER
    if wer_diff > 1.0:
        print(
            f"::error::ACCURACY GATE FAILED: model.int8.onnx corpus WER is {wer_diff:.2f} points worse than original NeMo model "
            f"(exceeds 1.0 point limit).",
            file=sys.stderr,
        )
        sys.exit(1)

    print("::notice::ACCURACY GATE PASSED: Corpus WER degradation is within 1.0 point limit.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
