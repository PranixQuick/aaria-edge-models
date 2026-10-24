#!/usr/bin/env python3
"""
Accuracy gate: Kathbath Telugu valid (60 clips) evaluation.

Compares:
(a) Original .nemo checkpoint (koyal-te-120m-1.0)
(b) Quantized ONNX model (model.int8.onnx) run via sherpa_onnx.OfflineRecognizer (v1.13.8)

Both evaluated using identical text normalisation and WER / CER scoring.
Measures inference latency for (b) on runner CPU (median and 95th-percentile seconds per clip).
Enforces gate: FAIL the job if (b)'s WER is > 1.0 point worse than (a)'s.
"""

import argparse
import json
import math
import os
import statistics
import sys
import time
from pathlib import Path

# Add script directory to sys.path for normalise import
sys.path.insert(0, str(Path(__file__).parent))
from normalise import cer, normalise, wer


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
        wer_a = wer(ref_text, hyp_a)
        cer_a = cer(ref_text, hyp_a)

        # Transcribe with (b) sherpa-onnx
        hyp_b, sec_b = transcribe_sherpa_onnx(recognizer, audio_file)
        wer_b = wer(ref_text, hyp_b)
        cer_b = cer(ref_text, hyp_b)
        latencies_b.append(sec_b)

        results.append({
            "audio": item["audio"],
            "reference": ref_text,
            "hyp_a_nemo": hyp_a,
            "wer_a": wer_a,
            "cer_a": cer_a,
            "hyp_b_onnx": hyp_b,
            "wer_b": wer_b,
            "cer_b": cer_b,
            "latency_b_sec": sec_b,
        })
        if idx % 10 == 0 or idx == len(entries):
            print(f"[{idx:02d}/{len(entries)}] a: wer={wer_a:.2f} cer={cer_a:.2f} | b: wer={wer_b:.2f} cer={cer_b:.2f} ({sec_b:.2f}s)")

    # Aggregate statistics
    mean_wer_a = statistics.mean(r["wer_a"] for r in results) * 100.0
    mean_cer_a = statistics.mean(r["cer_a"] for r in results) * 100.0
    mean_wer_b = statistics.mean(r["wer_b"] for r in results) * 100.0
    mean_cer_b = statistics.mean(r["cer_b"] for r in results) * 100.0

    median_sec_b = statistics.median(latencies_b)
    p95_sec_b = compute_p95(latencies_b)
    wer_diff = mean_wer_b - mean_wer_a

    # Print summary table
    print("\n" + "=" * 70)
    print("ACCURACY GATE RESULTS: Kathbath Telugu Valid (60 clips)")
    print("=" * 70)
    print(f"{'Model':<40} | {'WER (%)':<10} | {'CER (%)':<10}")
    print("-" * 70)
    print(f"{'(a) Original NeMo (koyal-te-120m-1.0)':<40} | {mean_wer_a:>9.2f}% | {mean_cer_a:>9.2f}%")
    print(f"{'(b) model.int8.onnx (sherpa-onnx 1.13.8)':<40} | {mean_wer_b:>9.2f}% | {mean_cer_b:>9.2f}%")
    print("-" * 70)
    print(f"WER degradation ((b) - (a)): {wer_diff:+.2f} percentage points (threshold: <= 1.00)")
    print(f"Runner CPU Latency for (b): Median = {median_sec_b:.3f} s/clip, 95th-percentile = {p95_sec_b:.3f} s/clip")
    print("=" * 70 + "\n")

    summary = {
        "dataset": "ai4bharat/Kathbath",
        "split": "valid",
        "clips": len(results),
        "nemo_original": {
            "wer_pct": round(mean_wer_a, 2),
            "cer_pct": round(mean_cer_a, 2),
        },
        "onnx_int8_sherpa": {
            "wer_pct": round(mean_wer_b, 2),
            "cer_pct": round(mean_cer_b, 2),
            "latency_median_sec": round(median_sec_b, 3),
            "latency_p95_sec": round(p95_sec_b, 3),
        },
        "wer_degradation_points": round(wer_diff, 2),
        "threshold_points": 1.0,
        "gate_passed": bool(wer_diff <= 1.0),
    }

    if args.out_results:
        args.out_results.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Hard gate check
    if wer_diff > 1.0:
        print(
            f"::error::ACCURACY GATE FAILED: model.int8.onnx WER is {wer_diff:.2f} points worse than original NeMo model "
            f"(exceeds 1.0 point limit).",
            file=sys.stderr,
        )
        sys.exit(1)

    print("::notice::ACCURACY GATE PASSED: WER degradation is within 1.0 point limit.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
