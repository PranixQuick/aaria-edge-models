#!/usr/bin/env python3
"""
Accuracy gate: Kathbath Telugu valid (60 clips) evaluation.

Compares:
(a) Original .nemo checkpoint (koyal-te-120m-1.0)
(b) Quantized ONNX model (model.int8.onnx) run via sherpa_onnx.OfflineRecognizer (v1.13.8)

Both evaluated using identical text normalisation and Levenshtein edit distance scoring.
Corpus-level metrics (total edits / total reference units over all clips) serve as
the definitive evaluation figures, with per-clip mean values preserved as additional columns.

Hard Gate Rules:
1. Manifest must contain exactly 60 entries, and all audio files must exist.
2. Total reference volume must be > 0 words and > 0 characters.
3. No more than 5 of the 60 clips may produce an empty transcript from either model.
4. Quality floor: corpus WER must be <= 35.0% for both original and quantized models.
5. Degradation limit: quantized model's corpus WER must be <= 1.0 point worse than original.
6. Like-for-like decoding: hybrid models must strictly switch decoding to CTC (stop on failure).

Measures inference latency for (b) on runner CPU (median and 95th-percentile seconds per clip).
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

# Gate limits & constants
REQUIRED_MANIFEST_ENTRIES = 60
MAX_EMPTY_TRANSCRIPTS = 5
MAX_CORPUS_WER_DEGRADATION = 1.0  # Percentage points degradation allowed
QUALITY_FLOOR_WER_PCT = 35.0      # Maximum allowed corpus WER % for either model


def is_hybrid_model(model) -> bool:
    """
    Check if model is a hybrid (has both RNNT and CTC heads).
    Criteria:
    - class name contains "Hybrid"
    - OR has a ctc_decoder attribute alongside a decoder or joint attribute
    """
    class_name = type(model).__name__
    if "Hybrid" in class_name:
        return True
    has_ctc = hasattr(model, "ctc_decoder")
    has_rnnt = hasattr(model, "decoder") or hasattr(model, "joint")
    return bool(has_ctc and has_rnnt)


def is_empty_transcript(text: str) -> bool:
    """Check if transcript is empty, whitespace, or normalises to empty string."""
    if not text or not str(text).strip():
        return True
    return not normalise(text).strip()


def count_empty_transcripts(results):
    """Count empty transcripts for model a, model b, and clips empty in either model."""
    empty_a = sum(1 for r in results if is_empty_transcript(r.get("hyp_a_nemo", "")))
    empty_b = sum(1 for r in results if is_empty_transcript(r.get("hyp_b_onnx", "")))
    empty_either = sum(
        1
        for r in results
        if is_empty_transcript(r.get("hyp_a_nemo", ""))
        or is_empty_transcript(r.get("hyp_b_onnx", ""))
    )
    return {
        "nemo_empty": empty_a,
        "onnx_empty": empty_b,
        "either_empty": empty_either,
    }


def check_manifest_entries(entries, required_count: int = REQUIRED_MANIFEST_ENTRIES):
    """Validate manifest entry count."""
    if len(entries) != required_count:
        raise ValueError(f"Manifest must contain exactly {required_count} entries, found {len(entries)}")


def check_audio_files(entries, audio_dir: Path):
    """Validate that every audio file referenced in manifest exists on disk."""
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio directory does not exist: {audio_dir}")
    missing_files = []
    for idx, item in enumerate(entries):
        audio_name = item.get("audio")
        if not audio_name:
            missing_files.append(f"Entry {idx} missing 'audio' property")
            continue
        audio_path = audio_dir / audio_name
        if not audio_path.is_file():
            missing_files.append(str(audio_path))
    if missing_files:
        raise FileNotFoundError(f"Missing {len(missing_files)} audio file(s): {', '.join(missing_files[:5])}")


def check_gate_rules(
    results,
    quality_floor_wer_pct: float = QUALITY_FLOOR_WER_PCT,
    max_wer_degradation: float = MAX_CORPUS_WER_DEGRADATION,
    max_empty_transcripts: int = MAX_EMPTY_TRANSCRIPTS,
):
    """
    Evaluate all gate criteria on evaluation results:
    1. Reference volume > 0 (both words and characters).
    2. Empty transcripts in either model <= max_empty_transcripts.
    3. Quality floor: corpus WER <= quality_floor_wer_pct for both models.
    4. Degradation: (corpus_wer_b - corpus_wer_a) <= max_wer_degradation.
    Returns (passed: bool, message: str, stats: dict).
    """
    total_ref_words = sum(r["edits_a"]["ref_words"] for r in results)
    total_ref_chars = sum(r["edits_a"]["ref_chars"] for r in results)
    empty_counts = count_empty_transcripts(results)

    stats = {
        "total_ref_words": total_ref_words,
        "total_ref_chars": total_ref_chars,
        "empty_counts": empty_counts,
    }

    if total_ref_words == 0 or total_ref_chars == 0:
        return (
            False,
            f"Zero reference volume: total_ref_words={total_ref_words}, total_ref_chars={total_ref_chars}",
            stats,
        )

    if empty_counts["either_empty"] > max_empty_transcripts:
        return (
            False,
            f"Too many clips with empty transcripts: {empty_counts['either_empty']} clips "
            f"(model a: {empty_counts['nemo_empty']}, model b: {empty_counts['onnx_empty']}, max allowed: {max_empty_transcripts})",
            stats,
        )

    total_word_edits_a = sum(r["edits_a"]["word_edits"] for r in results)
    total_char_edits_a = sum(r["edits_a"]["char_edits"] for r in results)
    total_word_edits_b = sum(r["edits_b"]["word_edits"] for r in results)
    total_char_edits_b = sum(r["edits_b"]["char_edits"] for r in results)

    corpus_wer_a = (total_word_edits_a / total_ref_words * 100.0) if total_ref_words > 0 else 0.0
    corpus_cer_a = (total_char_edits_a / total_ref_chars * 100.0) if total_ref_chars > 0 else 0.0
    corpus_wer_b = (total_word_edits_b / total_ref_words * 100.0) if total_ref_words > 0 else 0.0
    corpus_cer_b = (total_char_edits_b / total_ref_chars * 100.0) if total_ref_chars > 0 else 0.0

    wer_diff = corpus_wer_b - corpus_wer_a

    stats.update({
        "total_word_edits_a": total_word_edits_a,
        "total_char_edits_a": total_char_edits_a,
        "total_word_edits_b": total_word_edits_b,
        "total_char_edits_b": total_char_edits_b,
        "corpus_wer_a": corpus_wer_a,
        "corpus_cer_a": corpus_cer_a,
        "corpus_wer_b": corpus_wer_b,
        "corpus_cer_b": corpus_cer_b,
        "wer_diff": wer_diff,
    })

    if corpus_wer_a > quality_floor_wer_pct:
        return (
            False,
            f"Original NeMo model corpus WER {corpus_wer_a:.2f}% exceeds quality floor limit of {quality_floor_wer_pct:.2f}%",
            stats,
        )

    if corpus_wer_b > quality_floor_wer_pct:
        return (
            False,
            f"Quantized INT8 model corpus WER {corpus_wer_b:.2f}% exceeds quality floor limit of {quality_floor_wer_pct:.2f}%",
            stats,
        )

    if wer_diff > max_wer_degradation:
        return (
            False,
            f"Corpus WER degradation {wer_diff:+.2f} points exceeds limit of {max_wer_degradation:.2f} points",
            stats,
        )

    return True, "All accuracy gate checks passed", stats


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
    if isinstance(out, tuple) and len(out) > 0:
        out = out[0]
    if isinstance(out, (list, tuple)):
        first = out[0] if len(out) > 0 else ""
    else:
        first = out
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
    recognizer.decode_stream(stream)
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

    if not args.manifest.is_file():
        print(f"::error::Manifest file does not exist: {args.manifest}", file=sys.stderr)
        sys.exit(1)

    entries = [json.loads(line) for line in args.manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    print(f"Loaded {len(entries)} evaluation clips from {args.manifest}")

    # Fix 1: Validate exactly 60 entries and verify all audio files exist
    try:
        check_manifest_entries(entries, REQUIRED_MANIFEST_ENTRIES)
    except ValueError as e:
        print(f"::error::{e}", file=sys.stderr)
        sys.exit(1)

    try:
        check_audio_files(entries, args.audio_dir)
    except (FileNotFoundError, ValueError) as e:
        print(f"::error::{e}", file=sys.stderr)
        sys.exit(1)

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

    nemo_model_class = type(nemo_model).__name__
    print(f"NeMo model loaded: class={nemo_model_class}")

    # Fix 4: Like-for-like decoding for hybrid models
    if is_hybrid_model(nemo_model):
        print(f"Hybrid model detected ({nemo_model_class}): switching decoding strategy to CTC...")
        try:
            if not hasattr(nemo_model, "change_decoding_strategy"):
                raise AttributeError(f"Hybrid model {nemo_model_class} lacks change_decoding_strategy")
            nemo_model.change_decoding_strategy(decoder_type="ctc")
            print("Successfully switched hybrid model decoding strategy to CTC.")
        except Exception as e:
            print(f"::error::Failed to switch hybrid model decoding strategy to CTC: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Pure CTC model ({nemo_model_class}): leaving decoding strategy as-is.")

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

    # 3. Corpus-level calculations & Gate evaluation
    passed, gate_msg, stats = check_gate_rules(results)

    total_ref_words = stats.get("total_ref_words", 0)
    total_ref_chars = stats.get("total_ref_chars", 0)
    total_word_edits_a = stats.get("total_word_edits_a", 0)
    total_char_edits_a = stats.get("total_char_edits_a", 0)
    total_word_edits_b = stats.get("total_word_edits_b", 0)
    total_char_edits_b = stats.get("total_char_edits_b", 0)
    corpus_wer_a = stats.get("corpus_wer_a", 0.0)
    corpus_cer_a = stats.get("corpus_cer_a", 0.0)
    corpus_wer_b = stats.get("corpus_wer_b", 0.0)
    corpus_cer_b = stats.get("corpus_cer_b", 0.0)
    wer_diff = stats.get("wer_diff", 0.0)
    empty_counts = stats.get("empty_counts", count_empty_transcripts(results))

    # Per-clip mean error rates
    mean_wer_a = statistics.mean(r["wer_a"] for r in results) * 100.0 if results else 0.0
    mean_cer_a = statistics.mean(r["cer_a"] for r in results) * 100.0 if results else 0.0
    mean_wer_b = statistics.mean(r["wer_b"] for r in results) * 100.0 if results else 0.0
    mean_cer_b = statistics.mean(r["cer_b"] for r in results) * 100.0 if results else 0.0

    median_sec_b = statistics.median(latencies_b) if latencies_b else 0.0
    p95_sec_b = compute_p95(latencies_b)

    # Fix 5: Print real clip count in header, not hard-coded 60
    print("\n" + "=" * 96)
    print(f"ACCURACY GATE RESULTS: Kathbath Telugu Valid ({len(results)} clips)")
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
    print(f"Corpus-level WER degradation ((b) - (a)): {wer_diff:+.2f} percentage points (threshold: <= {MAX_CORPUS_WER_DEGRADATION:.2f})")
    print(f"Quality floor limit: <= {QUALITY_FLOOR_WER_PCT:.2f}% corpus WER for both models")
    print(f"Empty transcripts: nemo={empty_counts['nemo_empty']}, onnx={empty_counts['onnx_empty']}, either={empty_counts['either_empty']} (max allowed: {MAX_EMPTY_TRANSCRIPTS})")
    print(f"Runner CPU Latency for (b): Median = {median_sec_b:.3f} s/clip, 95th-percentile = {p95_sec_b:.3f} s/clip")
    print(f"Total reference volume: {total_ref_words} words, {total_ref_chars} characters across {len(results)} clips")
    print("=" * 96 + "\n")

    summary = {
        "dataset": "ai4bharat/Kathbath",
        "split": "valid",
        "clips": len(results),
        "gate_metric": "corpus_wer",
        "nemo_model_class": nemo_model_class,
        "quality_floor_wer_pct": QUALITY_FLOOR_WER_PCT,
        "threshold_points": MAX_CORPUS_WER_DEGRADATION,
        "max_empty_transcripts_allowed": MAX_EMPTY_TRANSCRIPTS,
        "total_reference_words": total_ref_words,
        "total_reference_chars": total_ref_chars,
        "empty_transcripts": empty_counts,
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
        "gate_passed": bool(passed),
        "gate_message": gate_msg,
        "installed_versions": installed_versions,
    }

    if args.out_results:
        args.out_results.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not passed:
        print(f"::error::ACCURACY GATE FAILED: {gate_msg}", file=sys.stderr)
        sys.exit(1)

    print(f"::notice::ACCURACY GATE PASSED: {gate_msg}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
