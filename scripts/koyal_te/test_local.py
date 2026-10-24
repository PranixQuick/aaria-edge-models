#!/usr/bin/env python3
"""
Unit tests for the scorer, gate hardening, and conversion pieces (runs locally without model downloads).
"""

import json
import os
import py_compile
import subprocess
import sys
import tempfile
from pathlib import Path
from normalise import cer, compute_edits_and_lengths, normalise, wer
from make_license import MODEL_ATTRIBUTION, CC_BY_4_NOTICE
from export_ctc import extract_subsampling_factor, is_hybrid_model as export_is_hybrid
from score_gate import (
    REQUIRED_MANIFEST_ENTRIES,
    MAX_EMPTY_TRANSCRIPTS,
    MAX_CORPUS_WER_DEGRADATION,
    QUALITY_FLOOR_WER_PCT,
    check_manifest_entries,
    check_gate_rules,
    is_hybrid_model,
)


def make_mock_result(audio_name, ref_text, hyp_a, hyp_b):
    edits_a = compute_edits_and_lengths(ref_text, hyp_a)
    edits_b = compute_edits_and_lengths(ref_text, hyp_b)
    return {
        "audio": audio_name,
        "reference": ref_text,
        "hyp_a_nemo": hyp_a,
        "wer_a": edits_a["clip_wer"],
        "cer_a": edits_a["clip_cer"],
        "edits_a": edits_a,
        "hyp_b_onnx": hyp_b,
        "wer_b": edits_b["clip_wer"],
        "cer_b": edits_b["clip_cer"],
        "edits_b": edits_b,
        "latency_b_sec": 0.05,
    }


def test_all_scripts_compile():
    print("Testing that every script in scripts/koyal_te compiles...")
    script_dir = Path(__file__).resolve().parent
    py_files = sorted(script_dir.glob("*.py"))
    assert len(py_files) > 0, "No Python files found in scripts/koyal_te"
    for py_file in py_files:
        print(f"  -> py_compile.compile: {py_file.name}")
        py_compile.compile(str(py_file), doraise=True)
    print(f"  -> all {len(py_files)} scripts compiled successfully!")


def test_fetch_kathbath_import():
    print("Testing that fetch_kathbath can be imported as a module without exiting...")
    import fetch_kathbath
    assert hasattr(fetch_kathbath, "main"), "fetch_kathbath must define main"
    assert hasattr(fetch_kathbath, "to_16k_mono_file"), "fetch_kathbath must define to_16k_mono_file"
    assert fetch_kathbath.DEFAULT_COUNT == 60, "fetch_kathbath DEFAULT_COUNT must be 60"
    print("  -> fetch_kathbath imported cleanly as a module!")

def test_fetch_kathbath_exit_message():
    print("Testing that fetch_kathbath prints exit reason to stderr when stopping...")
    script_path = Path(__file__).resolve().parent / "fetch_kathbath.py"
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        out_dir = tmp_path / "audio"
        manifest = tmp_path / "m.jsonl"
        env = dict(os.environ)
        for token_var in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACEHUB_API_TOKEN"):
            env.pop(token_var, None)
        cmd = [
            sys.executable,
            str(script_path),
            "--count",
            "1",
            "--out-dir",
            str(out_dir),
            "--manifest",
            str(manifest),
        ]
        proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
        assert proc.returncode == 1, f"Expected returncode 1, got {proc.returncode}"
        assert "HF_TOKEN secret missing" in proc.stderr, (
            f"Expected 'HF_TOKEN secret missing' in stderr, got:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    print("  -> fetch_kathbath exit code 1 and exit reason printed to stderr as expected!")



def test_normalise():
    print("Testing normalise.py...")
    # Basic normalization & number composition
    assert normalise("Paid Rs.1850 for petrol") == "paid 1850 for petrol"
    assert normalise("Paid one thousand eight hundred and fifty rupees") == "paid 1850"
    assert wer("Paid one thousand eight hundred and fifty rupees", "Paid Rs.1850") == 0.0

    # Telugu script
    assert normalise("రేపు ఉదయం పది గంటలకు కరెంట్ బిల్లు కట్టాలి।") == "రేపు ఉదయం 10 గంటలకు కరెంట్ బిల్లు కట్టాలి"

    # Idempotency
    t = "రేపు ఉదయం పది గంటలకు కరెంట్ బిల్లు కట్టాలి"
    assert normalise(normalise(t)) == normalise(t)

    # Detailed edit and length calculations with non-number words
    res = compute_edits_and_lengths("apple banana orange", "apple banana")
    assert res["ref_words"] == 3
    assert res["word_edits"] == 1
    assert abs(res["clip_wer"] - (1.0 / 3.0)) < 1e-4

    print("  -> normalise.py tests passed!")


def test_corpus_wer_vs_mean_gate():
    print("Testing corpus-level WER vs per-clip mean calculation...")
    clip1 = compute_edits_and_lengths("call Ravi tomorrow at noon for lunch meeting", "call Ravi tomorrow at noon for lunch dinner")
    clip2 = compute_edits_and_lengths("send message", "send")

    assert clip1["ref_words"] == 8
    assert clip1["word_edits"] == 1
    assert abs(clip1["clip_wer"] - 0.125) < 1e-4

    assert clip2["ref_words"] == 2
    assert clip2["word_edits"] == 1
    assert abs(clip2["clip_wer"] - 0.500) < 1e-4

    mean_wer = (clip1["clip_wer"] + clip2["clip_wer"]) / 2.0
    corpus_wer = (clip1["word_edits"] + clip2["word_edits"]) / (clip1["ref_words"] + clip2["ref_words"])

    assert abs(mean_wer - 0.3125) < 1e-4
    assert abs(corpus_wer - 0.200) < 1e-4
    print("  -> corpus WER correctly computed as total edits / total ref words (20.0% vs mean 31.25%)")


def test_gate_logic():
    print("Testing accuracy gate threshold logic...")
    corpus_wer_a_pct = 17.30
    corpus_wer_b_pct = 18.15
    diff = corpus_wer_b_pct - corpus_wer_a_pct
    assert diff <= 1.0, f"Expected {diff} <= 1.0"

    corpus_wer_b_fail = 18.50
    diff_fail = corpus_wer_b_fail - corpus_wer_a_pct
    assert diff_fail > 1.0, f"Expected {diff_fail} > 1.0"
    print("  -> gate threshold tests passed!")


def test_manifest_entries_count():
    print("Testing manifest entry count validation (59 entries -> fail, 60 entries -> pass)...")
    entries_59 = [{"audio": f"clip_{i:03d}.wav", "text": "foo"} for i in range(59)]
    try:
        check_manifest_entries(entries_59, REQUIRED_MANIFEST_ENTRIES)
        assert False, "Expected ValueError for 59 entries"
    except ValueError as e:
        assert "60" in str(e) and "59" in str(e)

    entries_60 = [{"audio": f"clip_{i:03d}.wav", "text": "foo"} for i in range(60)]
    check_manifest_entries(entries_60, REQUIRED_MANIFEST_ENTRIES)  # should not raise
    print("  -> 59 entries correctly fails; 60 entries passes!")


def test_zero_reference_words_fail():
    print("Testing zero reference words validation (0 reference words -> fail)...")
    results_zero = [
        make_mock_result(f"clip_{i:03d}.wav", "", "something", "something")
        for i in range(60)
    ]
    passed, msg, stats = check_gate_rules(results_zero)
    assert not passed, "Expected failure on zero reference volume"
    assert "zero reference" in msg.lower()
    print(f"  -> zero reference correctly failed with message: {msg}")


def test_quality_floor_wer_fail():
    print("Testing quality floor validation (both models 100% WER with 0.0 difference -> FAIL)...")
    results_100 = [
        make_mock_result(f"clip_{i:03d}.wav", "apple banana orange", "car bus train", "car bus train")
        for i in range(60)
    ]
    passed, msg, stats = check_gate_rules(results_100)
    assert not passed, "Expected quality floor failure when both models have 100% WER"
    assert abs(stats["corpus_wer_a"] - 100.0) < 1e-4
    assert abs(stats["corpus_wer_b"] - 100.0) < 1e-4
    assert abs(stats["wer_diff"] - 0.0) < 1e-4  # Difference is 0.0, but quality floor must catch it!
    assert "quality floor" in msg.lower()
    print(f"  -> 100% WER with 0.0 diff correctly failed quality floor: {msg}")


def test_empty_transcripts_fail():
    print("Testing empty transcripts validation (empty-transcript count above 5 -> fail)...")
    # 60 clips: 6 clips produce empty transcripts (> 5 limit)
    results_empty = []
    for i in range(60):
        ref = "రేపు ఉదయం పది గంటలకు కరెంట్ బిల్లు కట్టాలి"
        if i < 6:
            results_empty.append(make_mock_result(f"clip_{i:03d}.wav", ref, ref, ""))
        else:
            results_empty.append(make_mock_result(f"clip_{i:03d}.wav", ref, ref, ref))

    passed, msg, stats = check_gate_rules(results_empty)
    assert not passed, "Expected failure when >5 clips produce empty transcripts"
    assert stats["empty_counts"]["either_empty"] == 6
    assert "empty" in msg.lower()
    print(f"  -> 6 empty transcripts correctly failed with: {msg}")

    # Boundary test: exactly 5 empty transcripts does not fail the empty transcript check
    results_5_empty = []
    for i in range(60):
        ref = "apple banana cherry date elderberry fig"
        if i < 5:
            results_5_empty.append(make_mock_result(f"clip_{i:03d}.wav", ref, ref, ""))
        else:
            results_5_empty.append(make_mock_result(f"clip_{i:03d}.wav", ref, ref, ref))
    passed_5, msg_5, stats_5 = check_gate_rules(results_5_empty)
    assert stats_5["empty_counts"]["either_empty"] == 5
    assert "empty" not in msg_5.lower()
    print("  -> boundary test: <=5 empty transcripts passes empty check")


def test_normal_case_pass():
    print("Testing normal evaluation case (a normal case -> pass)...")
    results_normal = []
    for i in range(60):
        ref = "apple banana cherry date elderberry fig"
        # Model a: 1 word substituted on every clip -> WER = 1/6 = 16.67%
        hyp_a = "apple banana cherry date elderberry grape"
        # Model b: on 3 clips 2 errors, on 57 clips 1 error -> WER = (57*1 + 3*2) / 360 = 63 / 360 = 17.50%
        if i < 3:
            hyp_b = "apple banana cherry mango elderberry grape"
        else:
            hyp_b = "apple banana cherry date elderberry grape"
        results_normal.append(make_mock_result(f"clip_{i:03d}.wav", ref, hyp_a, hyp_b))

    passed, msg, stats = check_gate_rules(results_normal)
    assert passed, f"Expected normal case to pass, got: {msg}"
    assert stats["corpus_wer_a"] < QUALITY_FLOOR_WER_PCT
    assert stats["corpus_wer_b"] < QUALITY_FLOOR_WER_PCT
    assert stats["wer_diff"] <= MAX_CORPUS_WER_DEGRADATION
    assert stats["empty_counts"]["either_empty"] == 0
    print(
        f"  -> normal case passed! (corpus WER a={stats['corpus_wer_a']:.2f}%, "
        f"b={stats['corpus_wer_b']:.2f}%, diff={stats['wer_diff']:+.2f})"
    )


def test_hybrid_model_detection():
    print("Testing hybrid model detection logic...")
    class EncDecCTCModelBPE:
        pass
    assert not is_hybrid_model(EncDecCTCModelBPE())
    assert not export_is_hybrid(EncDecCTCModelBPE())

    class EncDecHybridRNNTCTCBPEModel:
        pass
    assert is_hybrid_model(EncDecHybridRNNTCTCBPEModel())
    assert export_is_hybrid(EncDecHybridRNNTCTCBPEModel())

    class ModelWithDualDecoders:
        ctc_decoder = object()
        decoder = object()
    assert is_hybrid_model(ModelWithDualDecoders())
    assert export_is_hybrid(ModelWithDualDecoders())

    class ModelWithJoint:
        ctc_decoder = object()
        joint = object()
    assert is_hybrid_model(ModelWithJoint())
    assert export_is_hybrid(ModelWithJoint())
    print("  -> hybrid model detection tests passed!")


def test_subsampling_factor_extraction():
    print("Testing subsampling factor extraction from model config...")
    class MockCfgEncoder:
        subsampling_factor = 4

    class MockCfg:
        encoder = MockCfgEncoder()

    class MockModelWithCfg:
        cfg = MockCfg()

    assert extract_subsampling_factor(MockModelWithCfg()) == 4

    class MockModelWithDictCfg:
        cfg = type("Obj", (), {"encoder": {"subsampling_factor": 8}})()

    assert extract_subsampling_factor(MockModelWithDictCfg()) == 8

    class MockEncoderOnly:
        subsampling_factor = 4

    class MockModelWithEncoder:
        encoder = MockEncoderOnly()

    assert extract_subsampling_factor(MockModelWithEncoder()) == 4

    class EmptyModel:
        pass

    try:
        extract_subsampling_factor(EmptyModel())
        assert False, "Should have raised exception for empty model"
    except RuntimeError:
        pass

    print("  -> subsampling factor extraction tests passed!")


def test_model_json_format():
    print("Testing model.json format...")
    config = {
        "type": "ctc",
        "model": "model.int8.onnx",
        "tokens": "tokens.txt",
    }
    dumped = json.dumps(config)
    loaded = json.loads(dumped)
    assert loaded["type"] == "ctc"
    assert loaded["model"] == "model.int8.onnx"
    assert loaded["tokens"] == "tokens.txt"
    print("  -> model.json format tests passed!")


def test_license_content():
    print("Testing license content requirements...")
    assert "CC-BY-4.0" in CC_BY_4_NOTICE
    assert "encoder from AI4Bharat MahaDhwani, MIT" in MODEL_ATTRIBUTION
    assert "Adalat AI" in MODEL_ATTRIBUTION
    print("  -> license content tests passed!")


if __name__ == "__main__":
    test_all_scripts_compile()
    test_fetch_kathbath_import()
    test_fetch_kathbath_exit_message()
    test_normalise()
    test_corpus_wer_vs_mean_gate()
    test_gate_logic()
    test_manifest_entries_count()
    test_zero_reference_words_fail()
    test_quality_floor_wer_fail()
    test_empty_transcripts_fail()
    test_normal_case_pass()
    test_hybrid_model_detection()
    test_subsampling_factor_extraction()
    test_model_json_format()
    test_license_content()
    print("\nALL LOCAL UNIT CHECKS PASSED.")
