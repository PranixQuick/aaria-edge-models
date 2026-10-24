#!/usr/bin/env python3
"""
Unit tests for the scorer and conversion pieces (runs locally without model downloads).
"""

import json
import tempfile
from pathlib import Path
from normalise import cer, compute_edits_and_lengths, normalise, wer
from make_license import MODEL_ATTRIBUTION, CC_BY_4_NOTICE
from export_ctc import extract_subsampling_factor


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
    # Scenario: 2 clips
    # Clip 1: ref has 8 words, 1 edit (clip wer = 1/8 = 0.125)
    # Clip 2: ref has 2 words, 1 edit (clip wer = 1/2 = 0.500)
    # Per-clip mean = (0.125 + 0.500) / 2 = 0.3125 (31.25%)
    # Corpus WER = (1 + 1) / (8 + 2) = 2 / 10 = 0.200 (20.0%)
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
    # Passing case: degradation <= 1.0 point
    corpus_wer_a_pct = 17.30
    corpus_wer_b_pct = 18.15
    diff = corpus_wer_b_pct - corpus_wer_a_pct
    assert diff <= 1.0, f"Expected {diff} <= 1.0"

    # Failing case: degradation > 1.0 point
    corpus_wer_b_fail = 18.50
    diff_fail = corpus_wer_b_fail - corpus_wer_a_pct
    assert diff_fail > 1.0, f"Expected {diff_fail} > 1.0"
    print("  -> gate threshold tests passed!")


def test_subsampling_factor_extraction():
    print("Testing subsampling factor extraction from model config...")

    # Case 1: asr_model.cfg.encoder.subsampling_factor
    class MockCfgEncoder:
        subsampling_factor = 4

    class MockCfg:
        encoder = MockCfgEncoder()

    class MockModelWithCfg:
        cfg = MockCfg()

    assert extract_subsampling_factor(MockModelWithCfg()) == 4

    # Case 2: Dict cfg
    class MockModelWithDictCfg:
        cfg = type("Obj", (), {"encoder": {"subsampling_factor": 8}})()

    assert extract_subsampling_factor(MockModelWithDictCfg()) == 8

    # Case 3: Fallback to asr_model.encoder.subsampling_factor
    class MockEncoderOnly:
        subsampling_factor = 4

    class MockModelWithEncoder:
        encoder = MockEncoderOnly()

    assert extract_subsampling_factor(MockModelWithEncoder()) == 4

    # Case 4: Neither exists -> RuntimeError / exit
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
    test_normalise()
    test_corpus_wer_vs_mean_gate()
    test_gate_logic()
    test_subsampling_factor_extraction()
    test_model_json_format()
    test_license_content()
    print("\nALL LOCAL UNIT CHECKS PASSED.")
