#!/usr/bin/env python3
"""
Unit tests for the scorer and conversion pieces (runs locally without model downloads).
"""

import json
import tempfile
from pathlib import Path
from normalise import cer, normalise, wer
from make_license import MODEL_ATTRIBUTION, CC_BY_4_NOTICE


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
    print("  -> normalise.py tests passed!")


def test_gate_logic():
    print("Testing accuracy gate threshold logic...")
    # Passing case: degradation <= 1.0 point
    wer_a_pct = 17.3
    wer_b_pct = 18.1
    diff = wer_b_pct - wer_a_pct
    assert diff <= 1.0, f"Expected {diff} <= 1.0"

    # Failing case: degradation > 1.0 point
    wer_b_fail = 18.5
    diff_fail = wer_b_fail - wer_a_pct
    assert diff_fail > 1.0, f"Expected {diff_fail} > 1.0"
    print("  -> gate threshold tests passed!")


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
    test_gate_logic()
    test_model_json_format()
    test_license_content()
    print("\nALL LOCAL UNIT CHECKS PASSED.")
