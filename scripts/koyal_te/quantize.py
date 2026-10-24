#!/usr/bin/env python3
"""
Dynamic INT8 quantization of exported ONNX model and model.json generation.

Outputs:
- model.int8.onnx (QUInt8 dynamic quantization)
- model.json = {"type":"ctc","model":"model.int8.onnx","tokens":"tokens.txt"}
"""

import argparse
import json
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--in-onnx", required=True, type=Path, help="Input model.onnx")
    parser.add_argument("--out-onnx", required=True, type=Path, help="Output model.int8.onnx")
    parser.add_argument("--out-json", required=True, type=Path, help="Output model.json")
    args = parser.parse_args()

    if not args.in_onnx.exists():
        sys.exit(f"Input ONNX file does not exist: {args.in_onnx}")

    args.out_onnx.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.parent.mkdir(parents=True, exist_ok=True)

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError:
        sys.exit("onnxruntime is required: pip install onnxruntime")

    print(f"Applying dynamic INT8 quantization: {args.in_onnx} -> {args.out_onnx}...")
    quantize_dynamic(
        model_input=str(args.in_onnx),
        model_output=str(args.out_onnx),
        per_channel=True,
        weight_type=QuantType.QUInt8,
    )

    in_size_mb = args.in_onnx.stat().st_size / (1024 * 1024)
    out_size_mb = args.out_onnx.stat().st_size / (1024 * 1024)
    print(f"Quantization complete! {in_size_mb:.2f} MB -> {out_size_mb:.2f} MB ({out_size_mb / in_size_mb * 100:.1f}%)")

    # Generate model.json exactly as required by the phone plugin CommandRecognizer
    model_config = {
        "type": "ctc",
        "model": args.out_onnx.name,
        "tokens": "tokens.txt",
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(model_config, f, indent=2)
    print(f"Wrote {args.out_json}: {model_config}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
