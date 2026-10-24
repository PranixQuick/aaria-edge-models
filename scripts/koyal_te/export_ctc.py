#!/usr/bin/env python3
"""
Export NeMo Koyal te-120M CTC head to ONNX with sherpa-onnx metadata.

Process:
1. Loads .nemo checkpoint.
2. Configures model for CTC decoding (single language, NO slicing).
3. Extracts vocabulary from model tokenizer / decoder and writes tokens.txt with <blk>.
4. Exports CTC graph to ONNX.
5. Injects metadata properties required by sherpa-onnx NeMo CTC offline recognizer.
"""

import argparse
import os
import sys
from pathlib import Path


def extract_vocabulary(asr_model):
    """Extract vocabulary list from NeMo model (tokenizer or CTC decoder)."""
    if hasattr(asr_model, "tokenizer") and hasattr(asr_model.tokenizer, "vocab_size"):
        try:
            vocab = [asr_model.tokenizer.ids_to_tokens([i])[0] for i in range(asr_model.tokenizer.vocab_size)]
            if vocab:
                return vocab
        except Exception:
            pass
    if hasattr(asr_model, "ctc_decoder") and hasattr(asr_model.ctc_decoder, "vocabulary"):
        return list(asr_model.ctc_decoder.vocabulary)
    if hasattr(asr_model, "decoder") and hasattr(asr_model.decoder, "vocabulary"):
        return list(asr_model.decoder.vocabulary)
    raise RuntimeError("Could not extract vocabulary from model tokenizer or decoder")


def write_tokens_file(vocab, out_path: Path):
    """Write tokens.txt in sherpa-onnx CTC format: '<token> <id>' ending with '<blk> <N>'."""
    with open(out_path, "w", encoding="utf-8") as f:
        for idx, token in enumerate(vocab):
            f.write(f"{token} {idx}\n")
        f.write(f"<blk> {len(vocab)}\n")
    total_tokens = len(vocab) + 1
    print(f"Wrote {total_tokens} tokens to {out_path} (vocabulary: {len(vocab)}, blank: 1)")
    return total_tokens


def inject_sherpa_onnx_metadata(onnx_path: Path, vocab_size: int, subsampling_factor: int = 4):
    """Inject required metadata properties into ONNX model graph for sherpa-onnx."""
    import onnx

    metadata = {
        "vocab_size": str(vocab_size),
        "normalize_type": "per_feature",
        "subsampling_factor": str(subsampling_factor),
        "model_type": "EncDecCTCModelBPE",
        "version": "1",
        "model_author": "nemo",
        "comment": "adalat-ai/koyal-te-120m-1.0 exported for sherpa-onnx",
    }

    model = onnx.load(str(onnx_path))
    # Remove existing properties with matching keys
    keys_to_set = set(metadata.keys())
    existing = [p for p in model.metadata_props if p.key in keys_to_set]
    for prop in existing:
        model.metadata_props.remove(prop)

    for k, v in metadata.items():
        prop = model.metadata_props.add()
        prop.key = k
        prop.value = v

    onnx.save(model, str(onnx_path))
    print(f"Successfully injected sherpa-onnx metadata into {onnx_path}: {metadata}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nemo-path", required=True, type=Path, help="Path to input .nemo checkpoint")
    parser.add_argument("--out-onnx", required=True, type=Path, help="Path to output model.onnx")
    parser.add_argument("--out-tokens", required=True, type=Path, help="Path to output tokens.txt")
    parser.add_argument("--subsampling-factor", type=int, default=4, help="Subsampling factor (default: 4)")
    args = parser.parse_args()

    if not args.nemo_path.exists():
        sys.exit(f"Input .nemo file does not exist: {args.nemo_path}")

    args.out_onnx.parent.mkdir(parents=True, exist_ok=True)
    args.out_tokens.parent.mkdir(parents=True, exist_ok=True)

    try:
        import nemo.collections.asr as nemo_asr
    except ImportError:
        sys.exit("nemo_toolkit[asr] is required: pip install 'nemo_toolkit[asr]'")

    print(f"Loading NeMo checkpoint from {args.nemo_path} (on CPU)...")
    asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=str(args.nemo_path), map_location="cpu")
    asr_model.eval()

    # Configure CTC decoding / export
    if hasattr(asr_model, "change_decoding_strategy"):
        try:
            asr_model.change_decoding_strategy(decoder_type="ctc")
            print("Changed decoding strategy to CTC.")
        except Exception as e:
            print(f"Notice: change_decoding_strategy: {e}")

    if hasattr(asr_model, "set_export_config"):
        try:
            asr_model.set_export_config({"decoder_type": "ctc"})
            print("Set export config decoder_type='ctc'.")
        except Exception as e:
            print(f"Notice: set_export_config: {e}")

    # Set preprocessor parameters for inference stability
    if hasattr(asr_model, "preprocessor") and hasattr(asr_model.preprocessor, "featurizer"):
        featurizer = asr_model.preprocessor.featurizer
        if hasattr(featurizer, "dither"):
            featurizer.dither = 0.0
        if hasattr(featurizer, "pad_to"):
            featurizer.pad_to = 0

    # Extract vocabulary and write tokens.txt
    vocab = extract_vocabulary(asr_model)
    vocab_size = write_tokens_file(vocab, args.out_tokens)

    # Export ONNX model
    print(f"Exporting model to {args.out_onnx}...")
    asr_model.export(str(args.out_onnx))
    print(f"Export completed. Size: {args.out_onnx.stat().st_size / (1024 * 1024):.2f} MB")

    # Inject metadata
    inject_sherpa_onnx_metadata(args.out_onnx, vocab_size, args.subsampling_factor)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
