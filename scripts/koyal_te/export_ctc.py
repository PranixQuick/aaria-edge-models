#!/usr/bin/env python3
"""
Export NeMo Koyal te-120M CTC head to ONNX with sherpa-onnx metadata.

Process:
1. Loads .nemo checkpoint.
2. Extracts subsampling_factor directly from model config (asr_model.cfg.encoder.subsampling_factor,
   falling back to asr_model.encoder.subsampling_factor); stops with error if neither exists.
3. Configures model for CTC decoding (single language, NO slicing).
   For hybrid models (RNNT + CTC), strictly switches to CTC; stops with error on failure.
4. Extracts vocabulary from model tokenizer / decoder and writes tokens.txt with <blk>.
5. Exports CTC graph to ONNX.
6. Injects metadata properties required by sherpa-onnx NeMo CTC offline recognizer.
"""

import argparse
import os
import sys
from pathlib import Path


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


def extract_subsampling_factor(asr_model) -> int:
    """
    Read subsampling_factor from loaded model config:
    asr_model.cfg.encoder.subsampling_factor, falling back to asr_model.encoder.subsampling_factor.
    Raises RuntimeError if neither exists.
    """
    factor = None
    # 1. Check asr_model.cfg.encoder.subsampling_factor
    if hasattr(asr_model, "cfg") and asr_model.cfg is not None:
        encoder_cfg = getattr(asr_model.cfg, "encoder", None)
        if encoder_cfg is not None:
            if isinstance(encoder_cfg, dict) and "subsampling_factor" in encoder_cfg:
                factor = encoder_cfg["subsampling_factor"]
            elif hasattr(encoder_cfg, "subsampling_factor"):
                factor = getattr(encoder_cfg, "subsampling_factor")

    # 2. Fallback: asr_model.encoder.subsampling_factor
    if factor is None and hasattr(asr_model, "encoder") and asr_model.encoder is not None:
        if hasattr(asr_model.encoder, "subsampling_factor"):
            factor = getattr(asr_model.encoder, "subsampling_factor")

    if factor is None:
        raise RuntimeError(
            "Could not determine subsampling_factor from model config "
            "(neither asr_model.cfg.encoder.subsampling_factor nor asr_model.encoder.subsampling_factor exists)"
        )

    return int(factor)


def inject_sherpa_onnx_metadata(onnx_path: Path, vocab_size: int, subsampling_factor: int):
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

    # Read subsampling factor from model config
    try:
        subsampling_factor = extract_subsampling_factor(asr_model)
    except RuntimeError as e:
        print(f"::error::{e}", file=sys.stderr)
        sys.exit(f"Error: {e}")
    print(f"Model subsampling factor extracted: {subsampling_factor}")

    # Configure CTC decoding / export strictly for hybrid models
    if is_hybrid_model(asr_model):
        print(f"Hybrid model detected ({type(asr_model).__name__}): switching decoding strategy to CTC...")
        try:
            if not hasattr(asr_model, "change_decoding_strategy"):
                raise AttributeError(f"Hybrid model {type(asr_model).__name__} lacks change_decoding_strategy method")
            asr_model.change_decoding_strategy(decoder_type="ctc")
            print("Changed decoding strategy to CTC.")
        except Exception as e:
            print(f"::error::Failed to switch hybrid model decoding strategy to CTC: {e}", file=sys.stderr)
            sys.exit(1)

        try:
            if hasattr(asr_model, "set_export_config"):
                asr_model.set_export_config({"decoder_type": "ctc"})
                print("Set export config decoder_type='ctc'.")
        except Exception as e:
            print(f"::error::Failed to set export config decoder_type to CTC: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"Pure CTC model detected ({type(asr_model).__name__}); leaving decoding strategy as-is.")

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

    # Inject metadata with model's own subsampling_factor
    inject_sherpa_onnx_metadata(args.out_onnx, vocab_size, subsampling_factor)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
