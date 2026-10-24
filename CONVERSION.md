# Model Conversion Process

## English (Moonshine Tiny)
The English model was downloaded directly as an int8 ONNX checkpoint from the sherpa-onnx releases:
sherpa-onnx-moonshine-tiny-en-int8.tar.bz2

## Hindi and Telugu (IndicConformer)
The original PyTorch checkpoints indicconformer_stt_hi_hybrid_rnnt_large.nemo and indicconformer_stt_te_hybrid_rnnt_large.nemo were downloaded from AI4Bharat.

We used the NeMo export scripts (export-onnx-ctc-non-streaming.py) to extract the CTC projection weights and convert the model to ONNX.
Because the original model uses a single unified CTC decoder for 22 languages (num_classes: 5633), we surgically sliced the ctc_decoder.decoder_layers.0.weight and bias matrices in PyTorch to keep only the 256 tokens corresponding to the target language (offset 1536 for Hindi, 5120 for Telugu), plus the <blk> token (index 5632). 
We generated a 257-token tokens.txt file by extracting the vocabulary directly from the .nemo model's BPE SentencePiece tokenizer.model for the corresponding language index.
The ONNX model was statically patched to set subsampling_factor: 4 in the ONNX metadata since it's a standard Conformer. Finally, the model was converted to model.int8.onnx using dynamic INT8 quantization.

## Telugu (Adalat AI Koyal te-120M)
The Telugu on-phone recogniser uses Adalat AI's Koyal te-120M (Hugging Face `adalat-ai/koyal-te-120m-1.0`, CC-BY-4.0; encoder from AI4Bharat MahaDhwani, MIT).

Conversion method:
1. Checkpoint: The gated checkpoint `koyal-te-120m-1.0.nemo` and model card are retrieved using the repository `HF_TOKEN` secret.
2. CTC Export: The model is a dedicated single-language Telugu model, so no vocabulary slicing is required. The non-streaming CTC head is exported to ONNX via NeMo export. `tokens.txt` is extracted from the model's SentencePiece tokenizer and appended with the `<blk>` token at the final index.
3. Metadata Injection: NeMo CTC metadata required by `sherpa-onnx` is injected into the ONNX graph:
   - `vocab_size`: total tokens in `tokens.txt` (including `<blk>`)
   - `subsampling_factor`: 4 (Conformer acoustic encoder)
   - `normalize_type`: `per_feature`
   - `model_type`: `EncDecCTCModelBPE`
   - `version`: `1`
   - `model_author`: `nemo`
4. Quantization: The exported graph is dynamically quantized to INT8 (`model.int8.onnx`) using `onnxruntime.quantization` with `QuantType.QUInt8` and `per_channel=True`.
5. Configuration: `model.json` is generated with format `{"type":"ctc","model":"model.int8.onnx","tokens":"tokens.txt"}` for the phone plugin's `CommandRecognizer`.
6. Accuracy Gate: Evaluated on the AI4Bharat Kathbath Telugu validation split (60 clips, held-out validation only, never train). Transcribed by both the original `.nemo` model and the `model.int8.onnx` through `sherpa_onnx.OfflineRecognizer` (sherpa-onnx 1.13.8) using identical normalisation. The workflow enforces that INT8 WER must not degrade by more than 1.0 percentage point compared to the unquantized NeMo baseline.
7. Release Packaging: If the gate passes, artifacts are uploaded to draft release `models-2026-10-koyal-te` with prefix `te-koyal__`:
   - `te-koyal__model.int8.onnx`
   - `te-koyal__tokens.txt`
   - `te-koyal__model.json`
   - `te-koyal__LICENSE-koyal-te-120m.txt`

Accuracy:
- Kathbath Telugu valid (60 clips): not yet run (requires founder to provide HF_TOKEN secret with accepted model and dataset access)

## Accuracy
Evaluated on the Google FLEURS test split using sherpa_onnx.OfflineRecognizer:
| Language | Model Repo/File | Model Size | WER |
| --- | --- | --- | --- |
| English (Moonshine) | sherpa-onnx-moonshine-tiny-en-int8 | ~50M | 0.143 |
| Hindi | indicconformer_stt_hi_hybrid_rnnt_large | ~130M | 0.133 |
| Telugu | indicconformer_stt_te_hybrid_rnnt_large | ~130M | 0.230 |
| Telugu (Koyal te-120M) | adalat-ai/koyal-te-120m-1.0 | ~120M | not yet run |
