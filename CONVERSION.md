# Model Conversion Process

## English (Moonshine Tiny)
The English model was downloaded directly as an int8 ONNX checkpoint from the sherpa-onnx releases:
sherpa-onnx-moonshine-tiny-en-int8.tar.bz2

## Hindi and Telugu (IndicConformer)
The original PyTorch checkpoints indicconformer_stt_hi_hybrid_rnnt_large.nemo and indicconformer_stt_te_hybrid_rnnt_large.nemo were downloaded from AI4Bharat.

We used the NeMo export scripts (xport-onnx-ctc-non-streaming.py) to extract the CTC projection weights and convert the model to ONNX.
Because the original model uses a single unified CTC decoder for 22 languages (
um_classes: 5633), we surgically sliced the ctc_decoder.decoder_layers.0.weight and ias matrices in PyTorch to keep only the 256 tokens corresponding to the target language (offset 1536 for Hindi, 5120 for Telugu), plus the <blk> token (index 5632). 
We generated a 257-token 	okens.txt file by extracting the vocabulary directly from the .nemo model's BPE SentencePiece 	okenizer.model for the corresponding language index.
The ONNX model was statically patched to set subsampling_factor: 4 in the ONNX metadata since it's a standard Conformer. Finally, the model was converted to model.int8.onnx using dynamic INT8 quantization.

## Accuracy
Evaluated on the Google FLEURS test split using sherpa_onnx.OfflineRecognizer:
| Language | Model Repo/File | Model Size | WER |
| --- | --- | --- | --- |
| English (Moonshine) | sherpa-onnx-moonshine-tiny-en-int8 | ~50M | 0.143 |
| Hindi | indicconformer_stt_hi_hybrid_rnnt_large | ~130M | 0.133 |
| Telugu | indicconformer_stt_te_hybrid_rnnt_large | ~130M | 0.230 |
