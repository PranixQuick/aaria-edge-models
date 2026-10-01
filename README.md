# aaria-edge-models

Speech model files that the Aaria Edge plugin downloads to phones at night (Wi-Fi + charging), plus their licences.
Code lives in PranixQuick/aaria-edge. This repository holds only release assets, licences and the signed file list.

| Language | Model | Upstream | Licence |
|---|---|---|---|
| en | Moonshine tiny, int8 (sherpa-onnx packaging) | moonshine-ai/moonshine via k2-fsa/sherpa-onnx asr-models release | MIT |
| hi | IndicConformer Hindi, converted by Pranix to sherpa-onnx int8 CTC | AI4Bharat/IndicConformerASR | MIT |
| te | IndicConformer Telugu, converted by Pranix to sherpa-onnx int8 CTC | AI4Bharat/IndicConformerASR | MIT |

Each release contains `manifest.json` (every file's link, size and SHA-256) and `manifest.json.sig`
(ECDSA P-256 signature). Phones refuse any file list whose signature does not verify against the public key
built into the app, and any file whose fingerprint does not match.

Conversion and accuracy checks: see CONVERSION.md. A converted model is published only if its word error on the
same 30 FLEURS test clips is within 1 point of the reference measured on 30 Sep 2026 (hi 12.3%, te 23.0%).
