# Remote Telugu Model Switch (QuietKeep Models Repo)

## Overview

QuietKeep edge devices running Aaria Edge download their speech recognition models and pre-rendered replies from a pinned, cryptographically signed release list:

```
https://github.com/PranixQuick/aaria-edge-models/releases/download/models-2026-10/manifest.json
https://github.com/PranixQuick/aaria-edge-models/releases/download/models-2026-10/manifest.json.sig
```

Each phone checks this manifest during its nightly sync window over Wi-Fi. It verifies the ECDSA P-256 signature using the pinned public key (`manifest_public_key.b64`), checks SHA-256 hashes of all assets, and downloads only the model files for its active language (`en`, `hi`, `te`).

## The Telugu Model Options

Both candidate models are non-streaming **CTC** architecture (`type: "ctc"`) evaluated via `sherpa_onnx.OfflineRecognizer`:

1. **AI4Bharat IndicConformer (Current Baseline)**:
   - Checkpoint: AI4Bharat IndicConformer CTC Telugu
   - Release assets in `models-2026-10`:
     - `te__model.int8.onnx` (137,563,742 bytes, SHA-256: `13774a654ce5152796e1129db880a721f7c1ec4cd6301e627e258336ec9ac49e`)
     - `te__model.json` (71 bytes)
     - `te__tokens.txt` (2,963 bytes)

2. **Adalat AI Koyal te-120M (Quantized INT8)**:
   - Checkpoint: `adalat-ai/koyal-te-120m-1.0` (MahaDhwani architecture with CTC head)
   - Draft release assets in `models-2026-10-koyal-te`:
     - `te-koyal__model.int8.onnx` (138,417,172 bytes, SHA-256: `8bbb11bd7cfe543148c49d24dba845395a8a683f82a63841ad6b90589e562781`)
     - `te-koyal__model.json` (75 bytes)
     - `te-koyal__tokens.txt` (5,623 bytes)
     - `te-koyal__LICENSE-koyal-te-120m.txt` (13,076 bytes)

Because both models share the exact same `type: "ctc"` schema and role files (`model.int8.onnx`, `model.json`, `tokens.txt`), QuietKeep phone firmware (`CommandRecognizer` CTC path and `OfflineNemoEncDecCtcModelConfig`) requires **zero application code changes**. The model switch is entirely governed by which files `models.te` points to in `manifest.json`.

---

## Operating the Switch: `.github/workflows/te-switch.yml`

The switch workflow is manual-only (`workflow_dispatch`) with two inputs:
- `target`: `koyal` or `indicconformer` (required)
- `dry_run`: boolean, **default `true`**

### 1. Dry Run (Default)
When `dry_run: true`:
- Downloads current `manifest.json`, `.sig`, and public key from `models-2026-10`.
- Verifies current release signature.
- Downloads Koyal draft assets (if `target=koyal`).
- Computes new manifest, enforces all SyncManifest rules, and logs a unified diff and total byte footprint.
- Uploads both current and proposed manifests as workflow artifacts.
- **Modifies nothing in releases or on phones.**

### 2. Switching to Koyal
To switch QuietKeep phones to Koyal te-120M:
1. Navigate to Actions -> `te-switch`.
2. Select `target: koyal`.
3. Set `dry_run: false`.
4. Trigger workflow.

Workflow actions:
- Copies `te-koyal__*` assets from draft release `models-2026-10-koyal-te` into published release `models-2026-10` (`--clobber` only on `te-koyal__*` names; never touches existing assets).
- Re-downloads and verifies SHA-256 of uploaded assets.
- Signs new `manifest.json` using repository secret `AARIA_MODELS_SIGNING_KEY` (ECDSA P-256 / SHA-256 raw DER).
- Validates signature with public key before publishing.
- Uploads `manifest.json` and `manifest.json.sig` together to `models-2026-10`.
- Preserves audit copy as `manifest-previous.json` and `manifest-previous.json.sig`.
- Re-downloads from public URL to verify end-to-end integrity.

Phones pick up Koyal at their next nightly sync over Wi-Fi.

### 3. Switching Back to IndicConformer (Reversion)
To revert to the original IndicConformer:
1. Navigate to Actions -> `te-switch`.
2. Select `target: indicconformer`.
3. Set `dry_run: false`.
4. Trigger workflow.

The workflow restores `models.te` from the committed, byte-for-byte baseline `manifests/te-indicconformer.json`, signs the manifest, and publishes it.

---

## SyncManifest Security & Constraint Rules

The builder script (`scripts/switch/build_te_switch.py`) strictly enforces:
- **Protocol**: `https://` only for all asset URLs.
- **Path Traversal**: Filenames must not contain `..`, `/`, or `\\`.
- **Per-file limit**: Maximum 400 MB per single file (largest model file is ~138 MB).
- **Total model limit**: Sum of all model files across all languages must stay <= 600 MB (current total with Koyal: ~489 MB).
- **Version**: `version: 1`.
- **Integrity**: Every non-target key (`en`, `hi`, `replies`) remains byte-identical.

---

## Governance & Product Decision

Turning the switch on (`dry_run: false`) is the founder's product decision, to be made after:
1. Reviewing the same-test comparative evaluation results from `te-compare.yml` (Kathbath Telugu 60 clips).
2. Reviewing mixed-recording real-world evaluation clips.
