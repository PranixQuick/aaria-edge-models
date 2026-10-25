# Remote Telugu Model Switch (QuietKeep Models Repo)

<!-- Note: sign-manifest.yml should not be run at the same time as te-switch.yml (they share the models-manifest release target). -->

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

### Concurrency Protection

`te-switch.yml` defines top-level concurrency:
```yaml
concurrency:
  group: models-manifest
  cancel-in-progress: false
```
This guarantees that no two switch runs execute concurrently or interleave manifest modifications.

> [!NOTE]
> Workflow `sign-manifest.yml` targets the same release manifest (`models-2026-10`) and should not be run at the same time as `te-switch.yml`.

### Two-Job Permissions Architecture

To protect production release assets and secrets, the workflow is split into two distinct jobs:

1. **`plan` job**:
   - Runs on every invocation (both dry-run and live).
   - Restricted to read-only permissions (`permissions: contents: read`).
   - Has **zero access** to repository signing secrets (`AARIA_MODELS_SIGNING_KEY` is never provided).
   - Downloads current manifest, verifies current signature, downloads Koyal draft files, builds the proposed manifest, enforces all SyncManifest limits, prints the unified diff, and uploads plan artifacts.
   - Computes plan manifest SHA-256 and outputs `manifest_sha256` to `$GITHUB_OUTPUT`.

2. **`apply` job**:
   - Strictly gated to `if: inputs.dry_run == false`.
   - Requires explicit elevation (`permissions: contents: write`) and accesses `AARIA_MODELS_SIGNING_KEY`.
   - Enforces `--expected-sha256` as **REQUIRED and non-empty** from the plan output. Apply NEVER rebuilds the manifest.
   - Never runs during dry runs.

---

## Safe Execution Order & Rollback on Failure

When `dry_run: false`, the `apply` job executes a fail-safe, atomic sequence:

1. **Plan Hash & No-op Verification**:
   - Requires `--expected-sha256` to be provided and match the downloaded plan artifact.
   - If `models.te` in the proposed manifest equals the current release manifest, the workflow prints `No change - nothing to do` and exits cleanly with code 0 without touching any assets or re-signing anything.
2. **Local Signing & Verification**: The new manifest is signed locally with ECDSA P-256, and immediately verified against the release public key (`manifest_public_key.b64`). **Nothing touches the live release before this local signature check passes.**
3. **Exact Asset Upload & SHA Verification**: Exactly four Koyal files (`te-koyal__model.int8.onnx`, `te-koyal__tokens.txt`, `te-koyal__model.json`, `te-koyal__LICENSE-koyal-te-120m.txt`) are uploaded with 3-attempt exponential retries. They are immediately re-downloaded from the release and their SHA-256 checksums verified against the new manifest.
4. **Live-Manifest Race Guard**:
   - Immediately before uploading the new manifest, `apply_switch.py` re-downloads the live `manifest.json` from the release.
   - It asserts that the live manifest is **byte-identical** to the baseline copy the plan started from (`current_rel/manifest.json`).
   - If any concurrent modification occurred, it aborts immediately with a clear error without uploading the new manifest.
5. **Audit Backup**: Current `manifest.json` and `.sig` are archived as `manifest-previous-<UTC timestamp>.json` and `manifest-previous-<UTC timestamp>.json.sig` on the release (no clobber).
6. **Manifest Atomic Update**: New `manifest.json.sig` and `manifest.json` are uploaded together (--clobber) with retries.
7. **Exact After-Check**: The workflow downloads `manifest.json` and `manifest.json.sig` from the public release URL and requires **exact byte-for-byte equality** with the locally signed manifest and signature, plus a passing cryptographic verification.

### Automatic Rollback on Failure
If any failure occurs during the live apply step (network disruption, upload failure, or verification mismatch), the final step triggers automatically (`if: (failure() || cancelled()) && steps.apply_switch.outcome != 'skipped'`):
- First inspects the live release manifest: if the live manifest already matches the saved previous bytes, it skips re-uploading entirely and prints `Live manifest unchanged - nothing to roll back`.
- If rollback is needed, re-uploads the saved previous `manifest.json` and `manifest.json.sig` using retries.
- Re-downloads and cryptographically verifies the restored signature against `manifest_public_key.b64`. **If the public key is missing, rollback strictly fails.**
- Logs clearly: `ROLLED BACK to previous manifest`.

---

## Phone Sync Safety Window

QuietKeep edge devices download both `manifest.json` and `manifest.json.sig` simultaneously during sync and verify:
1. The ECDSA P-256 signature against the pinned public key.
2. The SHA-256 checksum of every downloaded asset against the manifest.

If a phone happens to query the release during the few-second swap window:
- If it sees a signature mismatch or an asset that has not finished uploading, the phone's sync client safely **rejects the manifest update**.
- The phone remains on its currently installed, fully verified model and functions normally.
- The phone simply retries at its next scheduled sync cycle.

---

## SyncManifest Security & Constraint Rules

The builder script (`scripts/switch/build_te_switch.py`) strictly enforces:
- **URL Prefix**: Every model URL must start with `https://github.com/PranixQuick/aaria-edge-models/releases/download/models-2026-10/`.
- **Path Traversal**: Filenames must not contain `..`, `/`, or `\\`.
- **Per-file limit**: Maximum 400 MB per single file (largest model file is ~138 MB).
- **Total model limit**: Sum of all model files across all languages must stay <= 600 MB (current total with Koyal: ~489 MB).
- **Version**: `version: 1`.
- **Integrity**: Every non-target key (`en`, `hi`, `replies`) remains byte-identical.
