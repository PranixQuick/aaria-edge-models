#!/usr/bin/env python3
"""
Build new manifest.json for Telugu model switch (Koyal or IndicConformer).

Enforces all SyncManifest rules:
- Manifest version == 1
- Model file URLs must start with https://github.com/PranixQuick/aaria-edge-models/releases/download/models-2026-10/
- No path traversal in filenames
- Each single file <= 400 MB
- Total model files <= 600 MB
- Stable key order (json.dumps sort_keys=True, indent=2)
- All non-target languages (en, hi) and replies stay byte-identical

Prints total model size and unified diff against current manifest.
Outputs noop=true|false and manifest_sha256 to $GITHUB_OUTPUT.
Writes manifest.json and manifest.json.sha256 companion file.
Always exits 0 on success (no non-zero exit for no-op).
"""

import argparse
import difflib
import hashlib
import json
import os
import sys
from pathlib import Path

BASE_RELEASE_URL = "https://github.com/PranixQuick/aaria-edge-models/releases/download/models-2026-10"
REQUIRED_MODEL_URL_PREFIX = "https://github.com/PranixQuick/aaria-edge-models/releases/download/models-2026-10/"
MAX_FILE_BYTES = 400 * 1024 * 1024   # 400 MB
MAX_TOTAL_BYTES = 600 * 1024 * 1024  # 600 MB

# Exact four Koyal assets required from draft release
KOYAL_REQUIRED_ASSETS = {
    "LICENSE": "te-koyal__LICENSE-koyal-te-120m.txt",
    "model.int8.onnx": "te-koyal__model.int8.onnx",
    "model.json": "te-koyal__model.json",
    "tokens.txt": "te-koyal__tokens.txt",
}


def sha256_file(path: Path) -> str:
    """Compute hex sha256 of file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_koyal_entry(draft_dir: Path = None) -> dict:
    """
    Construct models.te entry for Koyal te-120M using downloaded draft assets.
    Requires exact four files. Fails if draft_dir is missing, if any required file
    is missing, or if unexpected te-koyal__ files are present.
    """
    if draft_dir is None or not draft_dir.is_dir():
        raise FileNotFoundError(
            f"Draft directory containing Koyal assets is required for target 'koyal', got: {draft_dir}"
        )

    expected_names = set(KOYAL_REQUIRED_ASSETS.values())
    found_files = {p.name: p for p in draft_dir.iterdir() if p.is_file()}

    # Check for unexpected te-koyal__ files
    unexpected = sorted([name for name in found_files if name.startswith("te-koyal__") and name not in expected_names])
    if unexpected:
        raise ValueError(
            f"Unexpected te-koyal__ asset(s) found in draft directory {draft_dir}: {unexpected}"
        )

    # Check for missing required files
    missing = sorted([name for name in expected_names if name not in found_files])
    if missing:
        raise FileNotFoundError(
            f"Missing required Koyal draft asset(s) in {draft_dir}: {missing}"
        )

    files = []
    for role in sorted(KOYAL_REQUIRED_ASSETS.keys()):
        asset_name = KOYAL_REQUIRED_ASSETS[role]
        asset_path = found_files[asset_name]
        file_bytes = asset_path.stat().st_size
        file_sha256 = sha256_file(asset_path)

        files.append({
            "bytes": file_bytes,
            "name": role,
            "sha256": file_sha256,
            "url": f"{BASE_RELEASE_URL}/{asset_name}",
        })

    return {
        "files": files,
        "type": "ctc",
    }


def build_indicconformer_entry(stored_path: Path) -> dict:
    """Load stored IndicConformer entry byte-for-byte."""
    if not stored_path.is_file():
        raise FileNotFoundError(f"Stored IndicConformer file not found: {stored_path}")
    data = json.loads(stored_path.read_text(encoding="utf-8"))
    return data


def check_is_noop(current_manifest: dict, new_manifest: dict) -> bool:
    """Check if models.te in new manifest is identical to current manifest."""
    return current_manifest.get("models", {}).get("te") == new_manifest.get("models", {}).get("te")


def validate_sync_manifest_rules(manifest: dict) -> int:
    """
    Re-check all SyncManifest rules:
    - version 1
    - all model URLs start with REQUIRED_MODEL_URL_PREFIX
    - no path traversal
    - each file <= 400 MB
    - total model files <= 600 MB
    Returns total model bytes.
    """
    if manifest.get("version") != 1:
        raise ValueError(f"Manifest version must be 1, got {manifest.get('version')}")

    total_model_bytes = 0

    models = manifest.get("models", {})
    if not models:
        raise ValueError("Manifest models map is empty")

    for lang, m in models.items():
        if "type" not in m:
            raise ValueError(f"Model {lang} missing 'type'")
        files = m.get("files", [])
        if not files:
            raise ValueError(f"Model {lang} has no files")

        for f in files:
            fname = f.get("name", "")
            furl = f.get("url", "")
            fbytes = f.get("bytes", 0)
            fsha = f.get("sha256", "")

            if not fname:
                raise ValueError(f"Empty filename in model {lang}")
            if ".." in fname or "/" in fname or chr(92) in fname:
                raise ValueError(f"Path traversal detected in model {lang} filename: {fname}")
            if not furl.startswith(REQUIRED_MODEL_URL_PREFIX):
                raise ValueError(
                    f"Model URL in model {lang} ({fname}) does not start with required prefix "
                    f"'{REQUIRED_MODEL_URL_PREFIX}': {furl}"
                )
            if not fsha or len(fsha) != 64:
                raise ValueError(f"Invalid sha256 in model {lang} ({fname}): {fsha}")
            if fbytes > MAX_FILE_BYTES:
                raise ValueError(
                    f"File {fname} in model {lang} ({fbytes} bytes) exceeds 400 MB limit ({MAX_FILE_BYTES} bytes)"
                )
            total_model_bytes += fbytes

    replies = manifest.get("replies", {})
    for voice_id, lang_dict in replies.items():
        for lang, items in lang_dict.items():
            for item in items:
                furl = item.get("url", "")
                if not furl.startswith("https://"):
                    raise ValueError(f"Non-https URL detected in replies ({voice_id}/{lang}): {furl}")

    if total_model_bytes > MAX_TOTAL_BYTES:
        raise ValueError(
            f"Total model files size ({total_model_bytes} bytes) exceeds 600 MB limit ({MAX_TOTAL_BYTES} bytes)"
        )

    return total_model_bytes


def build_switch_manifest(
    current_manifest: dict,
    target: str,
    stored_indicconformer_path: Path,
    draft_dir: Path = None,
) -> tuple:
    """
    Construct new manifest dictionary with switched models.te.
    All other keys stay byte-identical.
    """
    if target not in ("koyal", "indicconformer"):
        raise ValueError(f"Target must be 'koyal' or 'indicconformer', got: {target}")

    # Deep copy or construct new dictionary preserving exact values
    new_manifest = {
        "models": {},
        "release": current_manifest["release"],
        "replies": current_manifest["replies"],
        "version": current_manifest["version"],
    }

    # Copy existing models
    for lang, m in current_manifest["models"].items():
        if lang != "te":
            new_manifest["models"][lang] = m

    # Replace 'te'
    if target == "koyal":
        new_manifest["models"]["te"] = build_koyal_entry(draft_dir)
    else:
        new_manifest["models"]["te"] = build_indicconformer_entry(stored_indicconformer_path)

    # Re-check all SyncManifest rules
    total_bytes = validate_sync_manifest_rules(new_manifest)
    return new_manifest, total_bytes


def print_diff(current_json_str: str, new_json_str: str):
    """Print unified diff between current and new manifest."""
    diff_lines = list(difflib.unified_diff(
        current_json_str.splitlines(keepends=True),
        new_json_str.splitlines(keepends=True),
        fromfile="current/manifest.json",
        tofile="new/manifest.json",
    ))
    print("\n=== MANIFEST DIFF ===")
    if not diff_lines:
        print("(No changes - manifests are identical)")
    else:
        for line in diff_lines:
            sys.stdout.write(line)
    print("=====================\n")


def write_github_output(noop: bool, sha256_str: str):
    """Write noop and manifest_sha256 to $GITHUB_OUTPUT if present."""
    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        try:
            with open(gh_output, "a", encoding="utf-8") as f:
                f.write(f"noop={'true' if noop else 'false'}\n")
                f.write(f"manifest_sha256={sha256_str}\n")
        except Exception as e:
            print(f"::warning::Failed to write to GITHUB_OUTPUT: {e}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", required=True, type=Path, help="Path to current manifest.json")
    parser.add_argument("--target", required=True, choices=["koyal", "indicconformer"], help="Switch target")
    parser.add_argument("--draft-dir", type=Path, default=None, help="Directory containing draft release files")
    parser.add_argument(
        "--stored-indicconformer",
        type=Path,
        default=Path("manifests/te-indicconformer.json"),
        help="Path to committed te-indicconformer.json",
    )
    parser.add_argument("--out", type=Path, default=Path("manifest.json"), help="Output manifest.json path")
    args = parser.parse_args()

    if not args.current.is_file():
        print(f"::error::Current manifest not found: {args.current}", file=sys.stderr)
        sys.exit(1)

    current_text = args.current.read_text(encoding="utf-8")
    current_manifest = json.loads(current_text)

    # Locate stored indicconformer if path is relative
    stored_path = args.stored_indicconformer
    if not stored_path.is_file():
        alt = Path(__file__).resolve().parent.parent.parent / "manifests" / "te-indicconformer.json"
        if alt.is_file():
            stored_path = alt

    try:
        new_manifest, total_bytes = build_switch_manifest(
            current_manifest,
            args.target,
            stored_path,
            args.draft_dir,
        )
    except Exception as e:
        print(f"::error::Failed to build switch manifest: {e}", file=sys.stderr)
        sys.exit(1)

    new_text = json.dumps(new_manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    new_bytes = new_text.encode("utf-8")
    new_sha256 = hashlib.sha256(new_bytes).hexdigest()

    total_mb = total_bytes / (1024 * 1024)
    print(f"Target model switch: {args.target}")
    print(f"Total model files size: {total_bytes:,} bytes ({total_mb:.2f} MB / limit 600 MB)")
    print(f"Built manifest SHA-256: {new_sha256}")

    print_diff(current_text, new_text)

    is_noop = check_is_noop(current_manifest, new_manifest)

    # Write output manifest and companion .sha256 file
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_bytes(new_bytes)
        sha_file = args.out.parent / f"{args.out.name}.sha256"
        sha_file.write_text(f"{new_sha256}  {args.out.name}\n", encoding="utf-8")
        print(f"Wrote manifest to {args.out}")
        print(f"Wrote manifest digest to {sha_file}")

    write_github_output(is_noop, new_sha256)

    if is_noop:
        print("No change - nothing to do")
        return 0

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
