#!/usr/bin/env python3
"""
Build new manifest.json for Telugu model switch (Koyal or IndicConformer).

Enforces all SyncManifest rules:
- Manifest version == 1
- HTTPS URLs only
- No path traversal in filenames
- Each single file <= 400 MB
- Total model files <= 600 MB
- Stable key order (json.dumps sort_keys=True, indent=2)
- All non-target languages (en, hi) and replies stay byte-identical

Prints total model size and unified diff against current manifest.
"""

import argparse
import difflib
import hashlib
import json
import sys
from pathlib import Path

BASE_RELEASE_URL = "https://github.com/PranixQuick/aaria-edge-models/releases/download/models-2026-10"
MAX_FILE_BYTES = 400 * 1024 * 1024   # 400 MB
MAX_TOTAL_BYTES = 600 * 1024 * 1024  # 600 MB

# Fallback known hashes and sizes for draft release models-2026-10-koyal-te assets
KNOWN_KOYAL_ASSETS = {
    "model.int8.onnx": {
        "asset_name": "te-koyal__model.int8.onnx",
        "bytes": 138417172,
        "sha256": "8bbb11bd7cfe543148c49d24dba845395a8a683f82a63841ad6b90589e562781",
    },
    "tokens.txt": {
        "asset_name": "te-koyal__tokens.txt",
        "bytes": 5623,
        "sha256": "2f915352a1589f34af9251801804b43c558f429504b9c42c56d2d71d4eafcea0",
    },
    "model.json": {
        "asset_name": "te-koyal__model.json",
        "bytes": 75,
        "sha256": "1d8a8a2be5c31dd8dc0e25c38abe98725227e86d9c8acd015328c0a5106c476b",
    },
    "LICENSE": {
        "asset_name": "te-koyal__LICENSE-koyal-te-120m.txt",
        "bytes": 13076,
        "sha256": "49d48f40a4ecc2bbc126ba54c46cde45f01d71f28d6b56b8932e374cec4deec6",
    },
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
    Construct models.te entry for Koyal te-120M.
    If draft_dir is provided and contains assets, computes bytes and sha256 dynamically.
    """
    files = []
    for role in sorted(KNOWN_KOYAL_ASSETS.keys()):
        meta = KNOWN_KOYAL_ASSETS[role]
        asset_name = meta["asset_name"]

        file_bytes = meta["bytes"]
        file_sha256 = meta["sha256"]

        if draft_dir and draft_dir.is_dir():
            candidates = [draft_dir / asset_name, draft_dir / role]
            found = None
            for cand in candidates:
                if cand.is_file():
                    found = cand
                    break
            if found:
                file_bytes = found.stat().st_size
                file_sha256 = sha256_file(found)

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


def validate_sync_manifest_rules(manifest: dict) -> int:
    """
    Re-check all SyncManifest rules:
    - version 1
    - https only
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
            if ".." in fname or "/" in fname or "\\" in fname:
                raise ValueError(f"Path traversal detected in model {lang} filename: {fname}")
            if not furl.startswith("https://"):
                raise ValueError(f"Non-https URL detected in model {lang} ({fname}): {furl}")
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
) -> dict:
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
        # Try relative to repo root if run from a subdir
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

    total_mb = total_bytes / (1024 * 1024)
    print(f"Target model switch: {args.target}")
    print(f"Total model files size: {total_bytes:,} bytes ({total_mb:.2f} MB / limit 600 MB)")

    print_diff(current_text, new_text)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(new_text, encoding="utf-8")
        print(f"Wrote new manifest to {args.out}")

    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
